#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_keeper.py — GPU0 占位守护（fake load keeper）

背景
----
在共享/租用 GPU 上，长时间空闲可能被调度器回收。这个小程序在 GPU0 空闲足够久后，
启动一个只占用极低（默认 10%）GPU 利用率的“假计算程序”来保持卡被“占用”的状态；
一旦有真实 GPU 任务挂上来，立即让位（杀掉假程序）并恢复正常监测。

行为（默认）
----------
  1. 监测 GPU0：若连续 IDLE_THRESHOLD 秒（默认 1200s = 20min）没有任何真实计算进程
     占用，且利用率很低，判定为“空闲”。
  2. 空闲达到阈值后，启动一个只占 ~TARGET_UTIL 利用率的假计算程序，运行
     FAKE_DURATION 秒（默认 1200s = 20min；设为 0 表示一直跑到有真实任务出现为止）。
  3. 假程序运行期间，若检测到 GPU0 上出现“其他”计算进程（真实任务挂上），
     立即杀掉假程序，恢复空闲监测。

启动（无 systemd 时，长驻后台）
--------------------------------
  setsid nohup python3 scripts/gpu_keeper.py >> logs/gpu_keeper.log 2>&1 &

更方便：scripts/start_gpu_keeper.sh / scripts/stop_gpu_keeper.sh

配置（环境变量，均为可选）
-------------------------
  GK_GPU_ID           物理 GPU 序号，默认 0
  GK_TRIGGER          idle（默认）| busy
                      idle = 空闲这么久后启动；busy = 被真实任务占用这么久后启动
  GK_IDLE_THRESHOLD_S 空闲/占用判定时长（秒），默认 1200
  GK_FAKE_DURATION_S  假程序运行时长（秒），默认 1200；0 = 跑到真任务出现
  GK_POLL_S           轮询间隔（秒），默认 10
  GK_TARGET_UTIL      目标 GPU 利用率（百分比，近似），默认 10
"""
import os
import sys
import time
import signal
import logging
import subprocess

GPU_ID = int(os.environ.get("GK_GPU_ID", "0"))
TRIGGER = os.environ.get("GK_TRIGGER", "idle").lower()
IDLE_THRESHOLD_S = int(os.environ.get("GK_IDLE_THRESHOLD_S", "1200"))
FAKE_DURATION_S = int(os.environ.get("GK_FAKE_DURATION_S", "1200"))
POLL_S = int(os.environ.get("GK_POLL_S", "10"))
TARGET_UTIL = float(os.environ.get("GK_TARGET_UTIL", "10"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gpu-keeper] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gpu-keeper")

# 全局句柄，便于信号处理时清理
_fake_proc = None


def _nvidia_smi(extra_args):
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(GPU_ID)] + extra_args,
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout
    except Exception as e:  # noqa: BLE001
        log.warning("nvidia-smi 调用失败: %s", e)
        return ""


def compute_pids(exclude=()):
    """返回当前在 GPU0 上运行计算任务的进程 pid 集合。"""
    txt = _nvidia_smi(["--query-compute-apps=pid", "--format=csv,noheader"])
    pids = set()
    for line in txt.strip().splitlines():
        line = line.strip()
        if line:
            try:
                pids.add(int(line))
            except ValueError:
                pass
    return pids - set(exclude)


def gpu_util():
    txt = _nvidia_smi(["--query-gpu=utilization.gpu", "--format=csv,noheader"])
    for line in txt.strip().splitlines():
        line = line.strip().replace("%", "")
        try:
            return int(line)
        except ValueError:
            pass
    return -1


def is_busy(exclude=()):
    """GPU0 是否正被（除 exclude 之外的）进程占用。"""
    pids = compute_pids(exclude=exclude)
    if pids:
        return True, pids
    util = gpu_util()
    if util > 5:
        return True, set()
    return False, set()


def launch_fake():
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--fake", "--gpu", str(GPU_ID), "--target-util", str(TARGET_UTIL),
    ]
    p = subprocess.Popen(cmd)
    log.info("已启动假计算程序 pid=%s（目标利用率 ~%.0f%%）", p.pid, TARGET_UTIL)
    return p


def stop_fake(p):
    if p is None:
        return
    if p.poll() is None:
        log.info("正在终止假计算程序 pid=%s", p.pid)
        try:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
    else:
        log.info("假计算程序 pid=%s 已自行退出", p.pid)


def fake_worker(gpu_id, target_util):
    """假计算程序本体：以 ~target_util 的占空比做小矩阵乘法，制造低利用率负载。"""
    import torch  # 延迟导入，监控主进程不依赖 torch

    dev = f"cuda:{gpu_id}"
    N = 4096  # 足够大以压过 Python 循环开销，保证占空比可控
    a = torch.randn(N, N, device=dev)
    b = torch.randn(N, N, device=dev)
    duty = max(0.01, min(0.99, target_util / 100.0))

    log.info("假计算程序在 %s 上运行，目标利用率 ~%.0f%%", dev, target_util)
    c = a
    while True:
        t0 = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        dt = time.time() - t0
        sleep_t = dt * (1.0 / duty - 1.0)
        if sleep_t > 0:
            time.sleep(sleep_t)
        # 制造数据依赖，避免被任何惰性求值优化掉
        if (int(time.time() * 10) % 1000) == 0:
            a.copy_(c)


def _handle_term(signum, frame):
    log.info("收到信号 %s，清理退出", signum)
    stop_fake(_fake_proc)
    sys.exit(0)


def main_monitor():
    global _fake_proc
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    log.info(
        "监测启动: gpu=%s trigger=%s 阈值=%.0fs 假程序时长=%.0fs 目标利用率=%.0f%%",
        GPU_ID, TRIGGER, IDLE_THRESHOLD_S, FAKE_DURATION_S, TARGET_UTIL,
    )

    idle_since = None
    fake_start = 0.0

    while True:
        now = time.time()
        busy, real_pids = is_busy(exclude=(_fake_proc.pid,) if _fake_proc else ())

        if _fake_proc is None:
            # ---- 监测态 ----
            if TRIGGER == "busy":
                # busy 模式：被真实任务持续占用达到阈值后，再叠加假程序
                if busy and real_pids:
                    idle_since = idle_since or now
                    if now - idle_since >= IDLE_THRESHOLD_S:
                        _fake_proc = launch_fake()
                        fake_start = now
                        idle_since = None
                else:
                    idle_since = None
            else:
                # idle 模式（默认）：空闲达到阈值后启动假程序
                if busy:
                    idle_since = None
                else:
                    if idle_since is None:
                        idle_since = now
                    idle_for = now - idle_since
                    if idle_for >= IDLE_THRESHOLD_S:
                        _fake_proc = launch_fake()
                        fake_start = now
                        idle_since = None
            time.sleep(POLL_S)
        else:
            # ---- 假程序运行态 ----
            # 出现“其他”GPU 进程（真实任务）→ 立刻让位
            real_now, _ = is_busy(exclude=(_fake_proc.pid,))
            elapsed = now - fake_start
            if real_now:
                log.info("检测到真实 GPU 任务（pids=%s），停止假程序并让位", compute_pids(exclude=(_fake_proc.pid,)))
                stop_fake(_fake_proc)
                _fake_proc = None
                idle_since = None
            elif FAKE_DURATION_S > 0 and elapsed >= FAKE_DURATION_S:
                log.info("假程序已运行 %.0fs，到期停止", elapsed)
                stop_fake(_fake_proc)
                _fake_proc = None
                idle_since = None
            time.sleep(POLL_S)


def main():
    if "--fake" in sys.argv:
        # 假计算程序子进程
        gi = sys.argv
        gpu_arg = gi[gi.index("--gpu") + 1] if "--gpu" in gi else GPU_ID
        util_arg = gi[gi.index("--target-util") + 1] if "--target-util" in gi else TARGET_UTIL
        fake_worker(int(gpu_arg), float(util_arg))
    else:
        main_monitor()


if __name__ == "__main__":
    main()
