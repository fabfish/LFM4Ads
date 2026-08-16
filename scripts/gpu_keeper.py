#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_keeper.py — 多卡 GPU 占位守护（fake load keeper，按 GK_GPU_ID 区分）

背景
----
在共享/租用 GPU 上，长时间空闲可能被调度器回收。这个小程序在 GPU0 空闲足够久后，
启动一个只占显存（默认零计算利用率、不发热不伤卡）的"占位程序"来保持卡被"占用"的状态；
一旦有真实 GPU 任务挂上来，立即让位（杀掉占位程序）并恢复正常监测。

行为（默认）
----------
  1. 监测 GPU0：若连续 IDLE_THRESHOLD 秒（默认 1200s = 20min）没有任何真实计算进程
     占用，且利用率很低，判定为"空闲"。
  2. 空闲达到阈值后，启动一个占位程序（默认只占显存、计算核空闲，见 GK_HOLD_MEMORY_MB），
     运行 FAKE_DURATION 秒（默认 1200s = 20min；设为 0 表示一直跑到有真实任务出现为止）。
  3. 假程序运行期间，若检测到 GPU0 上出现"其他"计算进程（真实任务挂上），
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
  GK_IDLE_THRESHOLD_S 空闲判定时长（秒），默认 60（空 1 分钟才启动，避免抖动误判）
  GK_FAKE_DURATION_S  假程序运行时长（秒），默认 1200（跑满 20 分钟）；0 = 跑到真任务出现
  GK_REST_S           每轮假程序结束后的冷却间隔（秒），默认 1200（歇 20 分钟再下一轮）
  GK_POLL_S           轮询间隔（秒），默认 10
  GK_HOLD_MEMORY_MB   占位显存大小（MiB），默认 2048
  GK_TARGET_UTIL      目标 GPU 利用率（百分比，近似），默认 25（实测略低于标称，确保计算核稳定 >=20%）
"""
import os
import sys
import time
import signal
import fcntl
import logging
import subprocess

# 单实例锁与 pidfile（按 GPU 隔离，避免多卡守护互相冲突）
_HERE = os.path.dirname(os.path.abspath(__file__))
GPU_ID = int(os.environ.get("GK_GPU_ID", "0"))
_PID_NAME = "gpu_keeper.pid" if GPU_ID == 0 else f"gpu_keeper_gpu{GPU_ID}.pid"
PIDFILE = os.path.normpath(os.path.join(_HERE, "..", "logs", _PID_NAME))

TRIGGER = os.environ.get("GK_TRIGGER", "idle").lower()
IDLE_THRESHOLD_S = int(os.environ.get("GK_IDLE_THRESHOLD_S", "60"))
FAKE_DURATION_S = int(os.environ.get("GK_FAKE_DURATION_S", "1200"))
REST_S = int(os.environ.get("GK_REST_S", "1200"))  # 每次假程序结束后歇多久再下一轮（秒）
POLL_S = int(os.environ.get("GK_POLL_S", "10"))
HOLD_MEMORY_MB = int(os.environ.get("GK_HOLD_MEMORY_MB", "2048"))
TARGET_UTIL = float(os.environ.get("GK_TARGET_UTIL", "25"))

# 运行期状态。注意：容器环境下 nvidia-smi 报告的 pid 与 Python 子进程 pid（宿主机命名空间）
# 处于不同命名空间，二者对不上，因此不能用 _fake_proc.pid 去排除假程序自己。
# 改为：启动假程序时"差分"抓取它在 nvidia-smi 中暴露的真实 GPU pid，检测"真任务"时排除之。
_fake_proc = None
_fake_gpu_pids = set()    # 假程序在 nvidia-smi 中暴露的（GPU 命名空间）pid 集合
_fake_cleared_at = 0.0    # 上次停止假程序的时间戳（用于短暂宽限，避免残影误判为真任务）

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


def excluded_pids():
    """当前应被当作"自己人"的 GPU pid 集合（即我们的假程序）。"""
    if _fake_proc is not None:
        return set(_fake_gpu_pids)
    # 刚停止的短暂宽限期内，仍把上一批假程序 pid 视为自己人，避免残影误判为真任务
    if _fake_gpu_pids and (time.time() - _fake_cleared_at) < 10:
        return set(_fake_gpu_pids)
    return set()


def real_tasks():
    """返回 (是否有真实任务, 真实任务 pid 集合)。真实任务 = 出现在 GPU 上、且不是我们假程序的进程。"""
    pids = compute_pids()
    reals = pids - excluded_pids()
    return bool(reals), reals


def capture_fake_pids(baseline):
    """假程序启动后，差分抓取它在 nvidia-smi 中暴露的（GPU 命名空间）pid。"""
    for _ in range(20):
        newp = compute_pids() - baseline
        if newp:
            return newp
        time.sleep(0.5)
    return set()


def launch_fake():
    baseline = compute_pids()
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--fake", "--gpu", str(GPU_ID), "--target-util", str(TARGET_UTIL),
        "--hold-mem", str(HOLD_MEMORY_MB),
    ]
    p = subprocess.Popen(cmd)
    global _fake_gpu_pids
    _fake_gpu_pids = capture_fake_pids(baseline)
    log.info("已启动占位程序（宿主 pid=%s，GPU pid=%s，占位显存 ~%d MiB，目标利用率 ~%.0f%%）",
             p.pid, ",".join(map(str, _fake_gpu_pids)) or "?", HOLD_MEMORY_MB, TARGET_UTIL)
    return p


def stop_fake(p):
    global _fake_cleared_at
    if p is None:
        return
    # 先记录清理时间，再终止，保证宽限期内不会把残影当成真任务
    _fake_cleared_at = time.time()
    if p.poll() is None:
        log.info("正在终止假计算程序（宿主 pid=%s）", p.pid)
        try:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
    else:
        log.info("假计算程序（宿主 pid=%s）已自行退出", p.pid)


def fake_worker(gpu_id, target_util, hold_memory_mb):
    """占位程序本体。

    - 先分配 hold_memory_mb 大小的显存并持有引用（nvidia-smi 进程表里能看到该进程占显存）。
    - target_util > 0（默认 25）：以该占空比做小矩阵乘「心跳」，把计算核利用率维持住，
      用于必须按「计算利用率」判定占用/回收的调度器。实测采样略低于标称，故默认上浮到 25。
    - target_util <= 0：纯显存占位，计算核空闲，功耗≈待机，用于只看显存的调度器。
    """
    import torch  # 延迟导入，监控主进程不依赖 torch

    dev = f"cuda:{gpu_id}"

    # 1) 显存占位：真正触碰显存，确保分配落地并被驱动登记
    numel = max(1, int(hold_memory_mb) * 1024 * 1024 // 4)
    _hold = torch.zeros(numel, dtype=torch.float32, device=dev)  # 持引用，防 GC 回收
    torch.cuda.synchronize()
    log.info("显存占位: %s 持有 ~%d MiB（%d 个 float32）", dev, hold_memory_mb, numel)

    if target_util <= 0:
        log.info("目标利用率 0%%：纯显存占位，不制造计算负载，等待 SIGTERM")
        while True:
            time.sleep(3600)

    # 2) 心跳负载（可选）：小矩阵乘 + 低占空比，仅维持极低利用率
    N = 4096  # 足够大以压过 Python 循环开销，保证占空比可控
    a = torch.randn(N, N, device=dev)
    b = torch.randn(N, N, device=dev)
    duty = max(0.0001, min(1.0, target_util / 100.0))

    log.info("心跳负载在 %s 上运行，目标利用率 ~%.1f%%", dev, target_util)
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
    # ---- 单实例保护：flock 保证同一时刻只有一个监测进程 ----
    # 启动脚本里 `setsid ... &; echo $!` 写的是 setsid 的 pid，不等于本进程 pid，
    # 单靠 pidfile 易被 cron 看门狗再次拉起造成"双实例互殴"。这里由本进程自己持锁。
    try:
        _pidf = open(PIDFILE, "w")
        fcntl.flock(_pidf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.error("另一实例已在运行（无法获取 %s 锁），退出", PIDFILE)
        sys.exit(1)
    _pidf.write(str(os.getpid()))
    _pidf.flush()
    # --------------------------------------------------------

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    log.info(
        "监测启动: gpu=%s trigger=%s 空闲阈值=%.0fs 假程序时长=%.0fs 冷却间隔=%.0fs 占位显存=%dMiB 目标利用率=%.0f%%",
        GPU_ID, TRIGGER, IDLE_THRESHOLD_S, FAKE_DURATION_S, REST_S, HOLD_MEMORY_MB, TARGET_UTIL,
    )

    # 状态机：monitor（等待空闲阈值） -> fake（运行） -> rest（冷却） -> monitor ...
    state = "monitor"
    idle_since = None
    fake_start = 0.0
    rest_until = 0.0

    while True:
        now = time.time()
        busy, real_pids = real_tasks()  # busy=有非本守护假程序的 GPU 进程

        if state == "fake":
            elapsed = now - fake_start
            if busy:
                # 出现"其他"GPU 进程（真实任务）→ 立刻让位
                log.info("检测到真实 GPU 任务（pids=%s），停止假程序并让位", sorted(real_pids))
                stop_fake(_fake_proc)
                _fake_proc = None
                state = "monitor"
                idle_since = None
            elif FAKE_DURATION_S <= 0 or elapsed >= FAKE_DURATION_S:
                # 跑满时长（或 0=持续模式仅被真任务打断）→ 进入冷却
                if FAKE_DURATION_S > 0:
                    log.info("假程序已运行 %.0fs，到期停止，进入冷却 %.0fs", elapsed, REST_S)
                stop_fake(_fake_proc)
                _fake_proc = None
                state = "rest"
                rest_until = now + REST_S
            time.sleep(POLL_S)

        elif state == "rest":
            # 冷却期：真任务来了就让位（用完再恢复监测）；否则歇满 REST_S 后下一轮
            if busy:
                log.info("冷却期出现真实 GPU 任务（pids=%s），让位并恢复监测", sorted(real_pids))
                state = "monitor"
                idle_since = None
            elif now >= rest_until:
                # 冷却结束且 GPU 仍空闲 → 直接下一轮跑满（无需再等空闲阈值）
                if not busy:
                    _fake_proc = launch_fake()
                    fake_start = now
                    state = "fake"
                else:
                    state = "monitor"
                    idle_since = None
            time.sleep(POLL_S)

        else:  # monitor
            if TRIGGER == "busy":
                if busy:
                    idle_since = idle_since or now
                    if now - idle_since >= IDLE_THRESHOLD_S:
                        _fake_proc = launch_fake()
                        fake_start = now
                        state = "fake"
                        idle_since = None
                else:
                    idle_since = None
            else:
                if busy:
                    idle_since = None
                else:
                    if idle_since is None:
                        idle_since = now
                    idle_for = now - idle_since
                    if idle_for >= IDLE_THRESHOLD_S:
                        _fake_proc = launch_fake()
                        fake_start = now
                        state = "fake"
                        idle_since = None
            time.sleep(POLL_S)


def main():
    if "--fake" in sys.argv:
        # 占位程序子进程
        gi = sys.argv
        gpu_arg = gi[gi.index("--gpu") + 1] if "--gpu" in gi else GPU_ID
        util_arg = gi[gi.index("--target-util") + 1] if "--target-util" in gi else TARGET_UTIL
        hold_arg = gi[gi.index("--hold-mem") + 1] if "--hold-mem" in gi else HOLD_MEMORY_MB
        fake_worker(int(gpu_arg), float(util_arg), int(hold_arg))
    else:
        main_monitor()


if __name__ == "__main__":
    main()
