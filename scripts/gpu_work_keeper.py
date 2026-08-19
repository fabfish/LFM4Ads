#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_work_keeper.py — 真实任务占卡守护（work keeper）

政策背景（2026-08-19）
----------------------
禁用假负载占卡（原 scripts/gpu_keeper.py 的 ``--fake`` 矩阵乘心跳）。GPU 占用只能来自
真实、有用的实验任务。本守护把「占卡」替换为「持续跑真实实验任务」：

    空闲 GPU → 派一个真实训练任务（满利用率占卡 + 产出 run json）
              → 跑完自动续下一个 → 任务表耗尽即自然退出（无真实任务则不占卡）。

设计
----
1. 任务表复用 scripts/matrix/run_macro_auc_matrix.py 的预注册 stage（``build_tasks``），
   保证只跑「已预注册、有授权」的真实实验，绝不为了占卡乱造配置。
2. 每张卡一个 worker，串行跑该卡的真实任务（subprocess 阻塞等待跑完）。
3. 断点续跑：``run_<tag>.json`` 已存在则跳过，可随时 kill/重启。
4. 失败隔离：单个 run 崩溃记录后继续下一个，不拖垮守护。
5. 空闲自检：派活前用 nvidia-smi 确认该卡无其他计算进程，不抢占手动任务、不叠卡。
6. 任务表跑完即打印总结并退出。

用法
----
    # 占 cuda:0/cuda:1，跑预注册主线 stage（s1 2x2 + s2sent 哨兵）
    nohup python scripts/gpu_work_keeper.py \
        --devices cuda:0,cuda:1 --stages s1,s2sent \
        > logs/gpu_work_keeper.log 2>&1 &

    # dry-run：只打印任务分配，不派活
    python scripts/gpu_work_keeper.py --devices cuda:0,cuda:1 --stages s1,s2sent --dry-run

守卡模式（跑真实任务占卡，但输出到隔离目录，不污染真实结论 / results/INDEX）：
    # 用 27K（显存 61GB/卡）+ 输出到 cache/keepalive/，跑一个已验证的干净任务
    LFM_DATASET=$PWD/dataset_27k.feather LFM_VOCAB_JSON=$PWD/cache/fields_27k.json \
    LFM_SAMPLE_COUNTS_JSON=$PWD/cache/sample_counts_27k.json LFM_SITE=A \
    python scripts/gpu_work_keeper.py --output-dir cache/keepalive \
        --devices cuda:0,cuda:1 --stages s2sent

    --output-dir 会把任务产物 run_*.json 与日志写到该目录（logs/ 派生），
    且给子进程注入 LFM_MACRO_OUT=<output-dir>，让真实训练任务也写到这里。
    要重新守卡时 rm -rf cache/keepalive logs/keepalive 即可（结果随时可丢弃）。

环境变量（与矩阵 runner 相同，缺一会静默用错数据集或直接 S1 FAILED）：
    LFM_DATASET / LFM_VOCAB_JSON / LFM_SAMPLE_COUNTS_JSON / LFM_MACRO_OUT / LFM_SITE
"""
import argparse
import os
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# 复用矩阵 runner 的预注册任务表（真实、有授权的实验任务）
sys.path.insert(0, os.path.join(_HERE, "matrix"))
import run_macro_auc_matrix as rmm  # noqa: E402

OUT_DIR = rmm.OUT_DIR          # 由 LFM_MACRO_OUT 决定，与矩阵 runner 完全一致
LOG_DIR = rmm.LOG_DIR          # logs/<实验>，与矩阵 runner 一致
PRIMARY_STAGES = ",".join(rmm.PRIMARY_STAGES)
_OVERRIDE_OUT_DIR = None       # --output-dir 守卡覆盖：None = 用 LFM_MACRO_OUT


def _gpu_index(device):
    """``cuda:0`` -> ``0``。默认未设 CUDA_VISIBLE_DEVICES 时与 nvidia-smi 序号一致。"""
    return int(str(device).split(":")[-1])


def gpu_compute_pids(gpu_id):
    """返回物理 GPU gpu_id 上的计算进程 pid 集合（空 = 空闲）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"[work-keeper] nvidia-smi 失败: {e}", flush=True)
        return None
    pids = set()
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if line and line.isdigit():
            pids.add(int(line))
    return pids


def wait_idle(gpu_id, poll_s):
    """阻塞直到该卡无计算进程（空闲）。返回 False 表示查询失败，跳过自检直接派。"""
    while True:
        pids = gpu_compute_pids(gpu_id)
        if pids is None:
            return False  # 查不到就放行，避免因查询失败卡死守护
        if not pids:
            return True
        print(f"[work-keeper] gpu{gpu_id} 忙（pids={sorted(pids)}），等待 {poll_s}s...",
              flush=True)
        time.sleep(poll_s)


def worker(gpu_id, device, my_tasks, poll_s):
    for idx, task in enumerate(my_tasks, 1):
        tag = task["tag"]
        out_json = os.path.join(OUT_DIR, f"run_{tag}.json")
        n = len(my_tasks)
        if os.path.exists(out_json):
            print(f"[{device}] ({idx}/{n}) skip {tag} (done)", flush=True)
            continue
        # 占卡核心：卡空闲才派真实任务，不抢占、不叠卡
        wait_idle(gpu_id, poll_s)
        log_file = os.path.join(LOG_DIR, f"{tag}.log")
        os.makedirs(LOG_DIR, exist_ok=True)
        t0 = time.time()
        print(f"[{device}] ({idx}/{n}) RUN {tag}", flush=True)
        # 守卡覆盖：把 LFM_MACRO_OUT 注入子进程，真实训练任务输出到隔离目录
        env = os.environ.copy()
        if _OVERRIDE_OUT_DIR:
            env["LFM_MACRO_OUT"] = _OVERRIDE_OUT_DIR
        try:
            with open(log_file, "w") as lf:
                rc = subprocess.call(task["cmd"], cwd=_ROOT,
                                     stdout=lf, stderr=subprocess.STDOUT, env=env)
        except Exception as e:  # noqa: BLE001
            rc, err = -99, repr(e)
        else:
            err = None
        wall = (time.time() - t0) / 60.0
        ok = rc == 0 and os.path.exists(out_json)
        status = "OK" if ok else f"FAIL rc={rc}"
        if err:
            status += f" exc={err}"
        print(f"[{device}] ({idx}/{n}) {status} {tag} {wall:.1f}min "
              f"(log={log_file})", flush=True)


def main():
    global OUT_DIR, LOG_DIR, _OVERRIDE_OUT_DIR
    ap = argparse.ArgumentParser(description="真实任务占卡守护（跑预注册真实实验，不跑假负载）")
    ap.add_argument("--stages", default=PRIMARY_STAGES,
                    help="预注册 stage，逗号分隔（复用矩阵 runner 的任务表）")
    ap.add_argument("--devices", default="cuda:0,cuda:1",
                    help="要占的卡，逗号分隔")
    ap.add_argument("--output-dir", default=None,
                    help="守卡输出目录（隔离）。指定后产物与日志写到这里，"
                         "并注入子进程 LFM_MACRO_OUT；不污染真实结果")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--poll-s", type=int, default=30, help="空闲轮询间隔（秒）")
    ap.add_argument("--dry-run", action="store_true", help="只打印任务分配，不派活")
    args = ap.parse_args()

    if args.output_dir:
        _OVERRIDE_OUT_DIR = os.path.abspath(args.output_dir)
        OUT_DIR = _OVERRIDE_OUT_DIR
        LOG_DIR = _OVERRIDE_OUT_DIR.replace("cache", "logs", 1)

    stages = tuple(s for s in args.stages.split(",") if s)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    tasks = rmm.build_tasks(stages=stages, epochs=args.epochs,
                            patience=args.patience)
    tasks = [t for t in tasks if t["device"] in devices]

    by_device = {}
    for t in tasks:
        by_device.setdefault(t["device"], []).append(t)

    pending = [t for t in tasks
               if not os.path.exists(os.path.join(OUT_DIR, f"run_{t['tag']}.json"))]

    print(f"[work-keeper] 任务表={args.stages} 设备={devices} "
          f"OUT_DIR={OUT_DIR} 总任务={len(tasks)} 待跑={len(pending)}", flush=True)
    for d, ts in by_device.items():
        done = sum(1 for t in ts
                   if os.path.exists(os.path.join(OUT_DIR, f"run_{t['tag']}.json")))
        print(f"  {d}: {len(ts)} 任务（已完成 {done}，待跑 {len(ts) - done}）",
              flush=True)

    if args.dry_run:
        for d, ts in by_device.items():
            print(f"\n{d} ({len(ts)} runs):")
            for t in ts:
                mark = "DONE" if os.path.exists(
                    os.path.join(OUT_DIR, f"run_{t['tag']}.json")) else "TODO"
                print(f"  [{mark}] {t['tag']}")
        return

    if not pending:
        print("[work-keeper] 无待跑真实任务（全部已 done），不占卡，退出", flush=True)
        return

    threads = [threading.Thread(target=worker,
                                args=(_gpu_index(d), d, ts, args.poll_s),
                                daemon=False)
               for d, ts in by_device.items()]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("[work-keeper] 任务表跑完，守护退出（不再占卡，符合「无真实任务不占卡」）",
          flush=True)


if __name__ == "__main__":
    main()
