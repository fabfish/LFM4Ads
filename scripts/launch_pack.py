#!/usr/bin/env python3
"""launch_pack.py — 多 job 并发打包器（双卡 GPU 容量填充）。

设计目标
--------
把互相独立、方法学上允许并行的实验 job 打包到空闲 GPU 容量上，提升
GPU 利用率（当前单 job 仅 ~4GB / <40% util，卡有 ~94GB 富余）。

特性
----
* round-robin 分配 device 列表（默认 cuda:0, cuda:1）。
* 每卡并发上限可配（--max-per-device，默认 3）——单 job 属 data-bound，
  塞 3 个即可把 GPU 打满，且显存 12GB ≪ 98GB。
* 后台启动（setsid + nohup），输出重定向到 cache/<tag>.log。
* 并发 job 自动设 LFM_NUM_WORKERS=4，防止 384 核被 worker 扇出打爆
  （独立 job 仍可显式 --num-workers 覆盖）。
* 轻量 monitor：周期性打印 nvidia-smi 利用率 + 各 log 尾部，不刷屏。

重要方法学约束
--------------
本工具只负责"把独立 job 并发跑起来"。它**不**保证可比性：
* 同一研究课题内的横向变体（如 D9-A 冻结强度 full / freeze-dnn-head /
  rx-only）必须**单卡独占、串行**（见 AGENTS.md §2、DRIVERS.md §4），
  不要放进同一个 launch_pack 调用里并发混跑——应改用链式调度器
  （cache/d20_chain.sh 模式）等 rx-only 结束后补跑。
* 这里并发的 job 必须是**不同研究课题**（如 D6 MoE V2 shared 验证 与
  backbone 选择标准），它们之间不存在需要控制的变量。

用法
----
python scripts/launch_pack.py \
    --devices cuda:0 cuda:1 \
    --max-per-device 3 \
    --monitor-interval 120 \
    -- \
    "D6_v2_shared_s1|experiments/main_moe_v2.py|--shuffle --seed 1 --tag d6_v2_shared_s1" \
    "D6_v2_shared_s7|experiments/main_moe_v2.py|--shuffle --seed 7 --tag d6_v2_shared_s7" \
    "bb_vanilla|experiments/main.py|cuda --seed 1 --tag bb_vanilla"

job 格式：用 `|` 分隔的三段 —— <tag>|<script>|<extra-args>
（tag 仅用于日志文件名与排序，不影响结果）
"""

import argparse
import os
import shlex
import subprocess
import sys
import time


def parse_job(spec):
    parts = spec.split("|")
    if len(parts) != 3:
        raise SystemExit(
            f"[launch_pack] bad job spec (need tag|script|args):\n  {spec!r}"
        )
    tag, script, extra = (p.strip() for p in parts)
    return tag, script, extra


def spawn(tag, script, extra, device, num_workers):
    os.makedirs("cache", exist_ok=True)
    log_path = f"cache/lfm4ads_{tag}.log"
    cmd = [sys.executable, script, device, *shlex.split(extra)]
    env = dict(os.environ)
    env["LFM_NUM_WORKERS"] = str(num_workers)
    with open(log_path, "w") as lf:
        lf.write(
            f"[launch_pack] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[launch_pack] device={device} tag={tag}\n"
            f"[launch_pack] cmd={' '.join(cmd)}\n"
            f"[launch_pack] LFM_NUM_WORKERS={num_workers}\n\n"
        )
    proc = subprocess.Popen(
        cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    print(f"[launch_pack] START {tag} on {device} (pid={proc.pid}) -> {log_path}")
    return tag, device, proc


def run_monitor(devices, jobs, interval):
    """Tail each job log + nvidia-smi until all procs exit."""
    print(f"[launch_pack] monitoring every {interval}s; "
          f"{len(jobs)} job(s) live on {', '.join(devices)}")
    while any(p.poll() is None for _, _, p in jobs):
        # GPU util
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            print(f"\n[monitor {time.strftime('%H:%M:%S')}] GPU util% / mem-MiB")
            for line in out.splitlines():
                print(f"  gpu {line}")
        except Exception as e:  # pragma: no cover
            print(f"[monitor] nvidia-smi failed: {e}")
        # Per-job tail
        for tag, _, proc in jobs:
            if proc.poll() is None:
                status = "RUNNING"
            elif proc.returncode == 0:
                status = "DONE"
            else:
                status = f"EXIT={proc.returncode}"
            log_path = f"cache/lfm4ads_{tag}.log"
            try:
                with open(log_path) as lf:
                    lines = lf.readlines()[-3:]
                tail = " | ".join(ln.strip() for ln in lines)
            except Exception:
                tail = "(no log)"
            print(f"  [{status}] {tag}: {tail[-140:]}")
        time.sleep(interval)
    print("[launch_pack] all jobs finished.")


def main():
    ap = argparse.ArgumentParser(description="GPU capacity packer")
    ap.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"],
                    help="device list to round-robin over")
    ap.add_argument("--max-per-device", type=int, default=3,
                    help="max concurrent jobs per device")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="LFM_NUM_WORKERS for packed jobs (default 4)")
    ap.add_argument("--monitor-interval", type=int, default=120,
                    help="seconds between monitor ticks (0 = no monitor)")
    ap.add_argument("jobs", nargs="*", help="job specs (tag|script|args)")
    args = ap.parse_args()

    if not args.jobs:
        ap.error("no jobs given; see module docstring for spec format")

    parsed = [parse_job(j) for j in args.jobs]

    # round-robin with per-device cap
    slots = {d: 0 for d in args.devices}
    live = []
    pending = list(parsed)
    dev_cycle = list(args.devices)
    di = 0

    print(f"[launch_pack] {len(parsed)} jobs, "
          f"cap={args.max_per_device}/device, devices={dev_cycle}")

    while pending or any(p.poll() is None for _, _, p in live):
        # free finished
        still_live = []
        for entry in live:
            tag, dev, proc = entry
            if proc.poll() is not None:
                slots[dev] -= 1
                print(f"[launch_pack] FINISH {tag} on {dev} "
                      f"(rc={proc.returncode})")
            else:
                still_live.append(entry)
        live = still_live

        # fill slots round-robin
        while pending and any(slots[d] < args.max_per_device for d in dev_cycle):
            tag, script, extra = pending.pop(0)
            # pick the next device that has a free slot, advancing di
            placed = False
            for _ in range(len(dev_cycle)):
                d = dev_cycle[di % len(dev_cycle)]
                di += 1
                if slots[d] < args.max_per_device:
                    live.append(spawn(tag, script, extra, d, args.num_workers))
                    slots[d] += 1
                    placed = True
                    break
            if not placed:
                # no free slot right now; put job back and wait
                pending.insert(0, (tag, script, extra))
                break

        if pending and all(p.poll() is not None for _, _, p in live) \
                and not any(slots[d] < args.max_per_device for d in dev_cycle):
            # should not happen, safety
            pass

        if args.monitor_interval > 0 and (pending or live):
            run_monitor_once(args.devices, live, args.monitor_interval)
        elif pending:
            time.sleep(5)

    print("[launch_pack] all jobs complete.")


def run_monitor_once(devices, live, interval):
    """single monitor tick (lightweight)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"\n[monitor {time.strftime('%H:%M:%S')}] GPU util% / mem-MiB")
        for line in out.splitlines():
            print(f"  gpu {line}")
    except Exception as e:  # pragma: no cover
        print(f"[monitor] nvidia-smi failed: {e}")
    for tag, _, proc in live:
        status = "RUNNING" if proc.poll() is None else f"EXIT={proc.returncode}"
        log_path = f"cache/lfm4ads_{tag}.log"
        try:
            with open(log_path) as lf:
                tail = " | ".join(ln.strip() for ln in lf.readlines()[-2:])
        except Exception:
            tail = "(no log)"
        print(f"  [{status}] {tag}: {tail[-140:]}")
    time.sleep(interval)


if __name__ == "__main__":
    main()
