"""E17 Stage 1 并行启动器：seed 101 → cuda:0，seed 202 → cuda:1。

同 seed 的 S→T→A 在同一张卡顺序执行（配对纪律）；两卡并行（用户 2026-08-18
授权的并行执行）。已完成的 run（json 存在且 status=completed）自动跳过。

用法：
  nohup python scripts/matrix/run_adatask_win_case.py > logs/adatask_win_case/runner.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(os.environ.get(
    "LFM_E17_OUT", "cache/adatask_win_s0_27k_siteA"))
LOG_DIR = REPO_ROOT / "logs" / "adatask_win_case"

ENV = {
    **os.environ,
    "LFM_DATASET": "dataset_27k.feather",
    "LFM_VOCAB_JSON": "cache/fields_27k.json",
    "LFM_SAMPLE_COUNTS_JSON": "cache/sample_counts_27k.json",
    "LFM_SITE": "A",
    "LFM_E17_OUT": str(OUT_DIR),
}

DEFAULT_SEED_DEVICE = {101: "cuda:0", 202: "cuda:1"}
ARM_ORDER = ("S", "T", "A")


def completed(out_dir: Path, arm: str, seed: int) -> bool:
    path = out_dir / f"run_e17_{arm}_s{seed}.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def launch(seed: int, device: str) -> subprocess.Popen:
    commands = [
        f"python experiments/main_adatask_win_case.py "
        f"--device {device} --arm {arm} --seed {seed} "
        f">> {LOG_DIR}/e17_s{seed}_{device.replace(':', '')}.log 2>&1"
        for arm in ARM_ORDER
        if not completed(OUT_DIR, arm, seed)
    ]
    script = "set -e; " + " && ".join(commands)
    return subprocess.Popen(["bash", "-c", script], cwd=REPO_ROOT, env=ENV)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=sorted(DEFAULT_SEED_DEVICE))
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / OUT_DIR).mkdir(parents=True, exist_ok=True)

    pending = {seed: DEFAULT_SEED_DEVICE[seed] for seed in args.seeds
               if any(not completed(OUT_DIR, arm, seed)
                      for arm in ARM_ORDER)}
    print(f"[runner] 待运行: {pending or '无'}", flush=True)
    procs = {seed: launch(seed, device) for seed, device in pending.items()}
    t0 = time.time()
    while procs:
        for seed in list(procs):
            code = procs[seed].poll()
            if code is None:
                continue
            print(f"[runner] seed {seed} 退出码 {code} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del procs[seed]
        time.sleep(30)
    summary = {
        seed: {arm: ("done" if completed(OUT_DIR, arm, seed) else "MISSING")
               for arm in ARM_ORDER}
        for seed in args.seeds
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if any(v == "MISSING" for s in summary.values() for v in s.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
