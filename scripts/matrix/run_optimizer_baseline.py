"""F1 — optimizer state-sharing baseline matrix (2 GPUs, site A).

Runs the four optimizer arms of the pre-registration
``docs/20260818-1115-F1-优化器状态共享度baseline预注册.md``:

    shared_adamw (control) / task_state_uniform (DualOptim) /
    dual_optim_plus (DualOptim+) / role_isolated (SkewAdam)

Fixed: 27K, lightweight MoE(K=5, top_k=2), balanced loss, lr=5e-4,
20 epochs / patience 10, macro endpoint, 2 seeds (42 -> cuda:0, 123 -> cuda:1).

Exploration stage (2 seeds): reports direction + magnitude only, no formal
verdict (formal needs 4 seeds). Breakpoint-resume via existing run json.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(os.environ.get(
    "LFM_TASK_ROLE_OUT", str(ROOT / "cache" / "task_role_optimizer_27k_siteA")))
LOG_DIR = ROOT / "logs" / "task_role_optimizer_27k_siteA"

#: frozen arm x seed matrix (seed -> device keeps a seed's arms on one card)
ARMS = ("shared_adamw", "task_state_uniform", "dual_optim_plus", "role_isolated")
SEED_DEVICE = {42: "cuda:0", 123: "cuda:1"}
SEEDS = (42, 123)

ENTRY = str(ROOT / "experiments" / "main_task_role_optimizer.py")
#: 探索阶段口径（子采样 + 少 epoch，方向性 baseline，非正式判定）。
#: jacrev 全数据单 epoch 约 1.5-2h（15 场景 = 15 次 vjp），20 epoch 不可行；
#: 探索规模 = max-batches 1000（约 3.9% train，与 site B 短测口径一致）
#: + 5 epoch + patience 3。截断口径写入 provenance.explore / max_batches。
MAX_EPOCHS = 5
PATIENCE = 3
MAX_BATCHES = 1000


def tag(arm: str, seed: int) -> str:
    return f"f1_{arm}_s{seed}"


def cmd(arm: str, seed: int, device: str) -> list[str]:
    return [
        sys.executable, ENTRY, device,
        "--optimizer-mode", arm,
        "--seed", str(seed),
        "--max-epochs", str(MAX_EPOCHS),
        "--patience", str(PATIENCE),
        "--max-batches", str(MAX_BATCHES),
        "--explore",
        "--tag", tag(arm, seed),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(arm, seed, SEED_DEVICE[seed])
             for seed in SEEDS for arm in ARMS]
    pending = []
    for arm, seed, device in tasks:
        dst = OUT_DIR / f"run_{tag(arm, seed)}.json"
        if dst.exists():
            print(f"[skip] {dst.name} already exists")
            continue
        pending.append((arm, seed, device, dst))

    print(f"F1 baseline: {len(tasks)} tasks, {len(pending)} pending, "
          f"{len(tasks) - len(pending)} already done")

    # split pending by device to run each card's arms serially (paired-seed rule)
    by_device: dict[str, list] = {}
    for item in pending:
        by_device.setdefault(item[2], []).append(item)

    procs: dict[str, subprocess.Popen] = {}
    for device, items in by_device.items():
        log = LOG_DIR / f"f1_{device.replace(':', '')}.log"
        lf = open(log, "a")
        # per-card serial shell command; ';' so one arm's failure doesn't block
        # the rest of the card's queue (failed arms re-run manually later)
        script = " ; ".join(
            " ".join(cmd(arm, seed, device)) for arm, seed, _, _ in items)
        procs[device] = subprocess.Popen(
            script, shell=True, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(ROOT))
        print(f"[launch] {device}: {len(items)} arms -> {log.name}")

    if not pending:
        print("nothing to run (all arms present)")
        return

    # wait for all cards to finish
    while procs:
        for device, proc in list(procs.items()):
            if proc.poll() is not None:
                print(f"[done] {device} exited with {proc.returncode}")
                del procs[device]
        time.sleep(30)

    done = sum(1 for _, _, _, dst in pending if dst.exists())
    print(f"finished: {done}/{len(pending)} arms produced run json")


if __name__ == "__main__":
    main()
