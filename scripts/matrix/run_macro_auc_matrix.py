"""E5 long-run matrix driver — unattended, resume-safe, ~8h of 12h budget.

Pre-registered in docs/20260815-0018-场景内泛化MoE长程矩阵预注册.md.
The task list below IS the pre-registration: it is fixed before launch so no
configuration can be added afterwards to manufacture a positive result.

Design guarantees
-----------------
* **Pairing**: seed -> device is a fixed map, so *every* configuration of a
  given seed runs on the same card (AGENTS.md rule). Different seeds run
  concurrently on different cards.
* **Resume-safe**: a run whose ``cache/macro_auc/run_<tag>.json`` exists is
  skipped, so the matrix can be killed and restarted at any time.
* **Failure isolation**: a crashing run is recorded and the worker moves on;
  it never takes the matrix down.
* **Wall-clock guard**: stops dispatching new runs once ``--budget-hours`` is
  exhausted (in-flight runs are allowed to finish), so it cannot overrun.
* **Auto-summary**: on completion the verdict + markdown tables are produced
  by ``scripts/summarize/summarize_macro_auc.py`` — nothing waits on a human.

Usage:  nohup python scripts/matrix/run_macro_auc_matrix.py > logs/macro_matrix.log 2>&1 &
"""

import argparse
import json
import os
import subprocess
import threading
import time

#: scripts/matrix/ -> repo root (three levels up)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: 1K and 27K evidence MUST stay in separate dirs (identical tags otherwise
#: collide with the resume guard). Controlled by LFM_MACRO_OUT; LOG_DIR follows.
OUT_DIR = os.environ.get("LFM_MACRO_OUT", os.path.join(ROOT, "cache", "macro_auc"))
LOG_DIR = OUT_DIR.replace("cache", "logs", 1)
STATE = os.path.join(OUT_DIR, "matrix_state.json")

SEEDS = (42, 123, 456, 789)
#: extra seeds for E8's pooled-loss robustness (pre-registered {101, 202})
SEEDS_EXTRA = (101, 202)
#: Fixed so that ALL configs of one seed share one card (paired-difference rule,
#: AGENTS.md §2). Two-site collaboration: site A has 2 GPUs, site B has 3, so
#: the map is selected by LFM_SITE. A seed is never split across cards, and a
#: PAIR is never split across sites (docs/20260817-1400 §2.1/§2.5).
_SEED_DEVICE_BY_SITE = {
    "A": {42: "cuda:0", 456: "cuda:0", 101: "cuda:0",
          123: "cuda:1", 789: "cuda:1", 202: "cuda:1"},
    # 3 does not divide 4, so 42 and 789 share cuda:0; each seed still keeps
    # all of its arms on one card.
    "B": {42: "cuda:0", 789: "cuda:0", 101: "cuda:0",
          123: "cuda:1", 202: "cuda:1",
          456: "cuda:2"},
}
SITE = os.environ.get("LFM_SITE", "A").upper()
SEED_DEVICE = _SEED_DEVICE_BY_SITE.get(SITE, _SEED_DEVICE_BY_SITE["A"])
MAIN_K = 5           # 330 = 2*3*5*11, so K must divide 330
K_SWEEP = (2, 3, 6, 10, 11)
LR_SWEEP = (2e-4, 3e-3)   # 1e-3 is covered by stage 1
#: top-k values for s6sparse (E10). Reduced from (2, 3) to (2,) on 2026-08-16:
#: tk2 already showed +0.0053 vs dense (2 seeds, stronger than soft routing's
#: +0.0019), so tk3 is a low-information middle point; authorized reduction
#: registered in docs/20260816-1250-三个后续任务预注册.md.
TOP_KS = (2,)
EPOCHS = 80
PATIENCE = 12
#: Stage1 primary is 2 arch x 2 loss; on 27K (27x data, ~200s/epoch) a run is
#: far more expensive, so we drop the exploratory sweeps and keep only the
#: primary 2x2 + the function-preservation sentinel. Stages are additive and
#: each experiment may override via --stages.
PRIMARY_STAGES = ("s1", "s2sent")


def _task(stage, seed, arch, loss, K=MAIN_K, lr=1e-3, top_k=None, extra=(),
          tag=None, epochs=EPOCHS, patience=PATIENCE):
    tag = tag or (f"{stage}_{arch}_{loss}"
                  + (f"_K{K}" if arch == "moe" else "")
                  + (f"_tk{top_k}" if arch == "moe" and top_k is not None
                     else "")
                  + (f"_lr{lr:g}" if lr != 1e-3 else "")
                  + f"_s{seed}")
    cmd = ["python", "experiments/main_macro_auc.py", SEED_DEVICE[seed],
           "--arch", arch, "--loss", loss, "--seed", str(seed),
           "--lr", str(lr), "--max-epochs", str(epochs),
           "--patience", str(patience), "--tag", tag]
    if arch == "moe":
        cmd += ["--K", str(K)]
        if top_k is not None:
            cmd += ["--top-k", str(top_k)]
    cmd += list(extra)
    return {"stage": stage, "tag": tag, "seed": seed, "arch": arch,
            "loss": loss, "K": K if arch == "moe" else None,
            "top_k": top_k if arch == "moe" else None, "lr": lr,
            "device": SEED_DEVICE[seed], "cmd": cmd}


def build_tasks(stages=PRIMARY_STAGES, epochs=EPOCHS, patience=PATIENCE):
    """The frozen task list, dispatched in the ORDER of ``stages``.

    ``stages`` selects which pre-registered stages to run AND their order, so a
    quick-verdict stage (e.g. ``s7pool``) can be scheduled first. The stage set
    and each stage's inner loop are the pre-registration; only the *selection*
    and *order* of stages vary per dataset/budget.
    """
    builders = {
        # Stage 1 (PRIMARY): 2 arch x 2 loss x 4 seeds = 16 runs.
        #   arch dense->moe = parameter isolation (capacity held constant)
        #   loss pooled->balanced = gradient rebalancing toward small scenarios
        "s1": lambda: [t for seed in SEEDS for loss in ("balanced", "pooled")
                       for arch in ("dense", "moe")
                       for t in [_task("s1", seed, arch, loss,
                                       epochs=epochs, patience=patience)]],
        # Stage 2 sentinel: frozen-uniform router must match dense.
        "s2sent": lambda: [_task("s2sent", seed, "moe", "balanced",
                                 extra=["--freeze-router"],
                                 tag=f"s2sent_moe_frozen_s{seed}",
                                 epochs=epochs, patience=patience)
                           for seed in (42, 123)],
        # Stage 3 (E9): isolation granularity at CONSTANT capacity.
        "s3": lambda: [_task("s3", seed, "moe", "balanced", K=K,
                             epochs=epochs, patience=patience)
                       for seed in SEEDS for K in K_SWEEP],
        # Stage 4: lr robustness of the primary contrast.
        "s4": lambda: [_task("s4", seed, arch, "balanced", lr=lr,
                             epochs=epochs, patience=patience)
                       for seed in SEEDS for lr in LR_SWEEP
                       for arch in ("dense", "moe")],
        # Stage 5: does the conclusion survive the full embeddings?
        "s5full": lambda: [_task("s5full", seed, arch, "balanced",
                                 extra=["--full-embeddings"],
                                 tag=f"s5full_{arch}_balanced_s{seed}",
                                 epochs=epochs, patience=patience)
                           for seed in (42, 123)
                           for arch in ("dense", "moe")],
        # Stage 6 (E10): hard top-k sparsity, K=5, params preserved.
        "s6sparse": lambda: [_task("s6sparse", seed, "moe", "balanced",
                                   K=MAIN_K, top_k=tk,
                                   epochs=epochs, patience=patience)
                             for seed in SEEDS for tk in TOP_KS],
        # Stage 7 (E8 extra seeds): pooled-loss robustness on {101, 202}.
        "s7pool": lambda: [_task("s7pool", seed, arch, "pooled",
                                 epochs=epochs, patience=patience)
                           for seed in SEEDS_EXTRA
                           for arch in ("dense", "moe")],
        # Stage 8 (E11): full-ID fairness — re-add the 550M ID embeddings and
        # re-run the FINAL config (dense vs moe K=5 hard top_k=2, balanced).
        # Same optimizer (dense AdamW), same params except routing; answers
        # "does the MoE gain survive adding back the ID tables?".
        "s8full": lambda: [_task("s8full", seed, arch, "balanced",
                                 K=MAIN_K, top_k=2 if arch == "moe" else None,
                                 extra=["--full-embeddings"],
                                 tag=f"s8full_{arch}_balanced_s{seed}",
                                 epochs=epochs, patience=patience)
                           for seed in SEEDS
                           for arch in ("dense", "moe")],
        # Stage 9 (E15): MODEL-SELECTION ENDPOINT sensitivity. Same final config
        # as E10 (lightweight, balanced, K=5 hard top_k=2), but each run keeps
        # one checkpoint per selection endpoint (macro / pooled valid) in a
        # single trajectory and evaluates test for both -> answers "does the MoE
        # gain survive selecting epochs by the OLD endpoint?" at zero extra
        # training cost, perfectly paired. Also dumps per-epoch test curves so
        # any other selection rule can be replayed post-hoc.
        # Pre-registered: docs/20260817-1330-E15-选择端点敏感性预注册.md
        "s9sel": lambda: [_task("s9sel", seed, arch, "balanced",
                                K=MAIN_K, top_k=2 if arch == "moe" else None,
                                extra=["--selection", "both",
                                       "--eval-test-each-epoch"],
                                tag=f"e15_{arch}_s{seed}",
                                epochs=epochs, patience=patience)
                          for seed in SEEDS
                          for arch in ("dense", "moe")],
        # ------------------------------------------------------------------
        #  site B stages (3-GPU machine).每个 stage 都是 SITE 内自足的：
        #  它自带 dense 对照臂，噪声地板由本 site 的 dense 臂算出。
        #  跨 site 相减一律禁止 —— docs/20260817-1400-两端协作分离设计与实验分工.md
        # ------------------------------------------------------------------
        # B0: site B 的 dense 基线（4 seed，建立本 site 噪声地板 + 可比性校验）
        #     + 1 个 moe top_k=2 run 用于 E12 专家利用率诊断（dump router 权重）
        "b0repro": lambda: (
            [_task("b0repro", seed, "dense", "balanced",
                   tag=f"b0repro_dense_s{seed}",
                   epochs=epochs, patience=patience)
             for seed in SEEDS]
            + [_task("b0repro", 42, "moe", "balanced", K=MAIN_K, top_k=2,
                     tag="b0repro_moe_tk2_s42",
                     epochs=epochs, patience=patience)]),
        # B1 (was E14): top_k monotonicity. dense arm reused from b0repro;
        # only top_k varies, parameter count identical across all arms.
        "b1topk": lambda: [_task("b1topk", seed, "moe", "balanced",
                                 K=MAIN_K, top_k=tk,
                                 tag=f"b1_moe_tk{tk}_s{seed}",
                                 epochs=epochs, patience=patience)
                           for seed in SEEDS for tk in (2, 1)],
    }
    tasks = []
    for s in stages:
        if s not in builders:
            raise SystemExit(f"unknown stage {s!r}; valid: {sorted(builders)}")
        tasks.extend(builders[s]())
    return tasks


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed": [], "failed": [], "skipped": []}


class Matrix:
    def __init__(self, tasks, budget_hours):
        self.lock = threading.Lock()
        self.state = load_state()
        self.deadline = time.time() + budget_hours * 3600
        self.budget_hours = budget_hours
        self.tasks = tasks
        self.done_tags = {r["tag"] for r in self.state["completed"]}

    def save(self):
        with self.lock:
            self.state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.state["n_completed"] = len(self.state["completed"])
            self.state["n_failed"] = len(self.state["failed"])
            tmp = STATE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp, STATE)

    def worker(self, device, my_tasks):
        for idx, task in enumerate(my_tasks, 1):
            out_json = os.path.join(OUT_DIR, f"run_{task['tag']}.json")
            if os.path.exists(out_json):
                with self.lock:
                    self.state["skipped"].append(task["tag"])
                print(f"[{device}] ({idx}/{len(my_tasks)}) skip {task['tag']} "
                      f"(already done)", flush=True)
                continue
            if time.time() > self.deadline:
                print(f"[{device}] budget exhausted, not starting "
                      f"{task['tag']}", flush=True)
                with self.lock:
                    self.state.setdefault("not_started", []).append(task["tag"])
                continue
            log = os.path.join(LOG_DIR, f"{task['tag']}.log")
            t0 = time.time()
            print(f"[{device}] ({idx}/{len(my_tasks)}) RUN {task['tag']}",
                  flush=True)
            try:
                with open(log, "w") as lf:
                    rc = subprocess.call(task["cmd"], cwd=ROOT, stdout=lf,
                                         stderr=subprocess.STDOUT)
            except Exception as exc:  # never let one run kill the matrix
                rc, exc_txt = -99, repr(exc)
            else:
                exc_txt = None
            wall = time.time() - t0
            rec = {**{k: v for k, v in task.items() if k != "cmd"},
                   "returncode": rc, "wall_sec": round(wall, 1), "log": log}
            if exc_txt:
                rec["exception"] = exc_txt
            with self.lock:
                if rc == 0 and os.path.exists(out_json):
                    self.state["completed"].append(rec)
                    status = "OK"
                else:
                    self.state["failed"].append(rec)
                    status = f"FAIL rc={rc}"
            self.save()
            print(f"[{device}] ({idx}/{len(my_tasks)}) {status} "
                  f"{task['tag']} {wall / 60:.1f}min", flush=True)

    def run(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        by_device = {}
        for t in self.tasks:
            by_device.setdefault(t["device"], []).append(t)
        self.state["plan"] = {d: [t["tag"] for t in ts]
                              for d, ts in by_device.items()}
        self.state["budget_hours"] = self.budget_hours
        self.save()
        print(f"[matrix] {len(self.tasks)} runs, budget {self.budget_hours}h, "
              f"devices={list(by_device)}", flush=True)
        for d, ts in by_device.items():
            print(f"  {d}: {len(ts)} runs (seeds "
                  f"{sorted({t['seed'] for t in ts})})", flush=True)
        threads = [threading.Thread(target=self.worker, args=(d, ts),
                                   daemon=False)
                   for d, ts in by_device.items()]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        print(f"\n[matrix] done: {len(self.state['completed'])} ok, "
              f"{len(self.state['failed'])} failed, "
              f"{len(self.state['skipped'])} skipped", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=11.0,
                    help="stop dispatching new runs after this many hours")
    ap.add_argument("--stages", default=",".join(PRIMARY_STAGES),
                    help="comma-separated stages to run (s1,s2sent,s3,s4,s5full)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = tuple(s for s in args.stages.split(",") if s)
    tasks = build_tasks(stages=stages, epochs=args.epochs,
                        patience=args.patience)
    if args.dry_run:
        by_dev = {}
        for t in tasks:
            by_dev.setdefault(t["device"], []).append(t["tag"])
        print(f"{len(tasks)} runs total")
        for d, tags in by_dev.items():
            print(f"\n{d} ({len(tags)} runs):")
            for tg in tags:
                print(f"  {tg}")
        return

    Matrix(tasks, args.budget_hours).run()

    print("\n[matrix] summarizing ...", flush=True)
    subprocess.call(["python", "scripts/summarize/summarize_macro_auc.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
