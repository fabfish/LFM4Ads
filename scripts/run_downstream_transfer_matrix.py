"""Orchestrate the held-out-domain downstream transfer matrix with G1/G2 gates.

Order is strictly staged:
  G1  -> source pretrain (dense+moe) for seed 42, then 12 downstream trials (seed 42).
        Gate on validation AUC: for every target,
          moe-router > dense-adapter-r2  AND  moe-router > moe-head.
        Only if G1 passes do we spend compute on seeds 123 and 456.
  G2  -> after all 36 trials, compute delta_primary and delta_adaptation and apply the
        pre-registered gate (PASS / FAIL / INCONCLUSIVE / BLOCKED).

All jobs run on a single device (cuda:1) as isolated subprocesses so model state and
CUDA memory never cross-contaminate. The same seed's four arms share this device, so
the paired-seed-on-same-device rule holds trivially.

Usage:
  python scripts/run_downstream_transfer_matrix.py plan
  python scripts/run_downstream_transfer_matrix.py execute
  python scripts/run_downstream_transfer_matrix.py status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from downstream_transfer_protocol import (
    TRANSFER_ARMS, TARGET_SCENARIOS, source_checkpoint_path,
)

DEVICE = "cuda:1"
SEEDS = (42, 123, 456)
TRIALS_DIR = Path("cache/downstream_transfer/trials")
GATE_DIR = Path("cache/downstream_transfer")
PY = sys.executable


def source_done(model: str, seed: int) -> bool:
    return source_checkpoint_path(model, seed).exists()


def trial_done(arm: str, target: int, seed: int) -> bool:
    return (TRIALS_DIR / f"{arm}_t{target}_s{seed}.json").exists()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_source(model: str, seed: int) -> None:
    if source_done(model, seed):
        print(f"[matrix] source {model} seed {seed} already present")
        return
    run([PY, "scripts/run_downstream_source_pretrain.py",
         "--model", model, "--device", DEVICE, "--seed", str(seed)])


def run_trial(arm: str, target: int, seed: int) -> None:
    if trial_done(arm, target, seed):
        print(f"[matrix] trial {arm} t{target} s{seed} already present")
        return
    run([PY, "scripts/run_downstream_transfer_trial.py",
         "--arm", arm, "--target", str(target), "--seed", str(seed), "--device", DEVICE])


def load_trial(arm: str, target: int, seed: int) -> dict:
    return json.loads((TRIALS_DIR / f"{arm}_t{target}_s{seed}.json").read_text())


def g1_gate_passed():
    ok = True
    detail = {}
    for target in TARGET_SCENARIOS:
        r_router = load_trial("moe-router", target, 42)["best_val_auc"]
        r_adapter = load_trial("dense-adapter-r2", target, 42)["best_val_auc"]
        r_head = load_trial("moe-head", target, 42)["best_val_auc"]
        passed = (r_router > r_adapter) and (r_router > r_head)
        detail[target] = {"moe_router": r_router, "dense_adapter_r2": r_adapter,
                          "moe_head": r_head, "passed": passed}
        ok = ok and passed
    return ok, detail


def compute_gate():
    delta_primary = {}
    delta_adapt = {}
    pairs = []
    for target in TARGET_SCENARIOS:
        for seed in SEEDS:
            mr = load_trial("moe-router", target, seed)["test_auc"]
            da = load_trial("dense-adapter-r2", target, seed)["test_auc"]
            mh = load_trial("moe-head", target, seed)["test_auc"]
            dh = load_trial("dense-head", target, seed)["test_auc"]
            dp = mr - da
            dadapt = (mr - mh) - (da - dh)
            delta_primary[(target, seed)] = dp
            delta_adapt[(target, seed)] = dadapt
            pairs.append((target, seed, dp, dadapt))

    all_dp = list(delta_primary.values())
    all_da = list(delta_adapt.values())
    mean_dp_by_target = {t: sum(delta_primary[(t, s)] for s in SEEDS) / len(SEEDS)
                         for t in TARGET_SCENARIOS}
    mean_da_by_target = {t: sum(delta_adapt[(t, s)] for s in SEEDS) / len(SEEDS)
                         for t in TARGET_SCENARIOS}
    macro_dp = sum(all_dp) / len(all_dp)
    macro_da = sum(all_da) / len(all_da)
    n_pos = sum(1 for v in all_dp if v > 0)
    all_neg = all(v <= 0 for v in all_dp)
    all_targets_nonpos = all(mean_dp_by_target[t] <= 0 for t in TARGET_SCENARIOS)

    if (n_pos == 9
            and all(v >= 0.001 for v in mean_dp_by_target.values())
            and macro_dp >= 0.0015 and macro_da >= 0.0005
            and all(v > 0 for v in mean_da_by_target.values())):
        verdict = "PASS"
    elif all_neg or all_targets_nonpos:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "n_pairs_positive": n_pos,
        "macro_mean_delta_primary": macro_dp,
        "macro_mean_delta_adaptation": macro_da,
        "mean_delta_primary_by_target": mean_dp_by_target,
        "mean_delta_adaptation_by_target": mean_da_by_target,
        "pairs": [{"target": t, "seed": s, "delta_primary": dp, "delta_adaptation": da}
                  for (t, s, dp, da) in pairs],
    }


def all_trials_done() -> bool:
    return all(trial_done(a, t, s) for a in TRANSFER_ARMS
               for t in TARGET_SCENARIOS for s in SEEDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["plan", "execute", "status"])
    args = ap.parse_args()

    if args.action == "plan":
        print("Plan (single device, staged G1 -> G2):")
        for seed in SEEDS:
            print(f"  seed {seed}: source(dense,moe) + "
                  f"{len(TRANSFER_ARMS) * len(TARGET_SCENARIOS)} downstream trials")
        return

    if args.action == "status":
        done = sum(1 for a in TRANSFER_ARMS for t in TARGET_SCENARIOS for s in SEEDS
                   if trial_done(a, t, s))
        print(f"trials done: {done}/36")
        if all_trials_done():
            print(json.dumps(compute_gate(), indent=2))
        else:
            print("G2 gate not computed (missing trials).")
        return

    # execute -- G1 seed 42
    print("[matrix] G1: seed 42")
    for model in ("dense", "moe"):
        ensure_source(model, 42)
    for arm in TRANSFER_ARMS:
        for target in TARGET_SCENARIOS:
            run_trial(arm, target, 42)

    g1_ok, g1_detail = g1_gate_passed()
    print("[matrix] G1 gate:", json.dumps(g1_detail, indent=2))
    (GATE_DIR / "g1_decision.json").write_text(json.dumps(
        {"gate": "G1", "passed": g1_ok, "detail": g1_detail}, indent=2))

    if not g1_ok:
        print("[matrix] G1 FAILED -> stop (no compute on seeds 123/456)")
        return

    # G2 seeds
    for seed in (123, 456):
        print(f"[matrix] G2 seed {seed}")
        for model in ("dense", "moe"):
            ensure_source(model, seed)
        for arm in TRANSFER_ARMS:
            for target in TARGET_SCENARIOS:
                run_trial(arm, target, seed)

    gate = compute_gate()
    (GATE_DIR / "gate_decision.json").write_text(json.dumps(gate, indent=2))
    print("[matrix] G2 gate decision:")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
