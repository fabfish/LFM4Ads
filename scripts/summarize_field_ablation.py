"""Summarize E1 and emit the machine verdict (cache/embedding_capacity/e1_decision.json).

Applies the verdict rules frozen in
docs/20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md §5.5 —
they are hard-coded here on purpose so the thresholds cannot drift.

  PASS (dead weight)   : all seeds |Δ_id| < 0.001
  FAIL (ID contributes): all seeds Δ_id <= -0.001
  INCONCLUSIVE         : otherwise (sign flip / straddling the floor)
  BLOCKED              : any sentinel FAIL or a missing run
"""

import glob
import json
import os

OUT = "cache/embedding_capacity/e1_decision.json"
NOISE_FLOOR = 0.001
REQUIRED_ARMS = ("full", "idzero", "iddrop")


def main():
    files = sorted(glob.glob("cache/embedding_capacity/e1_history_s*.json"))
    if not files:
        raise SystemExit("no E1 history files found")

    seeds, sentinel_fail, missing = {}, [], []
    for path in files:
        b = json.load(open(path))
        seed = b["provenance"]["seed"]
        for name, s in b["sentinels"].items():
            if not s["ok"]:
                sentinel_fail.append(f"seed{seed}:{name}")
        for arm in REQUIRED_ARMS:
            if arm not in b["results"]:
                missing.append(f"seed{seed}:{arm}")
        seeds[seed] = {
            "device": b["provenance"]["device"],
            "lr": b["provenance"]["lr"],
            "max_epochs": b["provenance"]["max_epochs"],
            "test_auc": {a: r["test_auc"] for a, r in b["results"].items()},
            "peak_valid_auc": {a: r["best_valid_auc"]
                               for a, r in b["results"].items()},
            "best_epoch": {a: r["best_epoch"] for a, r in b["results"].items()},
            "total_params": {a: r["total_params"]
                             for a, r in b["results"].items()},
            "trainable_params": {a: r["trainable_params"]
                                 for a, r in b["results"].items()},
            "mean_wall_sec": {a: r["mean_wall_sec"]
                              for a, r in b["results"].items()},
            "delta_idzero_minus_full": b["deltas"]["delta_idzero_minus_full"],
            "delta_iddrop_minus_full": b["deltas"]["delta_iddrop_minus_full"],
            "delta_peak_valid_idzero_minus_full":
                b["results"]["idzero"]["best_valid_auc"]
                - b["results"]["full"]["best_valid_auc"],
        }

    d_id = [v["delta_idzero_minus_full"] for v in seeds.values()]
    if sentinel_fail or missing:
        verdict, reason = "BLOCKED", {"sentinel_fail": sentinel_fail,
                                      "missing_runs": missing}
    elif all(abs(d) < NOISE_FLOOR for d in d_id):
        verdict = "PASS"
        reason = {"rule": "all seeds |delta_id| < noise_floor",
                  "deltas": d_id}
    elif all(d <= -NOISE_FLOOR for d in d_id):
        verdict = "FAIL"
        reason = {"rule": "all seeds delta_id <= -noise_floor",
                  "deltas": d_id}
    else:
        verdict = "INCONCLUSIVE"
        reason = {"rule": "sign flip or straddling the floor", "deltas": d_id}

    ref = seeds[sorted(seeds)[0]]
    decision = {
        "stage": "E1_id_embedding_dead_weight",
        "status": "done",
        "verdict": verdict,
        "verdict_reason": reason,
        "main_endpoint": "delta_id = test_AUC(idzero) - test_AUC(full), "
                         "paired per seed (same card)",
        "noise_floor": NOISE_FLOOR,
        "n_seeds": len(seeds),
        "n_runs": len(seeds) * len(REQUIRED_ARMS),
        "sentinels_all_pass": not (sentinel_fail or missing),
        "id_embedding_params_removed": 83_984_250,
        "param_reduction_iddrop": 1 - (ref["total_params"]["iddrop"]
                                       / ref["total_params"]["full"]),
        "trainable_param_reduction_idzero": 1 - (
            ref["trainable_params"]["idzero"]
            / ref["trainable_params"]["full"]),
        "wall_speedup_iddrop": (ref["mean_wall_sec"]["full"]
                                / ref["mean_wall_sec"]["iddrop"]),
        "per_seed": seeds,
        "unlock_feature_information_track": verdict == "PASS",
        "unlock_id_capacity_track": verdict == "FAIL",
        "boundary": [
            "|delta| < noise floor means NO measurable difference; it does NOT "
            "license the claim that removing the ID tables IMPROVES AUC.",
            "Budget bound: lr=1e-3, 15 epochs, batch=10000, from scratch, "
            "2 seeds. No extrapolation to longer budgets or other datasets.",
            "iddrop is an engineering control (params/wall-clock), never the "
            "main verdict.",
        ],
        "preregistration":
            "docs/20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md §5",
        "sources": files,
    }
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; machine verdicts are immutable.")
    with open(OUT, "w") as f:
        json.dump(decision, f, indent=2)

    print(f"verdict = {verdict}  (deltas={['%+.6f' % d for d in d_id]}, "
          f"floor={NOISE_FLOOR})")
    for seed in sorted(seeds):
        v = seeds[seed]
        print(f"  seed {seed} @{v['device']}: "
              f"full={v['test_auc']['full']:.6f} "
              f"idzero={v['test_auc']['idzero']:.6f} "
              f"iddrop={v['test_auc']['iddrop']:.6f} "
              f"Δ_id={v['delta_idzero_minus_full']:+.6f} "
              f"Δ_drop={v['delta_iddrop_minus_full']:+.6f}")
    print(f"param reduction (iddrop) = "
          f"{decision['param_reduction_iddrop']:.2%}; "
          f"wall speedup = {decision['wall_speedup_iddrop']:.2f}x")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
