"""Summarize E5 and emit the machine verdict + a human-readable report.

Verdict rules are hard-coded here exactly as pre-registered in
docs/20260815-0018-场景内泛化MoE长程矩阵预注册.md §4, so thresholds cannot drift.

Primary endpoint
    delta_moe(seed) = macro_test(moe, balanced, K=5) - macro_test(dense, balanced)
    paired within seed (same card by construction).

Noise floor
    NOT the pooled-AUC 0.001 constant: macro AUC averages 8 scenarios, some of
    which are small, so its run-to-run spread is larger and must be *measured*.
    floor = 2 * sample-SD of the dense/balanced arm's macro AUC across seeds.
    This is an internal control estimated from the same runs, so it cannot be
    tuned after seeing the MoE numbers.

Verdict
    PASS         : all seeds delta_moe > 0 AND mean(delta_moe) > floor
    FAIL         : all seeds delta_moe < 0
    INCONCLUSIVE : otherwise (sign flip, or effect inside the floor)
    BLOCKED      : missing primary runs, or the frozen-router sentinel fails
"""

import glob
import json
import os
import statistics as st

#: overridable so the report logic can be verified end-to-end on synthetic runs
#: in a scratch directory without ever touching the real evidence bundles, and
#: so 1K vs 27K evidence stay separated (same var as main_macro_auc.py)
OUT_DIR = os.environ.get("LFM_MACRO_OUT", "cache/macro_auc")
DECISION = f"{OUT_DIR}/e5_decision.json"
REPORT = f"{OUT_DIR}/e5_report.md"
SEEDS = (42, 123, 456, 789)
MACRO_SCENARIOS = (0, 1, 2, 3, 4, 5, 6, 8)
SENTINEL_TOL = 0.010  # frozen-router MoE vs dense: algebraically equal arms


def load_runs():
    runs = {}
    for path in sorted(glob.glob(f"{OUT_DIR}/run_*.json")):
        b = json.load(open(path))
        tag = os.path.basename(path)[4:-5]
        p = b["provenance"]
        runs[tag] = {
            "tag": tag, "stage": tag.split("_")[0], "seed": p["seed"],
            "arch": p["arch"], "loss": p["loss"], "K": p["K"], "lr": p["lr"],
            "device": p["device"], "lightweight": p["lightweight"],
            "frozen": p.get("freeze_router", False),
            "macro": b["test"]["macro"], "pooled": b["test"]["pooled"],
            "per_scenario": b["test"]["per_scenario"],
            "best_valid_macro": b["best_valid_macro"],
            "best_epoch": b["best_epoch"], "epochs_run": b["epochs_run"],
            "total_params": b["total_params"],
            "mean_wall_sec": b["mean_wall_sec"],
        }
    return runs


def paired(runs, a_tag, b_tag, key="macro"):
    """b - a, or None if either side is missing."""
    if a_tag not in runs or b_tag not in runs:
        return None
    return runs[b_tag][key] - runs[a_tag][key]


def fmt(x, n=6):
    return "n/a" if x is None else f"{x:+.{n}f}" if isinstance(x, float) else str(x)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs found")
    lines = ["# E5 报告：场景内泛化（macro-AUC）下的 scenario-routed MoE", ""]
    lines.append(f"- 生成时间：自动汇总（{len(runs)} 个 run 已完成）")
    lines.append("- 主终点：macro AUC over "
                 f"scenarios {list(MACRO_SCENARIOS)}（test，等权）")
    lines.append("- 预注册：`docs/20260815-0018-场景内泛化MoE长程矩阵预注册.md`")
    lines.append("")

    # ---------- Stage 1: primary 2x2 ----------
    d_moe_bal, d_moe_pool, d_loss_dense, d_loss_moe = {}, {}, {}, {}
    lines += ["## Stage 1（主实验）：arch × loss × 4 seed", "",
              "| seed | dense+pooled | moe+pooled | dense+balanced | "
              "moe+balanced | Δ_moe(bal) | Δ_moe(pool) | Δ_loss(dense) |",
              "|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        dp = f"s1_dense_pooled_s{s}"
        mp = f"s1_moe_pooled_K5_s{s}"
        db = f"s1_dense_balanced_s{s}"
        mb = f"s1_moe_balanced_K5_s{s}"
        d_moe_bal[s] = paired(runs, db, mb)
        d_moe_pool[s] = paired(runs, dp, mp)
        d_loss_dense[s] = paired(runs, dp, db)
        d_loss_moe[s] = paired(runs, mp, mb)
        g = lambda t: f"{runs[t]['macro']:.6f}" if t in runs else "—"
        lines.append(
            f"| {s} | {g(dp)} | {g(mp)} | {g(db)} | {g(mb)} | "
            f"{fmt(d_moe_bal[s])} | {fmt(d_moe_pool[s])} | "
            f"{fmt(d_loss_dense[s])} |")
    lines.append("")

    dense_bal = [runs[f"s1_dense_balanced_s{s}"]["macro"] for s in SEEDS
                 if f"s1_dense_balanced_s{s}" in runs]
    floor = (2 * st.stdev(dense_bal)) if len(dense_bal) >= 2 else None
    d_primary = [v for v in (d_moe_bal[s] for s in SEEDS) if v is not None]

    # ---------- sentinel ----------
    sentinel = {}
    for s in (42, 123):
        d = paired(runs, f"s1_dense_balanced_s{s}", f"s2sent_moe_frozen_s{s}")
        sentinel[s] = d
    sent_vals = [v for v in sentinel.values() if v is not None]
    sentinel_ok = bool(sent_vals) and all(abs(v) <= SENTINEL_TOL
                                          for v in sent_vals)

    # ---------- verdict ----------
    if len(d_primary) < len(SEEDS):
        verdict, reason = "BLOCKED", {
            "rule": "missing primary runs",
            "have": len(d_primary), "need": len(SEEDS)}
    elif not sentinel_ok:
        verdict, reason = "BLOCKED", {
            "rule": "frozen-router sentinel failed (frozen MoE must match "
                    "dense within tolerance)",
            "deltas": sentinel, "tolerance": SENTINEL_TOL}
    elif all(v > 0 for v in d_primary) and st.mean(d_primary) > (floor or 0):
        verdict, reason = "PASS", {
            "rule": "all seeds positive AND mean > measured noise floor",
            "deltas": d_primary, "mean": st.mean(d_primary), "floor": floor}
    elif all(v < 0 for v in d_primary):
        verdict, reason = "FAIL", {"rule": "all seeds negative",
                                   "deltas": d_primary}
    else:
        verdict, reason = "INCONCLUSIVE", {
            "rule": "sign flip across seeds, or effect within the noise floor",
            "deltas": d_primary, "mean": st.mean(d_primary), "floor": floor}

    lines += [f"**噪声地板（实测）** = 2 × SD(dense+balanced macro) = "
              f"{floor:.6f}" if floor else "**噪声地板**：样本不足", ""]
    lines += [f"**Δ_moe(balanced) 均值 = {st.mean(d_primary):+.6f}**"
              if d_primary else "", ""]
    lines += [f"**判定 = `{verdict}`** — {reason['rule']}", "",
              f"**函数保持哨兵**（frozen-router MoE − dense，应 ≈0，容差 "
              f"{SENTINEL_TOL}）：" +
              ", ".join(f"s{k}={fmt(v)}" for k, v in sentinel.items()) +
              f" → {'PASS' if sentinel_ok else 'FAIL'}", ""]

    # ---------- per-scenario decomposition ----------
    lines += ["## 逐场景分解（moe+balanced − dense+balanced，seed 平均）", "",
              "| scenario | dense | moe | Δ |", "|---|---|---|---|"]
    for sc in MACRO_SCENARIOS:
        dv, mv = [], []
        for s in SEEDS:
            db, mb = f"s1_dense_balanced_s{s}", f"s1_moe_balanced_K5_s{s}"
            if db in runs and runs[db]["per_scenario"].get(str(sc)) is not None:
                dv.append(runs[db]["per_scenario"][str(sc)])
            if mb in runs and runs[mb]["per_scenario"].get(str(sc)) is not None:
                mv.append(runs[mb]["per_scenario"][str(sc)])
        if dv and mv:
            lines.append(f"| s{sc} | {st.mean(dv):.4f} | {st.mean(mv):.4f} | "
                         f"{st.mean(mv) - st.mean(dv):+.4f} |")
    lines.append("")

    # ---------- pooled vs macro: show the endpoint mismatch ----------
    lines += ["## 口径对照：同一批 run 在 pooled 端点上的 Δ", "",
              "| seed | Δ_moe(macro) | Δ_moe(pooled) | 倍数 |",
              "|---|---|---|---|"]
    for s in SEEDS:
        db, mb = f"s1_dense_balanced_s{s}", f"s1_moe_balanced_K5_s{s}"
        dm = paired(runs, db, mb, "macro")
        dp = paired(runs, db, mb, "pooled")
        ratio = (f"{abs(dm / dp):.1f}x" if dm and dp and abs(dp) > 1e-9
                 else "—")
        lines.append(f"| {s} | {fmt(dm)} | {fmt(dp)} | {ratio} |")
    lines += ["", "> 若 macro 的 |Δ| 显著大于 pooled 的 |Δ|，即定量证明"
              "「过去用 pooled 端点测不到该机制」。", ""]

    # ---------- Stage 3: K sweep ----------
    ks = sorted({r["K"] for r in runs.values()
                 if r["stage"] == "s3" and r["K"]} | {5})
    if len(ks) > 1:
        lines += ["## Stage 3：隔离粒度 K 扫描（容量恒定）", "",
                  "| K | macro（seed 平均） | Δ vs dense | n_seed |",
                  "|---|---|---|---|"]
        for K in ks:
            vals, deltas = [], []
            for s in SEEDS:
                tag = (f"s1_moe_balanced_K5_s{s}" if K == 5
                       else f"s3_moe_balanced_K{K}_s{s}")
                db = f"s1_dense_balanced_s{s}"
                if tag in runs:
                    vals.append(runs[tag]["macro"])
                    if db in runs:
                        deltas.append(runs[tag]["macro"] - runs[db]["macro"])
            if vals:
                lines.append(f"| {K} | {st.mean(vals):.6f} | "
                             f"{fmt(st.mean(deltas)) if deltas else '—'} | "
                             f"{len(vals)} |")
        lines.append("")

    # ---------- Stage 4: lr robustness ----------
    lr_rows = []
    for lr in (2e-4, 1e-3, 3e-3):
        ds = []
        for s in SEEDS:
            if lr == 1e-3:
                db, mb = f"s1_dense_balanced_s{s}", f"s1_moe_balanced_K5_s{s}"
            else:
                db = f"s4_dense_balanced_lr{lr:g}_s{s}"
                mb = f"s4_moe_balanced_K5_lr{lr:g}_s{s}"
            d = paired(runs, db, mb)
            if d is not None:
                ds.append(d)
        if ds:
            lr_rows.append(f"| {lr:g} | {fmt(st.mean(ds))} | "
                           f"{sum(1 for d in ds if d > 0)}/{len(ds)} |")
    if lr_rows:
        lines += ["## Stage 4：lr 稳健性", "",
                  "| lr | Δ_moe 均值 | 正向 seed 数 |", "|---|---|---|"]
        lines += lr_rows + [""]

    # ---------- Stage 5: full embeddings ----------
    full_rows = []
    for s in (42, 123):
        d = paired(runs, f"s5full_dense_balanced_s{s}",
                   f"s5full_moe_balanced_s{s}")
        if d is not None:
            full_rows.append(f"| {s} | {fmt(d)} |")
    if full_rows:
        lines += ["## Stage 5：84M 全 embedding 下复核", "",
                  "| seed | Δ_moe(macro) |", "|---|---|"] + full_rows + [""]

    # ---------- decision json ----------
    decision = {
        "stage": "E5_in_scenario_generalization_moe",
        "status": "done",
        "verdict": verdict,
        "verdict_reason": reason,
        "primary_endpoint": "macro AUC over scenarios "
                            f"{list(MACRO_SCENARIOS)} (test, equal weight)",
        "model_selection": "exact argmax of macro valid AUC",
        "noise_floor_measured": floor,
        "noise_floor_definition": "2 * sample SD of dense+balanced macro AUC "
                                  "across seeds (internal control)",
        "delta_moe_balanced_per_seed": d_moe_bal,
        "delta_moe_pooled_per_seed": d_moe_pool,
        "delta_loss_dense_per_seed": d_loss_dense,
        "delta_loss_moe_per_seed": d_loss_moe,
        "frozen_router_sentinel": {"deltas": sentinel, "tolerance":
                                   SENTINEL_TOL, "ok": sentinel_ok},
        "n_runs_found": len(runs),
        "runs": {t: {k: v for k, v in r.items() if k != "per_scenario"}
                 for t, r in runs.items()},
        "boundary": [
            "macro AUC weights all 8 scenarios equally, so it is NOT "
            "comparable to the historical pooled-AUC numbers; the two "
            "endpoints must never be mixed in one table.",
            "A macro gain does not imply a pooled gain; if pooled drops, that "
            "is a real trade-off and must be reported, not hidden.",
            "Capacity is held constant (moe - dense = 45*K router params), so "
            "any gain is attributable to routing/isolation, not capacity.",
            "Scenario set was frozen pre-launch (historical TARGET_SCENARIOS); "
            "s7/s12 are auxiliary and never enter the verdict.",
        ],
        "preregistration":
            "docs/20260815-0018-场景内泛化MoE长程矩阵预注册.md",
    }
    with open(DECISION, "w") as f:
        json.dump(decision, f, indent=2)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines[:60]))
    print(f"\nwrote {DECISION}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
