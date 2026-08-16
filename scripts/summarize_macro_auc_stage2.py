"""Summarize E8 (pooled combo) / E9 (K sweep) / E10 (top-k sparsity) and emit
the machine verdict + report.

Verdict rules are hard-coded exactly as pre-registered in
docs/20260816-1250-三个后续任务预注册.md, so thresholds cannot drift.

Uses ``provenance`` in each run json (not tag parsing) for arch/loss/K/top_k/seed.
"""

import glob
import json
import os
import statistics as st

OUT_DIR = os.environ.get("LFM_MACRO_OUT", "cache/macro_auc")
DECISION = f"{OUT_DIR}/e8910_decision.json"
REPORT = f"{OUT_DIR}/e8910_report.md"

SEEDS = (42, 123, 456, 789)
SEEDS_EXTRA = (101, 202)
MACRO_SCENARIOS = (0, 1, 2, 3, 4, 5, 6, 8)
K_SWEEP = (2, 3, 5, 6, 10, 11)   # 5 = the E7 primary, included for reference
TOP_K = (2, 3)


def load_runs():
    runs = {}
    for path in sorted(glob.glob(f"{OUT_DIR}/run_*.json")):
        b = json.load(open(path))
        p = b["provenance"]
        tag = os.path.basename(path)[4:-5]
        if p.get("stage", "").startswith("SMOKE") or tag.startswith("SMOKE"):
            continue
        runs[tag] = {
            "tag": tag, "seed": p["seed"], "arch": p["arch"], "loss": p["loss"],
            "K": p["K"], "top_k": p.get("top_k"), "stage": tag.split("_")[0],
            "macro": b["test"]["macro"], "pooled": b["test"]["pooled"],
            "per_scenario": b["test"]["per_scenario"],
            "best_valid_macro": b["best_valid_macro"],
            "best_epoch": b["best_epoch"], "epochs_run": b["epochs_run"],
            "total_params": b["total_params"],
            "mean_wall_sec": b["mean_wall_sec"],
        }
    return runs


def get(runs, stage, arch, loss, seed, K=None, top_k=None):
    # dense runs inherit args.K default (5) in provenance, so K only matters
    # for moe; top_k only matters when requested.
    for tag, r in runs.items():
        if (r["stage"] == stage and r["arch"] == arch and r["loss"] == loss
                and r["seed"] == seed
                and (arch != "moe" or r["K"] == K)
                and (top_k is None or r["top_k"] == top_k)):
            return r
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def fmt(x, n=6):
    return "n/a" if x is None else f"{x:+.{n}f}" if isinstance(x, float) else str(x)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs found")
    lines = ["# E8/E9/E10 报告（pooled 组合 / K 扫描 / top-k 稀疏）", "",
             "- 预注册：`docs/20260816-1250-三个后续任务预注册.md`", ""]

    # ===================== E8: pooled loss + MoE =============================
    lines += ["## E8：pooled loss 下的 MoE（含补 seed）", "",
              "| seed | dense+pooled | moe+pooled | Δ |", "|---|---|---|---|"]
    d_pool = {}
    for s in SEEDS:
        d, m = get(runs, "s1", "dense", "pooled", s), get(
            runs, "s1", "moe", "pooled", s, K=5)
        if d and m:
            d_pool[s] = m["macro"] - d["macro"]
            lines.append(f"| {s} | {d['macro']:.6f} | {m['macro']:.6f} | "
                         f"{d_pool[s]:+.6f} |")
    for s in SEEDS_EXTRA:
        d, m = get(runs, "s7pool", "dense", "pooled", s), get(
            runs, "s7pool", "moe", "pooled", s, K=5)
        if d and m:
            d_pool[s] = m["macro"] - d["macro"]
            lines.append(f"| {s}(新) | {d['macro']:.6f} | {m['macro']:.6f} | "
                         f"{d_pool[s]:+.6f} |")
    lines.append("")

    dense_pool_macro = [get(runs, "s1", "dense", "pooled", s)["macro"]
                        for s in SEEDS
                        if get(runs, "s1", "dense", "pooled", s)]
    floor_pool = 2 * st.stdev(dense_pool_macro) if len(dense_pool_macro) >= 2 else None
    deltas = list(d_pool.values())
    if len(deltas) < 4:
        e8_verdict = "BLOCKED"
    elif all(d > 0 for d in deltas) and mean(deltas) > (floor_pool or 0):
        e8_verdict = "PASS"
    elif all(d < 0 for d in deltas):
        e8_verdict = "FAIL"
    else:
        e8_verdict = "INCONCLUSIVE"

    # absolute-macro ordering (the headline of E8)
    arms = {}
    for name, stage, arch, loss in [
            ("dense+pooled", "s1", "dense", "pooled"),
            ("moe+pooled", "s1", "moe", "pooled"),
            ("dense+balanced", "s1", "dense", "balanced"),
            ("moe+balanced", "s1", "moe", "balanced")]:
        kk = 5 if arch == "moe" else None
        vals = [get(runs, stage, arch, loss, s, K=kk)["macro"] for s in SEEDS
                if get(runs, stage, arch, loss, s, K=kk)]
        arms[name] = mean(vals)
    arms = {k: v for k, v in arms.items() if v is not None}
    order = sorted(arms.items(), key=lambda kv: -kv[1])

    lines += [f"**E8 判定 = `{e8_verdict}`**（pooled 下 Δ_moe，n={len(deltas)} seed，"
              f"均值 {mean(deltas):+.6f}，地板 {floor_pool:.6f})", "",
              "**绝对 macro 排序**（越高越好）：", ""]
    for i, (nm, v) in enumerate(order, 1):
        lines.append(f"{i}. {nm}: {v:.6f}")
    lines.append("")

    # ===================== E9: K sweep =======================================
    lines += ["## E9：隔离粒度 K 扫描（balanced，容量恒定）", "",
              "| K | mean macro | Δ vs dense | 正向 seed 数 | n_seed |",
              "|---|---|---|---|---|"]
    k_stats = {}
    for K in K_SWEEP:
        deltas_k, macros_k = [], []
        for s in SEEDS:
            m = get(runs, "s3", "moe", "balanced", s, K=K) if K != 5 else get(
                runs, "s1", "moe", "balanced", s, K=5)
            d = get(runs, "s1", "dense", "balanced", s)
            if m:
                macros_k.append(m["macro"])
                if d:
                    deltas_k.append(m["macro"] - d["macro"])
        if macros_k:
            k_stats[K] = {"mean_macro": mean(macros_k),
                          "mean_delta": mean(deltas_k) if deltas_k else None,
                          "n_pos": sum(1 for x in deltas_k if x > 0),
                          "n": len(deltas_k)}
            lines.append(f"| {K} | {k_stats[K]['mean_macro']:.6f} | "
                         f"{fmt(k_stats[K]['mean_delta'])} | "
                         f"{k_stats[K]['n_pos']}/{k_stats[K]['n']} | "
                         f"{k_stats[K]['n']} |")
    lines.append("")
    # best K by rule: max mean delta with >= 2/4 positive
    best_k = None
    for K, s in sorted(k_stats.items(), key=lambda kv: -(kv[1]["mean_delta"]
                        if kv[1]["mean_delta"] is not None else -9)):
        if s["mean_delta"] is not None and s["n_pos"] >= max(2, s["n"] // 2):
            best_k = K
            break
    lines += [f"**E9 最优 K = {best_k}**（mean Δ 最大且 ≥2/4 正）", ""]

    # ===================== E10: top-k sparsity ===============================
    lines += ["## E10：top-k 稀疏（K=5，激活稀疏，参数守恒）", "",
              "| top_k | mean macro | Δ vs dense | 正向 seed 数 | wall 倍数 vs 软路由 |",
              "|---|---|---|---|---|"]
    soft_wall = mean([get(runs, "s1", "moe", "balanced", s, K=5)["mean_wall_sec"]
                      for s in SEEDS if get(runs, "s1", "moe", "balanced", s, K=5)])
    e10 = {}
    for tk in TOP_K:
        deltas_tk, macros_tk, walls_tk = [], [], []
        for s in SEEDS:
            m = get(runs, "s6sparse", "moe", "balanced", s, K=5, top_k=tk)
            d = get(runs, "s1", "dense", "balanced", s)
            if m:
                macros_tk.append(m["macro"])
                walls_tk.append(m["mean_wall_sec"])
                if d:
                    deltas_tk.append(m["macro"] - d["macro"])
        if macros_tk:
            e10[tk] = {"mean_delta": mean(deltas_tk) if deltas_tk else None,
                       "n_pos": sum(1 for x in deltas_tk if x > 0),
                       "n": len(deltas_tk), "mean_macro": mean(macros_tk)}
            speedup = (soft_wall / mean(walls_tk)) if soft_wall and walls_tk else None
            lines.append(f"| {tk} | {e10[tk]['mean_macro']:.6f} | "
                         f"{fmt(e10[tk]['mean_delta'])} | "
                         f"{e10[tk]['n_pos']}/{e10[tk]['n']} | "
                         f"{speedup:.2f}x |" if speedup else f"| {tk} | ... |")
    lines.append("")

    e10_verdicts = {}
    for tk, s in e10.items():
        md = s["mean_delta"]
        if md is None:
            e10_verdicts[tk] = "BLOCKED"
        elif md > 0.00135 and s["n_pos"] >= 3:
            e10_verdicts[tk] = "PASS（稀疏未吃掉收益）"
        elif -0.00135 <= md <= 0.00135:
            e10_verdicts[tk] = "打平（稀疏代价≈隔离收益）"
        else:
            e10_verdicts[tk] = "FAIL（稀疏代价>隔离收益）"
    for tk, v in e10_verdicts.items():
        lines.append(f"- top_k={tk}: **{v}**")
    lines.append("")

    decision = {
        "stage": "E8_E9_E10",
        "e8": {"verdict": e8_verdict, "delta_pooled_per_seed": d_pool,
               "noise_floor_pooled": floor_pool,
               "abs_macro_order": [{"arm": n, "macro": v} for n, v in order]},
        "e9": {"best_K": best_k, "k_stats": k_stats},
        "e10": {str(tk): {"verdict": v, **e10[tk]}
                for tk, v in e10_verdicts.items()},
        "boundary": [
            "macro endpoint, equal weight over the 8 pre-registered scenarios.",
            "K sweep keeps capacity constant (moe-dense = 45*K router params).",
            "top-k is activation sparsity, params preserved.",
            "E8 primary endpoint (pooled) was INCONCLUSIVE at 4 seeds; extra "
            "seeds {101,202} are the pre-registered fallback.",
        ],
        "preregistration":
            "docs/20260816-1250-三个后续任务预注册.md",
    }
    with open(DECISION, "w") as f:
        json.dump(decision, f, indent=2)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {DECISION}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
