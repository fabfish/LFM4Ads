"""E15 — model-SELECTION-endpoint sensitivity: summarise + verdict.

Answers the question "does the MoE gain survive selecting epochs by the OLD
(pooled) endpoint?" using the paired runs produced by
``run_macro_auc_matrix.py --stages s9sel``.

Every threshold and rule below is HARD-CODED from the pre-registration
(docs/20260817-1330-E15-选择端点敏感性预注册.md §4) so that no drift is
possible at analysis time:

  * primary metric : test macro AUC paired delta (moe - dense), same seed
  * noise floor    : 2 * SD(dense test macro across seeds), computed WITHIN
                     each selection rule
  * PASS           : 4/4 seeds same-sign positive AND mean > that floor
  * reproduction   : |E15(selection=macro) - E10| < 5e-4 on all 4 seeds,
                     otherwise the whole E15 result is void (§2.2 sentinel)

Usage: python scripts/summarize/summarize_selection_sensitivity.py
"""

import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "cache", "macro_auc_27k")
SEEDS = (42, 123, 456, 789)
SELECTIONS = ("macro", "pooled")
#: pre-registered sentinel tolerance for reproducing E10 (§2.2)
REPRO_TOL = 5e-4
#: E10 reference runs (selection=macro), used only for the sentinel
E10_TAGS = {"dense": "s1_dense_balanced_s{seed}",
            "moe": "s6sparse_moe_balanced_K5_tk2_s{seed}"}


def load(tag):
    p = os.path.join(OUT_DIR, f"run_{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def verdict(deltas, floor):
    pos = sum(d > 0 for d in deltas)
    mean = st.mean(deltas)
    if pos == len(deltas) and mean > floor:
        return "PASS", pos, mean
    if pos == 0 and -mean > floor:
        return "FAIL", pos, mean
    return "INCONCLUSIVE", pos, mean


def main():
    runs = {(arch, s): load(f"e15_{arch}_s{s}")
            for arch in ("dense", "moe") for s in SEEDS}
    missing = [k for k, v in runs.items() if v is None]
    if missing:
        print(f"[wait] {len(missing)} run(s) not finished yet: {missing}")
        if len(missing) == len(runs):
            return

    lines = ["# E15 结论：模型选择端点敏感性", "",
             "口径：27K、轻量(0.87M)、balanced loss、K=5 硬路由 top_k=2、",
             "max_epochs 20 / patience 10、4 seed 同卡配对。",
             "两套选择规则共享同一条训练轨迹（唯一变量 = 选了哪个 epoch 的权重）。", ""]

    report = {}
    for sel in SELECTIONS:
        rows, dm, dp, dense_macro = [], [], [], []
        ok = True
        for s in SEEDS:
            rd, rm = runs.get(("dense", s)), runs.get(("moe", s))
            if not rd or not rm:
                ok = False
                continue
            try:
                d = rd["test_by_selection"][sel]
                m = rm["test_by_selection"][sel]
            except KeyError:
                print(f"[skip] run seed={s} has no selection={sel} branch")
                ok = False
                continue
            dm.append(m["test"]["macro"] - d["test"]["macro"])
            dp.append(m["test"]["pooled"] - d["test"]["pooled"])
            dense_macro.append(d["test"]["macro"])
            rows.append((s, d["best_epoch"], m["best_epoch"],
                         d["test"]["macro"], m["test"]["macro"], dm[-1]))
        if not ok or len(dm) < 2:
            lines += [f"## selection = {sel} valid：数据不全，跳过", ""]
            continue
        floor = 2 * st.stdev(dense_macro)
        v, pos, mean = verdict(dm, floor)
        report[sel] = {"delta_macro": dm, "delta_pooled": dp, "floor": floor,
                       "mean": mean, "n_pos": pos, "verdict": v,
                       "x_floor": mean / floor if floor else None}
        lines += [f"## selection = {sel} valid AUC　→　**{v}**", "",
                  "| seed | dense 选中 ep | moe 选中 ep | dense test macro | "
                  "moe test macro | Δ(test macro) |", "|---|---|---|---|---|---|"]
        for s, ed, em, ad, am, d in rows:
            lines.append(f"| {s} | {ed} | {em} | {ad:.6f} | {am:.6f} | "
                         f"**{d:+.6f}** |")
        lines += ["",
                  f"- Δ 均值 **{mean:+.6f}**，同号正 seed **{pos}/{len(dm)}**",
                  f"- 噪声地板（本规则内）= 2×SD(dense test macro) = "
                  f"{floor:.6f} → 地板倍数 **{mean / floor:.2f}**",
                  f"- Δ(test pooled) 均值 {st.mean(dp):+.6f}，"
                  f"同号正 {sum(x > 0 for x in dp)}/{len(dp)}",
                  f"- **判定：{v}**", ""]

    # selection penalty (test macro lost by switching to the pooled rule)
    if all(s in report for s in SELECTIONS):
        lines += ["## 选择惩罚（test macro：pooled 选择 − macro 选择）", "",
                  "| arch | 均值 | 逐 seed |", "|---|---|---|"]
        for arch in ("dense", "moe"):
            pen = []
            for s in SEEDS:
                r = runs.get((arch, s))
                if not r or "test_by_selection" not in r:
                    continue
                tb = r["test_by_selection"]
                if "macro" in tb and "pooled" in tb:
                    pen.append(tb["pooled"]["test"]["macro"]
                               - tb["macro"]["test"]["macro"])
            if pen:
                lines.append(f"| {arch} | **{st.mean(pen):+.6f}** | "
                             + " ".join(f"{x:+.6f}" for x in pen) + " |")
        lines.append("")
        a, b = report["macro"]["verdict"], report["pooled"]["verdict"]
        outcome = ("A（两种选择规则都 PASS → 结论对选择规则稳健）"
                   if a == "PASS" and b == "PASS" else
                   "B（收益依赖 macro 选择 → 已有结论必须加限定词）"
                   if a == "PASS" else
                   "C（macro 选择也未 PASS → 触发复现哨兵排查）")
        lines += [f"## 预注册结局：**{outcome}**", ""]

    # reproduction sentinel vs E10
    lines += ["## 复现哨兵（E15 selection=macro 对 E10，容差 5e-4）", "",
              "| arch | seed | E15 | E10 | 差 | 判 |", "|---|---|---|---|---|---|"]
    sentinel_ok = True
    for arch in ("dense", "moe"):
        for s in SEEDS:
            r, ref = runs.get((arch, s)), load(E10_TAGS[arch].format(seed=s))
            if not r or not ref or "test_by_selection" not in r:
                continue
            if "macro" not in r["test_by_selection"]:
                continue
            a = r["test_by_selection"]["macro"]["test"]["macro"]
            b = ref["test"]["macro"]
            good = abs(a - b) < REPRO_TOL
            sentinel_ok &= good
            lines.append(f"| {arch} | {s} | {a:.6f} | {b:.6f} | {a - b:+.6f} | "
                         f"{'OK' if good else '**MISMATCH**'} |")
    lines += ["", f"哨兵：**{'通过' if sentinel_ok else '未通过 → E15 结果作废重查'}**", ""]

    md = "\n".join(lines)
    print(md)
    with open(os.path.join(OUT_DIR, "e15_report.md"), "w") as f:
        f.write(md)
    with open(os.path.join(OUT_DIR, "e15_decision.json"), "w") as f:
        json.dump({"report": report, "sentinel_ok": sentinel_ok,
                   "repro_tol": REPRO_TOL, "seeds": list(SEEDS)}, f, indent=2)
    print(f"\nwrote {OUT_DIR}/e15_report.md + e15_decision.json")


if __name__ == "__main__":
    main()
