"""分析 AdaTask 三模式的 AU（Accumulated Update）分布集中度。

`encourage` 放大高 AU 专家的梯度、`suppress` 压制之，其**直接可观测后果**应体现在
AU 在场景维度上的集中程度。本脚本给出两个度量：

    max_ratio  —— 每个 (layer, expert) 上占比最高场景的占比（越高越集中）
    entropy    —— 该占比分布的香农熵（越低越集中）

⚠️ D1-E 修复（2026-08-09）：AU 由 backward hook 按 `tab.unique()` 累计，覆盖训练数据里
**全部**出现的场景（实际 13~14 个，含非目标 tab 如 7/9/10/11/12/14），而旧版只用 8 个
目标场景（`SCENARIOS`）取占比、且归一化上限写死 `log(8)`。修复后：占比与归一化**统一用
AU 实际累计的场景集**（`all_scn`），使"累计场景数"与"归一化分母"一致。

来源: cache/adatask_au_{none,encourage,suppress}.json
用法: python scripts/analyze_au.py
输出: stdout + cache/archives/adatask/au_analysis.json
"""

from __future__ import annotations

import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
# 目标场景（仅用于文档/其他引用；AU 归一化不再依赖它，见 D1-E 修复）
SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
MODES = ["none", "encourage", "suppress"]


def parse(path: str) -> dict[tuple[int, int, int], float]:
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        nums = [int(x) for x in
                str(k).replace("(", "").replace(")", "").replace("_", ",").split(",")
                if x.strip().lstrip("-").isdigit()]
        if len(nums) == 3:
            out[tuple(nums)] = float(v)
    return out


report = {}
print(f"{'mode':<10} {'n_scn':>6} {'max_ratio':>10} {'entropy':>9} "
      f"{'ent/logN':>9} {'S5_share':>9} {'coverage':>9}")

for mode in MODES:
    path = os.path.join(CACHE, f"adatask_au_{mode}.json")
    au = parse(path)
    # D1-E fix: use the ACTUALLY accumulated scenario set, not the 8 target ones.
    all_scn = sorted({k[2] for k in au})
    n_scn = len(all_scn)
    max_ent = math.log(n_scn)  # 归一化上限 = 实际累计场景数
    maxes, ents, s5 = [], [], []
    per_cell = {}
    for layer in sorted({k[0] for k in au}):
        for e in sorted({k[1] for k in au}):
            vals = [au.get((layer, e, s), 0.0) for s in all_scn]
            tot = sum(vals) + 1e-30
            p = [v / tot for v in vals]
            ent = -sum(x * math.log(x + 1e-12) for x in p if x > 0)
            maxes.append(max(p))
            ents.append(ent)
            s5.append(p[all_scn.index(5)] if 5 in all_scn else 0.0)
            per_cell[f"L{layer}E{e}"] = {
                "ratios": {s: round(x, 4) for s, x in zip(all_scn, p)},
                "max_ratio": round(max(p), 4),
                "argmax_scenario": all_scn[p.index(max(p))],
                "entropy": round(ent, 4),
            }
    # ---- 跨专家轴（AdaTaskOptimizer 实际作用的那一轴，model.py:313-341）----
    # 对每个 (layer, scenario)，看 AU 在 4 个专家之间的分布集中度（同样用 all_scn）
    K = len({k[1] for k in au})
    max_ent_e = math.log(K)
    e_max, e_ent = [], []
    for layer in sorted({k[0] for k in au}):
        for s in all_scn:
            vals = [au.get((layer, e, s), 0.0) for e in range(K)]
            tot = sum(vals) + 1e-30
            q = [v / tot for v in vals]
            e_max.append(max(q))
            e_ent.append(-sum(x * math.log(x + 1e-12) for x in q if x > 0))

    report[mode] = {
        "n_entries": len(au),
        "scenarios_tracked": all_scn,
        "n_scenarios_tracked": n_scn,
        "mean_max_ratio": round(statistics.mean(maxes), 4),
        "mean_entropy": round(statistics.mean(ents), 4),
        "entropy_ratio": round(statistics.mean(ents) / max_ent, 4),
        "entropy_norm": round(max_ent, 4),
        "mean_S5_share": round(statistics.mean(s5), 4),
        "cross_expert": {
            "mean_max_share": round(statistics.mean(e_max), 4),
            "mean_entropy": round(statistics.mean(e_ent), 4),
            "entropy_ratio": round(statistics.mean(e_ent) / max_ent_e, 4),
            "max_entropy": round(max_ent_e, 4),
        },
        "per_cell": per_cell,
    }
    r = report[mode]
    print(f"{mode:<10} {n_scn:>6} {r['mean_max_ratio']:>10.4f} {r['mean_entropy']:>9.4f} "
          f"{r['entropy_ratio']:>9.4f} {r['mean_S5_share']:>9.4f} "
          f"{r['n_scenarios_tracked']:>7} tabs")

print(f"\n(entropy 上限 = log(N)=log({n_scn:.0f})={max_ent:.4f}，N = AU 实际累计场景数；"
      f"缺陷 D1-E 已闭合：累计场景集 == 归一化分母，不再写死 log8)")

print("\n跨专家轴（AdaTaskOptimizer 直接作用的一轴：每个 (layer, scenario) 上 AU 在 4 个专家间的分布）")
print(f"{'mode':<10} {'max_share':>10} {'entropy':>9} {'ent/log4':>9}")
for mode in MODES:
    ce = report[mode]["cross_expert"]
    print(f"{mode:<10} {ce['mean_max_share']:>10.4f} {ce['mean_entropy']:>9.4f} "
          f"{ce['entropy_ratio']:>9.4f}")

print("\n逐 (layer, expert) 的 max_ratio")
cells = list(report[MODES[0]]["per_cell"])
print(f"{'cell':<8} " + " ".join(f"{m:>10}" for m in MODES))
for c in cells:
    print(f"{c:<8} " + " ".join(
        f"{report[m]['per_cell'][c]['max_ratio']:>10.4f}" for m in MODES))

argmax_all = {m: sorted({v["argmax_scenario"] for v in report[m]["per_cell"].values()})
              for m in MODES}
print(f"\nargmax 场景集合: {argmax_all}")

with open(os.path.join(CACHE, "au_analysis.json"), "w") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)
print("\n已写入 cache/archives/adatask/au_analysis.json")
