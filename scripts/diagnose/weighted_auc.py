"""按 test 集样本数计算流量加权 AUC。

`result_moe.csv` 的 `Mean` 行是 8 个场景的**算术平均**，把 1,833 行的场景 5
和 803,191 行的场景 1 等权看待。真实业务关心的是流量加权效果，二者可能得出相反结论。

用法:
    python scripts/diagnose/weighted_auc.py                 # 读 dataset.feather 统计 test 样本数
    python scripts/diagnose/weighted_auc.py --cached        # 用 cache/provenance_report.json 里的 C9 结果

输出: stdout + cache/archives/sample_weighting/weighted_auc.json
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
TEST_FROM = 20220506


def test_counts() -> dict[int, int]:
    cached = os.path.join(ROOT, "cache", "provenance_report.json")
    if "--cached" in sys.argv and os.path.exists(cached):
        rep = json.load(open(cached))
        per_tab = rep["C9"]["detail"]["per_tab"]
        return {int(k): v["test"] for k, v in per_tab.items() if int(k) in SCENARIOS}
    df = pd.read_feather(os.path.join(ROOT, "dataset.feather"), columns=["tab", "date"])
    df = df[df["date"] >= TEST_FROM]
    return {s: int((df["tab"] == s).sum()) for s in SCENARIOS}


counts = test_counts()
total = sum(counts.values())
moe = pd.read_csv(os.path.join(ROOT, "results", "moe_exploration",
                               "result_moe.csv"))
rows = {str(r["Scenario"]): r for _, r in moe.iterrows()}

print(f"test 样本总数: {total:,}\n")
print(f"{'S':>3} {'test_n':>10} {'weight':>8} {'Vanilla':>9} {'MoE':>9} {'Delta':>9}")
for s in SCENARIOS:
    r = rows[str(s)]
    print(f"{s:>3} {counts[s]:>10,} {counts[s] / total:>8.4f} "
          f"{r['Vanilla_AUC']:>9.4f} {r['MoE_AUC']:>9.4f} {r['Delta']:>+9.4f}")

wv = sum(rows[str(s)]["Vanilla_AUC"] * counts[s] for s in SCENARIOS) / total
wm = sum(rows[str(s)]["MoE_AUC"] * counts[s] for s in SCENARIOS) / total
uv = sum(rows[str(s)]["Vanilla_AUC"] for s in SCENARIOS) / len(SCENARIOS)
um = sum(rows[str(s)]["MoE_AUC"] for s in SCENARIOS) / len(SCENARIOS)

out = {
    "test_counts": counts,
    "unweighted": {"vanilla": round(uv, 4), "moe": round(um, 4), "delta": round(um - uv, 4)},
    "traffic_weighted": {"vanilla": round(wv, 4), "moe": round(wm, 4),
                         "delta": round(wm - wv, 4)},
    "head_share": round((counts[0] + counts[1]) / total, 4),
}

print(f"\n{'口径':<22} {'Vanilla':>9} {'MoE':>9} {'Delta':>9}")
print(f"{'未加权（算术平均）':<18} {uv:>9.4f} {um:>9.4f} {um - uv:>+9.4f}")
print(f"{'流量加权（test 样本数）':<16} {wv:>9.4f} {wm:>9.4f} {wm - wv:>+9.4f}")
print(f"\n头部场景 (0+1) 占 test 流量 {out['head_share']:.2%}")
if (um - uv) * (wm - wv) < 0:
    print(">>> 两种口径结论**方向相反**，引用时必须写明口径！")

with open(os.path.join(ROOT, "cache", "weighted_auc.json"), "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)
print("\n已写入 cache/archives/sample_weighting/weighted_auc.json")
