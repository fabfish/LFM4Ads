"""从训练日志中提取三级下游评估结果（result_moe_downstream.csv 未跑完时的权威来源）。

用法:
    python scripts/extract_downstream.py [logfile]

默认解析 cache/moe_pretrain.log —— 该 run 完整跑完了 8 个场景 × 2 模型 × 全部方法，
而根目录 result_moe_downstream.csv 来自后续被中断的 run（缺 scenario 8/5）。
"""

from __future__ import annotations

import collections
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "cache", "moe_pretrain.log")
# method 名可能含空格（如 "concat CR_0"），必须非贪婪匹配到 " AUC="
PAT = re.compile(r"\[(Vanilla|MoE)\]\s+scenario=(\d+)\s+method=(.+?)\s+AUC=([0-9.]+)")

rows = []
with open(LOG) as fh:
    for ln in fh:
        m = PAT.search(ln)
        if m:
            rows.append((m.group(1), int(m.group(2)), m.group(3), float(m.group(4))))

print(f"log            : {os.path.relpath(LOG, ROOT)}")
print(f"total records  : {len(rows)}")
if not rows:
    sys.exit("no downstream records found")

scns = sorted({r[1] for r in rows})
methods = sorted({r[2] for r in rows})
print(f"scenarios      : {scns}")
print(f"methods ({len(methods)})   : {methods}")

pair = collections.defaultdict(dict)
for model, s, method, auc in rows:
    pair[(s, method)][model] = auc

complete = {k: v for k, v in pair.items() if len(v) == 2}
wins = sum(1 for v in complete.values() if v["MoE"] > v["Vanilla"])
rate = f"{wins / len(complete):.1%}" if complete else "n/a"
print(f"paired cells   : {len(complete)}  MoE wins {wins} ({rate})")
for model in ("Vanilla", "MoE"):
    vals = [a for m, _, _, a in rows if m == model]
    print(f"  {model:<8} n={len(vals):<4} mean={statistics.mean(vals):.4f}")

print("\nper-scenario mean AUC (paired methods only)")
print(f"{'S':>3} {'Vanilla':>9} {'MoE':>9} {'Delta':>9} {'MoE_win':>9}")
for s in scns:
    cells = [v for (ss, _), v in complete.items() if ss == s]
    if not cells:
        continue
    mv = statistics.mean(c["Vanilla"] for c in cells)
    mm = statistics.mean(c["MoE"] for c in cells)
    w = sum(1 for c in cells if c["MoE"] > c["Vanilla"])
    print(f"{s:>3} {mv:>9.4f} {mm:>9.4f} {mm - mv:>+9.4f} {w:>5}/{len(cells)}")

for target in (8, 5):
    print(f"\n--- scenario {target} per-method ---")
    for me in methods:
        v = pair.get((target, me), {})
        if len(v) == 2:
            print(f"  {me:<24} V={v['Vanilla']:.4f}  M={v['MoE']:.4f}  "
                  f"d={v['MoE'] - v['Vanilla']:+.4f}")
