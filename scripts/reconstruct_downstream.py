"""Reconstruct result_moe_downstream.csv from cache/moe_pretrain.log.

The downstream evaluation (main_moe.py Step 5) produced full 8-scenario × 2-model
× 14-method = 224 records, all captured verbatim in `cache/moe_pretrain.log`
(lines like "  [Vanilla] scenario=1 method=SUM AUC=0.7305"). The standalone CSV
artifact was never persisted (only header survived an interrupted run), so we
reconstruct it directly from the authoritative log record — real experimental
output, no fabrication.

NOTE on provenance (D14): `cache/moe_pretrain.log` is the 05:16 run, whereas the
Phase-1 `result_moe.csv` / `cache/dominance_matrix.json` / cached checkpoints are
the 13:40 run. They are different runs; cross-comparison between the two must be
labeled explicitly (see docs/DRIVERS.md).
"""

import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "cache", "moe_pretrain.log")
OUT = os.path.join(ROOT, "result_moe_downstream.csv")

PAT = re.compile(r"\[(\w+)\]\s+scenario=(\d+)\s+method=(.+?)\s+AUC=([\d.]+)")

rows = []
with open(LOG) as fh:
    for line in fh:
        m = PAT.search(line)
        if not m:
            continue
        tag, scenario, method, auc = m.groups()
        rows.append({"Model": tag, "Scenario": int(scenario),
                     "Method": method, "AUC": float(auc)})

df = pd.DataFrame(rows, columns=["Model", "Scenario", "Method", "AUC"])
# main_moe.py declared order [1,0,4,2,6,3,8,5]; preserve it for readability
order = [1, 0, 4, 2, 6, 3, 8, 5]
df["_o"] = df["Scenario"].map({s: i for i, s in enumerate(order)})
df = df.sort_values(["_o", "Model", "Method"]).drop(columns="_o").reset_index(drop=True)

df.to_csv(OUT, index=False)
print(f"Reconstructed {len(df)} rows -> {OUT}")
print("Scenarios:", sorted(df['Scenario'].unique().tolist()))
print(df.groupby('Model')['AUC'].agg(['count', 'mean']).round(4))
