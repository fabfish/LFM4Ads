"""分析 MoE Router 的门控分布（验证"Router 是否学到场景特定路由"）。

ScenarioRouter 的 gate 只依赖 tab（`model.py:59-60`）：

    p    = softmax(Embedding(tab))          # [K]，概率
    gate = p * K                            # [K]，均值恒为 1

因此**无需任何数据**，直接从 checkpoint 的 router embedding 即可精确复原
每个场景在每一层的门控向量与熵。这比"在测试集上统计平均 gate"更精确且零成本。

用法:
    python scripts/analyze_gate.py                    # 分析 cache/ 下全部 MoE checkpoint
    python scripts/analyze_gate.py cache/xxx.pt       # 指定 checkpoint

输出: stdout 表格 + cache/gate_analysis.json
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
DEFAULT_CKPTS = [
    "dcnv2_moe_k4.pt",
    "adatask_moe_none.pt",
    "adatask_moe_encourage.pt",
    "adatask_moe_suppress.pt",
]


def analyze(path: str) -> dict | None:
    if not os.path.exists(path):
        print(f"[skip] {os.path.relpath(path, ROOT)} 不存在")
        return None

    sd = torch.load(path, map_location="cpu", weights_only=True)
    # 两种历史键布局：
    #   router.embed.weight —— 当前 model.py（ScenarioRouter 内含 self.embed）
    #   router.weight       —— 旧版（router 直接就是 nn.Embedding），adatask_moe_*.pt 属此类
    keys = [k for k in sd if k.endswith("router.embed.weight")] \
        or [k for k in sd if k.endswith("router.weight")]
    if not keys:
        print(f"[skip] {os.path.relpath(path, ROOT)} 无 scenario router")
        return None

    print("\n" + "=" * 78)
    print(f"checkpoint: {os.path.relpath(path, ROOT)}")
    print("=" * 78)

    out = {}
    for key in sorted(keys):
        w = sd[key]                       # [num_scenarios, K]
        K = w.shape[1]
        max_ent = math.log(K)
        layer = key.split(".")[1] if key.startswith("cross_layers.") else key
        print(f"\n--- {key}   (K={K}, max entropy=log{K}={max_ent:.4f}) ---")
        print(f"{'S':>3} | " + " ".join(f"gate{e}" for e in range(K))
              + f" | {'entropy':>8} | {'max-min':>8}")
        rows = {}
        for s in SCENARIOS:
            p = torch.softmax(w[s], dim=-1)
            gate = (p * K).tolist()
            ent = float(-(p * (p + 1e-12).log()).sum())
            rows[s] = {"gate": [round(g, 4) for g in gate],
                       "entropy": round(ent, 6),
                       "entropy_deficit": round(max_ent - ent, 6),
                       "gate_spread": round(max(gate) - min(gate), 4)}
            print(f"{s:>3} | " + " ".join(f"{g:.3f}" for g in gate)
                  + f" | {ent:8.5f} | {max(gate) - min(gate):8.4f}")
        ents = [r["entropy"] for r in rows.values()]
        spread = [r["gate_spread"] for r in rows.values()]
        print(f"    entropy  min={min(ents):.5f}  max={max(ents):.5f}  "
              f"(max possible {max_ent:.5f})")
        print(f"    gate spread  min={min(spread):.4f}  max={max(spread):.4f}")
        out[key] = {"K": K, "max_entropy": round(max_ent, 6),
                    "per_scenario": rows,
                    "entropy_min": round(min(ents), 6),
                    "entropy_max": round(max(ents), 6),
                    "max_entropy_deficit_pct": round(
                        100 * (max_ent - min(ents)) / max_ent, 4),
                    "gate_spread_max": round(max(spread), 4)}

    worst_spread = max(v["gate_spread_max"] for v in out.values())
    worst_deficit = max(v["max_entropy_deficit_pct"] for v in out.values())
    # 以熵亏损占比判定：路由是否真的把容量分配给了不同场景
    verdict = ("Router 未分化（熵亏损 <1%，gate 全部接近 1）" if worst_deficit < 1 else
               "Router 轻微分化（熵亏损 1%-10%）" if worst_deficit < 10 else
               "Router 明显分化（熵亏损 >10%）")
    print(f"\n>>> 判定：{verdict}")
    print(f"    最大 gate 极差 = {worst_spread:.4f}   "
          f"最大熵亏损 = {worst_deficit:.4f}% of log(K)")
    out["_verdict"] = {"max_gate_spread": worst_spread,
                       "max_entropy_deficit_pct": worst_deficit,
                       "verdict": verdict}
    return out


targets = sys.argv[1:] or [os.path.join(CACHE, f) for f in DEFAULT_CKPTS]
report = {}
for t in targets:
    t = t if os.path.isabs(t) else os.path.join(ROOT, t)
    r = analyze(t)
    if r:
        report[os.path.relpath(t, ROOT)] = r

if report:
    with open(os.path.join(CACHE, "gate_analysis.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n机读报告已写入 cache/gate_analysis.json")
