"""E12 专家利用率诊断（预注册 docs/20260817-1215 §E12）。

从 run json 的 ``router_weights``（每层 Embedding(15,K).weight，由
experiments/main_macro_auc.py 在选出 best state 后 dump）重建 dispatch：

  gate 概率 = softmax(W, dim=-1)          # 每场景一行
  top-k 选择 = 每行概率最大的 k 个专家     # k = run json 的 top_k
  覆盖       = 15 场景选中专家的并集大小   # 满 = K
  负载       = 每 expert 被选中的场景数    # 均匀 = 15·k/K = 6 (k=2,K=5)
  路由熵     = 每场景 gate 分布的熵 (nats) # 均匀 = ln K = 1.609 (K=5)

判定（按 run json 的 top_k；跨 run/层取最保守聚合：coverage 取最小、
load_ratio 取最大，多 run 时同理；这是决策门而非效应判定，保守优先）：

  coverage == K 且 load_max/min <= 2  → differentiated   → 做 E13 (router 粒度)
  coverage <= 3 或 load_max/min > 3   → collapsed        → 先做 E12b (load-balance)
  其余                                 → partial         → E13 优先, E12b 补做

用法：
  python scripts/diagnose/diagnose_expert_usage.py \
      --runs cache/macro_auc_27k_siteB/run_b1_moe_tk2_s42.json [更多...] \
      --out cache/macro_auc_27k_siteB/expert_usage.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def diagnose_run(payload: dict) -> dict:
    import torch  # 延迟 import，诊断本身可在 CPU/任意机器跑

    weights = payload.get("router_weights")
    if not weights:
        raise SystemExit("run json 无 router_weights（需 scenario routing + "
                         "新版 main_macro_auc.py 训练的 run）")
    top_k = payload.get("top_k")
    K = len(weights[0][0])
    if top_k is None:
        top_k = K  # soft routing: all experts active
    n_scen = len(weights[0])
    uniform_entropy = math.log(K)
    uniform_load = n_scen * top_k / K

    layers = []
    for li, w in enumerate(weights):
        prob = torch.softmax(torch.tensor(w, dtype=torch.float64), dim=-1)
        top = prob.topk(top_k, dim=-1).indices            # [15, k]
        onehot = torch.zeros_like(prob).scatter_(-1, top, 1.0)
        load = onehot.sum(0)                              # [K]
        coverage = int((load > 0).sum())
        load_max, load_min = int(load.max()), int(load.min())
        load_ratio = (load_max / max(load_min, 1))
        ent = -(prob * prob.clamp_min(1e-12).log()).sum(-1)
        layers.append({
            "layer": li,
            "coverage": coverage,
            "expert_loads": [int(x) for x in load.tolist()],
            "load_max": load_max, "load_min": load_min,
            "load_ratio": round(load_ratio, 3),
            "uniform_load": uniform_load,
            "dispatch_topk": [[int(e) for e in row]
                              for row in top.tolist()],
            "entropy_per_scenario": [round(float(x), 3) for x in ent],
            "entropy_min": round(float(ent.min()), 3),
            "entropy_mean": round(float(ent.mean()), 3),
            "entropy_max": round(float(ent.max()), 3),
            "uniform_entropy": round(uniform_entropy, 3),
        })
    cov = min(l["coverage"] for l in layers)
    ratio = max(l["load_ratio"] for l in layers)
    if cov == K and ratio <= 2:
        verdict = "differentiated"
    elif cov <= 3 or ratio > 3:
        verdict = "collapsed"
    else:
        verdict = "partial"
    return {"top_k": top_k, "K": K, "verdict": verdict,
            "coverage_worst": cov, "load_ratio_worst": round(ratio, 3),
            "layers": layers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_run, verdicts = {}, []
    for path in args.runs:
        payload = json.loads(Path(path).read_text())
        res = diagnose_run(payload)
        per_run[Path(path).name] = res
        verdicts.append(res["verdict"])
        print(f"\n=== {Path(path).name} → {res['verdict']} ===")
        print(f"coverage(worst)={res['coverage_worst']} "
              f"load_ratio(worst)={res['load_ratio_worst']}")
        for l in res["layers"]:
            print(f"  layer {l['layer']}: coverage={l['coverage']}/5 "
                  f"loads={l['expert_loads']} (uniform {l['uniform_load']:.0f}) "
                  f"entropy[min/mean/max]="
                  f"{l['entropy_min']}/{l['entropy_mean']}/{l['entropy_max']} "
                  f"(uniform {l['uniform_entropy']})")

    if verdicts and all(v == verdicts[0] for v in verdicts):
        aggregate = verdicts[0]
    elif "collapsed" in verdicts:
        aggregate = "mixed_has_collapsed"
    else:
        aggregate = "mixed"
    out = {"aggregate_verdict": aggregate,
           "per_run_verdicts": verdicts,
           "decision_rule": "collapsed→E12b load-balance; "
                            "differentiated→E13 router granularity; "
                            "partial/mixed→E13 first, E12b follow-up",
           "per_run": per_run}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\naggregate={aggregate} → {args.out}")


if __name__ == "__main__":
    main()
