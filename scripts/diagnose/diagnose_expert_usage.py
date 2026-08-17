"""E12 — expert utilisation diagnosis from the dumped router weights.

Why this is nearly free
-----------------------
`ScenarioRouter` (model.py:79-95) is `Embedding(num_scenarios, K)` and its gate
depends ONLY on `tab`, never on the sample content. So 3 layers x 15 scenarios
x K floats fully reproduce every dispatch decision the trained model would ever
make — no inference pass required. `main_macro_auc.py` dumps them as
`router_weights` from the SELECTED checkpoint.

Question: does hard top-k routing actually send different scenarios to
different experts, or do all scenarios collapse onto the same top_k experts?
The answer decides the next experiment (grain upgrade vs load balancing).

Frozen criteria (docs/20260817-1400-两端协作分离设计与实验分工.md §3 B0):
    coverage == K and load max/min <= 2   -> DIFFERENTIATED  (do B2)
    coverage <= 3 or load max/min > 3     -> COLLAPSED       (do B2b first)
    otherwise                             -> PARTIAL         (B2 first, B2b too)

Usage:
    python scripts/diagnose/diagnose_expert_usage.py                  # auto-find
    python scripts/diagnose/diagnose_expert_usage.py --run <run.json>
"""

import argparse
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: only the 8 pre-registered macro scenarios count toward the verdict; the
#: others are reported for completeness (they never enter the endpoint).
MACRO_SCENARIOS = (0, 1, 2, 3, 4, 5, 6, 8)
#: frozen thresholds — do not tune after seeing the numbers
COVERAGE_FULL_OK = 2.0     # load max/min <= 2 counts as balanced
LOAD_COLLAPSE = 3.0        # load max/min > 3 counts as collapsed


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def entropy(ps):
    return -sum(p * math.log(p) for p in ps if p > 0)


def analyse_layer(weight, top_k):
    """weight: [num_scenarios][K] raw router logits."""
    K = len(weight[0])
    per_scenario, dispatch, load = {}, {}, [0] * K
    for s in MACRO_SCENARIOS:
        probs = softmax(weight[s])
        order = sorted(range(K), key=lambda i: probs[i], reverse=True)
        chosen = order[:top_k]
        per_scenario[s] = {
            "gate_probs": [round(p, 6) for p in probs],
            "entropy": entropy(probs),
            "top_experts": chosen,
        }
        dispatch[s] = chosen
        for i in chosen:
            load[i] += 1
    covered = sorted({i for cs in dispatch.values() for i in cs})
    nz = [x for x in load if x > 0]
    ratio = (max(load) / min(nz)) if nz and min(nz) > 0 else float("inf")
    return {"K": K, "top_k": top_k, "per_scenario": per_scenario,
            "expert_load": load, "coverage": len(covered),
            "covered_experts": covered, "load_max_min_ratio": ratio,
            "mean_entropy": sum(v["entropy"] for v in per_scenario.values())
                            / len(per_scenario),
            "uniform_entropy": math.log(K)}


def verdict(layers):
    """Frozen decision rule, applied to the WORST layer (most collapsed)."""
    K = layers[0]["K"]
    worst_cov = min(l["coverage"] for l in layers)
    worst_ratio = max(l["load_max_min_ratio"] for l in layers)
    if worst_cov == K and worst_ratio <= COVERAGE_FULL_OK:
        return "DIFFERENTIATED", "粒度是瓶颈 → 做 B2（router 粒度升级）"
    if worst_cov <= 3 or worst_ratio > LOAD_COLLAPSE:
        return "COLLAPSED", "容量未用满 → 先做 B2b（load-balance loss）"
    return "PARTIAL", "B2 优先，B2b 补做"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="path to a run json")
    ap.add_argument("--out", default=None, help="output json path")
    args = ap.parse_args()

    path = args.run
    if not path:
        pats = [os.path.join(ROOT, "cache", "macro_auc_27k_siteB",
                             "run_b0repro_moe_tk2_s42.json"),
                os.path.join(ROOT, "cache", "macro_auc_27k",
                             "run_*moe*tk2*.json")]
        cands = [p for pat in pats for p in sorted(glob.glob(pat))]
        cands = [p for p in cands
                 if "router_weights" in json.load(open(p))]
        if not cands:
            raise SystemExit(
                "no run json with `router_weights` found. Run one moe top_k=2 "
                "run after the T4 dump was added (site B: stage b0repro).")
        path = cands[0]

    with open(path) as f:
        run = json.load(f)
    if "router_weights" not in run:
        raise SystemExit(f"{path} has no `router_weights` (pre-T4 run)")
    rw = run["router_weights"]
    top_k = run["provenance"].get("top_k") or run.get("K") or 2

    layers = [analyse_layer(w, top_k) for w in rw]
    v, action = verdict(layers)

    print(f"run      : {path}")
    print(f"site     : {run['provenance'].get('site', '?')}  "
          f"seed={run['provenance']['seed']}  top_k={top_k}  "
          f"K={layers[0]['K']}")
    print(f"test macro: {run['test']['macro']:.6f}\n")
    for li, l in enumerate(layers):
        print(f"--- cross layer {li} ---")
        print(f"  expert load (times chosen over {len(MACRO_SCENARIOS)} "
              f"scenarios): {l['expert_load']}   "
              f"coverage={l['coverage']}/{l['K']}  "
              f"max/min={l['load_max_min_ratio']:.2f}")
        print(f"  mean gate entropy {l['mean_entropy']:.4f} "
              f"(uniform = {l['uniform_entropy']:.4f})")
        for s in MACRO_SCENARIOS:
            d = l["per_scenario"][s]
            print(f"    s{s:<2} top{top_k}={d['top_experts']} "
                  f"H={d['entropy']:.4f} gates={d['gate_probs']}")
    print(f"\nVERDICT: **{v}** → {action}")
    print("（禁止因本诊断改判 E10/E11 的 PASS：那由 test macro 配对差独立支撑）")

    out = args.out or os.path.join(os.path.dirname(path),
                                   "expert_usage.json")
    with open(out, "w") as f:
        json.dump({"run": path, "verdict": v, "action": action,
                   "top_k": top_k, "macro_scenarios": list(MACRO_SCENARIOS),
                   "thresholds": {"coverage_full_load_ratio": COVERAGE_FULL_OK,
                                  "collapse_load_ratio": LOAD_COLLAPSE},
                   "layers": layers}, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
