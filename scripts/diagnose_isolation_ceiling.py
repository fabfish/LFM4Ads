"""E6 — the ISOLATION CEILING: can any amount of parameter isolation help?

The question E5 could not settle
--------------------------------
E5 (macro endpoint, 58 runs) came back INCONCLUSIVE: Δ_moe flips sign across
seeds and the measured noise floor is 0.0113 — 11x the pooled-AUC floor,
because macro AUC averages 8 scenarios and the small ones have few test rows.
The frozen-router sentinel proved the point: two *algebraically identical*
models differ by 0.0072 on macro purely from initialization randomness.

So instead of chasing a 0.005 effect with more seeds, ask the question that
bounds the whole family of methods:

    What is the BEST that parameter isolation could ever do?

Train one **completely independent model per scenario** (own embeddings, own
cross layers, own head, own early stopping, no parameter sharing at all). That
is the K = n_scenarios, hard-routed, zero-sharing limit — strictly more
isolation than any MoE can express. Compare each one against the *shared*
model's AUC on that same scenario.

Interpretation (registered before running)
------------------------------------------
* If isolated LOSES on every scenario -> sharing is strictly beneficial, there
  is no negative transfer to recover, and **no MoE variant can win**. The whole
  direction closes, with a mechanism, not another "didn't tune it well".
* If isolated WINS on some scenarios -> those scenarios genuinely suffer from
  interference, and that subset is exactly where conditional capacity should be
  applied. It also tells us the size of the prize.

Every choice here is deliberately biased IN FAVOUR of isolation, so that a loss
is conclusive:
  * per-scenario batch size (>= ~50 steps/epoch) so small scenarios still get
    many optimizer steps instead of 1-2 full-batch steps;
  * generous epoch budget + patience, early-stopped on that scenario's OWN
    valid AUC (the shared model is early-stopped on a global criterion);
  * best epoch by exact argmax.
"""

import json
import os
import sys
import time
from copy import deepcopy

import torch
from torcheval.metrics import BinaryAUROC
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import GpuBatches, Split  # noqa: E402
from model import BIG_ID_FIELDS, DCNv2, SubsetSparse, lightweight_dim  # noqa: E402

OUT_DIR = "cache/macro_auc"
OUT_JSON = f"{OUT_DIR}/e6_isolation_ceiling.json"
SCENARIOS = (0, 1, 2, 3, 4, 5, 6, 8)
SEEDS = (42, 123)
MAX_EPOCHS = 60
PATIENCE = 10


def auc(logit, label):
    if label.numel() < 32 or float(label.min()) == float(label.max()):
        return None
    m = BinaryAUROC()
    m.update(logit, label)
    return float(m.compute())


def evaluate(model, src):
    model.eval()
    ls, ys = [], []
    for batch in src:
        with torch.inference_mode():
            model(batch)
        ls.append(batch["logit"].float())
        ys.append(batch["is_click"].float())
    return auc(torch.cat(ls), torch.cat(ys))


def train_isolated(scenario, seed, device):
    train_set, valid_set, test_set = Split(scenario)
    n = len(train_set)
    # >= ~50 optimizer steps per epoch even for the tiny scenarios
    batch = int(min(10000, max(256, n // 50)))
    torch.manual_seed(seed)
    model = DCNv2(dim=lightweight_dim(BIG_ID_FIELDS))
    model.sparse = SubsetSparse(drop=BIG_ID_FIELDS)
    model = model.to(device)
    tr = GpuBatches(train_set, batch, device, shuffle=True, seed=seed)
    va = GpuBatches(valid_set, batch, device, shuffle=False)
    te = GpuBatches(test_set, batch, device, shuffle=False)
    del train_set, valid_set, test_set

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    best, best_state, since = -1.0, None, 0
    best_ep = 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for b in tqdm(tr, desc=f"s{scenario} seed{seed} ep{ep}", leave=False):
            model(b)
            crit(b["logit"], b["is_click"].float()).backward()
            opt.step()
            opt.zero_grad()
        v = evaluate(model, va)
        if v is None:
            raise SystemExit(f"scenario {scenario}: valid AUC undefined")
        if v > best:
            best, best_state, best_ep, since = v, deepcopy(
                model.state_dict()), ep, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {"scenario": scenario, "seed": seed, "test_auc": evaluate(model, te),
            "best_valid_auc": best, "best_epoch": best_ep,
            "batch_size": batch, "n_train": n}


def shared_baseline():
    """Per-scenario test AUC of the SHARED dense model (E5 stage-1 runs)."""
    out = {}
    for loss in ("pooled", "balanced"):
        per = {s: [] for s in SCENARIOS}
        for seed in (42, 123, 456, 789):
            path = f"{OUT_DIR}/run_s1_dense_{loss}_s{seed}.json"
            if not os.path.exists(path):
                continue
            ps = json.load(open(path))["test"]["per_scenario"]
            for s in SCENARIOS:
                if ps.get(str(s)) is not None:
                    per[s].append(ps[str(s)])
        out[loss] = {s: (sum(v) / len(v) if v else None)
                     for s, v in per.items()}
    return out


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_JSON):
        raise SystemExit(f"{OUT_JSON} exists; evidence bundles are immutable.")

    shared = shared_baseline()
    runs, t0 = [], time.time()
    for s in SCENARIOS:
        for seed in SEEDS:
            r = train_isolated(s, seed, device)
            runs.append(r)
            print(f"[isolated] s{s} seed{seed} test={r['test_auc']:.6f} "
                  f"(n_train={r['n_train']:,} batch={r['batch_size']} "
                  f"ep{r['best_epoch']})", flush=True)

    iso = {s: [r["test_auc"] for r in runs
               if r["scenario"] == s and r["test_auc"] is not None]
           for s in SCENARIOS}
    rows, wins = [], 0
    for s in SCENARIOS:
        if not iso[s] or shared["pooled"][s] is None:
            continue
        iso_mean = sum(iso[s]) / len(iso[s])
        sh = shared["pooled"][s]
        d = iso_mean - sh
        wins += d > 0
        rows.append({"scenario": s, "isolated_mean": iso_mean,
                     "shared_pooled": sh,
                     "shared_balanced": shared["balanced"][s],
                     "delta_isolated_minus_shared": d,
                     "n_train": next(r["n_train"] for r in runs
                                     if r["scenario"] == s)})

    bundle = {
        "provenance": {
            "script": "scripts/diagnose_isolation_ceiling.py",
            "device": device, "seeds": list(SEEDS),
            "scenarios": list(SCENARIOS),
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": 1e-3,
            "model": "iddrop DCNv2 (dim=330), one fully independent model "
                     "per scenario, zero parameter sharing",
            "baseline": "shared dense model from E5 stage 1 (seed-averaged "
                        "per-scenario test AUC)",
            "bias": "every setting favours the isolated arm (own batch size, "
                    "own early stopping, own valid criterion), so a loss is "
                    "conclusive",
            "interpretation": "isolated = the K=n_scenarios hard-routed "
                              "zero-sharing limit; it upper-bounds what any "
                              "MoE parameter isolation can achieve",
        },
        "runs": runs,
        "comparison": rows,
        "n_scenarios_where_isolation_wins": wins,
        "n_scenarios_compared": len(rows),
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(bundle, f, indent=2)

    print("\n" + "=" * 74)
    print("E6 隔离上界：每场景独立模型 vs 共享模型（per-scenario test AUC）")
    print("=" * 74)
    print(f"{'scen':>5}{'n_train':>10}{'isolated':>11}{'shared':>10}"
          f"{'Δ(iso-sh)':>12}")
    for r in rows:
        print(f"{'s' + str(r['scenario']):>5}{r['n_train']:>10,}"
              f"{r['isolated_mean']:>11.4f}{r['shared_pooled']:>10.4f}"
              f"{r['delta_isolated_minus_shared']:>+12.4f}")
    print(f"\n完全隔离胜出的场景数: {wins}/{len(rows)}")
    print(f"wrote {OUT_JSON}  ({bundle['wall_sec'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
