"""E5 — Does scenario-routed MoE improve PER-SCENARIO generalization?

Why this experiment exists
--------------------------
Every previous MoE experiment in this repo optimised and measured **pooled
AUC**. A zero-cost decomposition of the frozen dense baseline showed that is
the wrong endpoint:

    pooled AUC                       = 0.7775
    scenario-prior-only AUC          = 0.6363   <- handed out for free
    in-scenario macro AUC            = 0.6736

Scenario 1 alone is 65.6% of training rows and per-scenario CTR spans
0.0206..0.5839 (26x), so most of pooled AUC is "which scenario is this",
not "which impression inside this scenario". Meanwhile the small scenarios
are broken: s7 = 0.4268 (worse than random ⇒ inverted), s12 = 0.5704,
s5 = 0.6764, s8 = 0.6799 versus s1 = 0.7315 — the signature of negative
transfer from one shared parameter set dominated by s1's gradients.

MoE's benefit mechanism (parameter isolation ⇒ less cross-scenario
interference) acts *entirely* on in-scenario ranking, while the old endpoint
is nearly blind to it: the four small scenarios are 0.5% of test rows, so
rescuing s7 from 0.43 to 0.73 moves pooled AUC by ~0.0001 — permanently
buried under the 0.001 noise floor. That, not tuning, is why no MoE case ever
"worked" here.

Design (2x2, fully paired)
--------------------------
  arch  : dense (one shared parameter set) vs moe (K scenario-routed experts)
  loss  : pooled (full-batch BCE, historical) vs balanced (equal weight per
          scenario present in the batch, aligned with the macro endpoint)

`model.DCNv2MoE`'s CrossExpertLayer splits Linear(dim,dim) into K
Linear(dim, dim/K) experts, so **total parameters are conserved**: moe minus
dense = 45*K router params only (K=5 ⇒ 225 of ~584k). Sweeping K therefore
varies isolation granularity at *constant capacity* — no capacity confound,
which matters because "capacity is not the bottleneck here" is already
established (dense-widened 4x, docs/20260814-2111).

Primary endpoint : macro AUC over MACRO_SCENARIOS (equal weight per scenario).
Model selection  : best epoch by macro *valid* AUC (aligned with the endpoint;
                   using pooled valid AUC would reintroduce the mismatch).
Reported also    : per-scenario AUC, pooled AUC, scenario-prior AUC.

Pre-registered in docs/20260815-0018-场景内泛化MoE长程矩阵预注册.md.
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from torcheval.metrics import BinaryAUROC
from tqdm import tqdm

import fields
from dataset import GpuBatches, Split
from model import (
    BIG_ID_FIELDS,
    DCNv2,
    DCNv2MoE,
    SubsetSparse,
    lightweight_dim,
)

CACHE_DIR = "cache"
#: overridable via LFM_MACRO_OUT so 1K and 27K evidence never mix
OUT_DIR = os.environ.get("LFM_MACRO_OUT", f"{CACHE_DIR}/macro_auc")
NUM_TABS = 15

#: Scenarios entering the macro endpoint. Frozen before launch.
#: Rationale: s9/s10/s13 are all-positive and s11 all-negative in test (AUC
#: undefined), s14 has zero test rows, and s7 has only 423 test rows (variance
#: too large for a primary endpoint). The remaining set is exactly the
#: historical GradientTracker.TARGET_SCENARIOS, so it is not chosen post-hoc.
MACRO_SCENARIOS = (0, 1, 2, 3, 4, 5, 6, 8)
#: Reported for completeness, never part of the verdict.
AUX_SCENARIOS = (7, 12)

#: Frozen sample counts for the split-conservation sentinel. Default = 1K;
#: override with ``LFM_SAMPLE_COUNTS_JSON`` pointing at a ``{"train": ..,
#: "valid": .., "test": ..}`` file (e.g. ``cache/sample_counts_27k.json``).
_COUNTS_OVERRIDE = os.environ.get("LFM_SAMPLE_COUNTS_JSON")
if _COUNTS_OVERRIDE and os.path.exists(_COUNTS_OVERRIDE):
    with open(_COUNTS_OVERRIDE) as _f:
        EXPECTED_COUNTS = {k: int(v) for k, v in json.load(_f).items()}
else:
    EXPECTED_COUNTS = {"train": 9_281_007, "valid": 1_230_368, "test": 1_201_670}


# ----------------------------------------------------------------------------
#  loss
# ----------------------------------------------------------------------------

def compute_loss(batch, weighting):
    """Pooled (full-batch mean) or scenario-balanced BCE.

    ``balanced`` = mean over the scenarios *present in this batch* of each
    scenario's own mean BCE. Fully vectorized via scatter_add_ (no ``.item()``,
    no boolean indexing) so there is no host sync in the training loop — the
    same discipline that gave the 11x throughput win in docs/20260814-2111.
    """
    logit, label = batch["logit"], batch["is_click"].float()
    if weighting == "pooled":
        return F.binary_cross_entropy_with_logits(logit, label)
    if weighting != "balanced":
        raise ValueError(f"unknown loss weighting {weighting!r}")
    per_row = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    tab = batch["tab"].long()
    sums = torch.zeros(NUM_TABS, device=per_row.device, dtype=per_row.dtype)
    cnts = torch.zeros_like(sums)
    sums.scatter_add_(0, tab, per_row)
    cnts.scatter_add_(0, tab, torch.ones_like(per_row))
    per_scenario_mean = sums / cnts.clamp(min=1.0)
    n_present = (cnts > 0).sum().clamp(min=1).to(per_row.dtype)
    return per_scenario_mean.sum() / n_present


# ----------------------------------------------------------------------------
#  evaluation
# ----------------------------------------------------------------------------

def _auc(logit, label):
    if label.numel() < 32:
        return None
    lo, hi = float(label.min()), float(label.max())
    if lo == hi:  # single-class ⇒ AUC undefined
        return None
    m = BinaryAUROC()
    m.update(logit, label)
    return float(m.compute())


def evaluate_all(model, src):
    """Return {pooled, macro, per_scenario{...}} in one pass."""
    model.eval()
    logits, labels, tabs = [], [], []
    for batch in src:
        with torch.inference_mode():
            model(batch)
        logits.append(batch["logit"].float())
        labels.append(batch["is_click"].float())
        tabs.append(batch["tab"])
    logit, label, tab = torch.cat(logits), torch.cat(labels), torch.cat(tabs)
    per = {}
    for s in list(MACRO_SCENARIOS) + list(AUX_SCENARIOS):
        mask = tab == s
        per[s] = _auc(logit[mask], label[mask])
    macro_vals = [per[s] for s in MACRO_SCENARIOS if per[s] is not None]
    return {
        "pooled": _auc(logit, label),
        "macro": sum(macro_vals) / len(macro_vals) if macro_vals else None,
        "n_macro_scenarios": len(macro_vals),
        "per_scenario": {str(k): v for k, v in per.items()},
    }


# ----------------------------------------------------------------------------
#  model construction
# ----------------------------------------------------------------------------

def freeze_router(model):
    """Pin the router at uniform (gate ≡ 1) — function-preservation sentinel.

    ``ScenarioRouter`` inits its embedding to zeros, so softmax is uniform and
    the ``* K`` scaling makes every gate exactly 1.0. With gates pinned at 1 the
    concatenated K experts are algebraically a single ``Linear(dim, dim)``, so a
    frozen-router MoE must match dense to within optimizer noise. Any large gap
    would mean the MoE arm differs from dense for reasons other than routing.
    """
    n = 0
    for layer in model.cross_layers:
        for p in layer.router.parameters():
            p.detach().zero_()
            p.requires_grad_(False)
            n += p.numel()
    return n


def build(arch, K, lightweight, device, top_k=None):
    drop = BIG_ID_FIELDS if lightweight else ()
    dim = lightweight_dim(drop) if lightweight else 360
    if arch == "dense":
        model = DCNv2(dim=dim)
    elif arch == "moe":
        if dim % K:
            raise SystemExit(
                f"dim={dim} not divisible by K={K}; the parameter-conserving "
                f"expert split requires dim % K == 0 (dim=330 ⇒ K in "
                f"{{2,3,5,6,10,11,...}})")
        model = DCNv2MoE(dim=dim, K=K, routing="scenario", top_k=top_k)
    else:
        raise SystemExit(f"unknown arch {arch!r}")
    if lightweight:
        model.sparse = SubsetSparse(drop=drop)
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    non_sparse = sum(p.numel() for n, p in model.named_parameters()
                     if not n.startswith("sparse."))
    router = sum(p.numel() for n, p in model.named_parameters()
                 if "router" in n)
    return model, {"arch": arch, "K": K if arch == "moe" else None, "dim": dim,
                   "top_k": top_k if arch == "moe" else None,
                   "total_params": total, "non_sparse_params": non_sparse,
                   "router_params": router}


def router_gate_mean(model, batch):
    """Mean gate value; ScenarioRouter is softmax*K so uniform init ⇒ 1.0."""
    if not hasattr(model, "cross_layers"):
        return None
    with torch.inference_mode():
        model(batch)
    gates = batch.get("_gate")
    if not gates:
        return None
    return float(torch.stack([g.mean() for g in gates]).mean())


# ----------------------------------------------------------------------------
#  training
# ----------------------------------------------------------------------------

def train_arm(model, srcs, args, label):
    train_src, valid_src, test_src = srcs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = {"macro": -1.0}
    best_state, hist, since_improve = None, [], 0
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        for i, batch in enumerate(
                tqdm(train_src, desc=f"[{label}] ep{epoch}", leave=False), 1):
            model(batch)
            compute_loss(batch, args.loss).backward()
            optimizer.step()
            optimizer.zero_grad()
            if args.max_batches and i >= args.max_batches:
                break
        val = evaluate_all(model, valid_src)
        wall = time.time() - t0
        hist.append({"epoch": epoch, "wall_clock_sec": wall,
                     "valid_macro": val["macro"], "valid_pooled": val["pooled"],
                     "valid_per_scenario": val["per_scenario"]})
        print(f"  [{label}] ep{epoch} valid macro={val['macro']:.6f} "
              f"pooled={val['pooled']:.6f} wall={wall:.1f}s")
        # model selection on the PRIMARY endpoint (macro), exact argmax
        if val["macro"] > best["macro"]:
            best = {"macro": val["macro"], "pooled": val["pooled"],
                    "epoch": epoch}
            best_state = deepcopy(model.state_dict())
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= args.patience:
                print(f"  [{label}] early stop at epoch {epoch} "
                      f"(patience={args.patience})")
                break
    model.load_state_dict(best_state)
    test = evaluate_all(model, test_src)
    return {"test": test, "best_valid_macro": best["macro"],
            "best_valid_pooled": best["pooled"], "best_epoch": best["epoch"],
            "epochs_run": len(hist), "history": hist,
            "mean_wall_sec": sum(h["wall_clock_sec"] for h in hist) / len(hist)}


def main():
    ap = argparse.ArgumentParser(description="E5 macro-AUC MoE experiment")
    ap.add_argument("device", nargs="?", default="cuda:0")
    ap.add_argument("--arch", default="dense", choices=("dense", "moe"))
    ap.add_argument("--loss", default="pooled", choices=("pooled", "balanced"))
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=None,
                    help="hard top-k sparsity; None/K = soft (all experts "
                         "active)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--full-embeddings", action="store_true",
                    help="keep the 84M dead-weight ID tables (default: drop, "
                         "per E1 PASS)")
    ap.add_argument("--freeze-router", action="store_true",
                    help="pin gates at uniform (function-preservation sentinel)")
    ap.add_argument("--max-batches", type=int, default=0, help="smoke only")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.tag or (f"{args.arch}_{args.loss}"
                       + (f"_K{args.K}" if args.arch == "moe" else "")
                       + (f"_tk{args.top_k}" if args.arch == "moe"
                          and args.top_k is not None else "")
                       + ("_frozen" if args.freeze_router else "")
                       + f"_s{args.seed}")
    out_json = f"{OUT_DIR}/run_{tag}.json"
    if os.path.exists(out_json):
        print(f"[skip] {out_json} already exists (resume-safe)")
        return

    lightweight = not args.full_embeddings
    print(f"[config] {tag} device={args.device} arch={args.arch} "
          f"loss={args.loss} K={args.K} seed={args.seed} lr={args.lr} "
          f"lightweight={lightweight} max_epochs={args.max_epochs} "
          f"patience={args.patience}")

    train_set, valid_set, test_set = Split("all")
    counts = {"train": len(train_set), "valid": len(valid_set),
              "test": len(test_set)}
    if counts != EXPECTED_COUNTS and not args.max_batches:
        raise SystemExit(f"S1 FAILED: {counts} != {EXPECTED_COUNTS}")
    print(f"[S1] sample counts {counts} PASS")

    srcs = (GpuBatches(train_set, args.batch_size, args.device,
                       shuffle=True, seed=args.seed),
            GpuBatches(valid_set, args.batch_size, args.device, shuffle=False),
            GpuBatches(test_set, args.batch_size, args.device, shuffle=False))
    del train_set, valid_set, test_set

    torch.manual_seed(args.seed)
    model, info = build(args.arch, args.K, lightweight, args.device,
                        top_k=args.top_k)
    info["router_frozen"] = bool(args.freeze_router)
    if args.freeze_router:
        if args.arch != "moe":
            raise SystemExit("--freeze-router only applies to --arch moe")
        info["router_params_frozen"] = freeze_router(model)
    probe = next(iter(srcs[1]))
    info["router_gate_mean_at_init"] = router_gate_mean(model, probe)
    print(f"  params total={info['total_params']:,} "
          f"non_sparse={info['non_sparse_params']:,} "
          f"router={info['router_params']:,} dim={info['dim']} "
          f"gate_mean_init={info['router_gate_mean_at_init']}")

    out = train_arm(model, srcs, args, tag)
    out.update(info)
    out["provenance"] = {
        "script": "main_macro_auc.py", "device": args.device,
        "arch": args.arch, "loss": args.loss, "K": args.K, "seed": args.seed,
        "top_k": args.top_k, "lr": args.lr, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "lightweight": lightweight, "freeze_router": bool(args.freeze_router),
        "sample_counts": counts,
        "macro_scenarios": list(MACRO_SCENARIOS),
        "aux_scenarios": list(AUX_SCENARIOS),
        "primary_endpoint": "macro AUC over macro_scenarios (test)",
        "model_selection": "exact argmax of macro valid AUC",
        "preregistration":
            "docs/20260815-0018-场景内泛化MoE长程矩阵预注册.md",
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    t = out["test"]
    print(f"\n[{tag}] TEST macro={t['macro']:.6f} pooled={t['pooled']:.6f} "
          f"(best valid macro {out['best_valid_macro']:.6f} "
          f"@ep{out['best_epoch']}, {out['epochs_run']} epochs run)")
    print("  per-scenario: " + " ".join(
        f"s{s}={t['per_scenario'][str(s)]:.4f}"
        for s in MACRO_SCENARIOS if t["per_scenario"][str(s)] is not None))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
