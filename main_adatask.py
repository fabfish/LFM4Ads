"""
AdaTask Optimizer for MoE: Encourage vs Suppress Expert Specialization
========================================================================
Compares three modes:
  - "none"      : Baseline MoE, no LR modulation
  - "encourage" : LR ∝ AU^α — amplify high-AU experts, push specialization
  - "suppress"  : LR ∝ (1/AU)^α — dampen dominant experts, force sharing

Key design: per-scenario sub-batch backward + AdaTask gradient modulation + optimizer step.
This ensures each scenario's gradient gets independently modulated before accumulation.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torcheval.metrics import BinaryAUROC

from dataset import Dataset, Split
from model import (
    AdaTaskOptimizer,
    DCNv2,
    DCNv2MoE,
    DCNv2CapacityMoE,
    apply_freeze,
    freeze_summary,
)
from train import infer


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="AdaTask: encourage vs suppress expert specialization")
    ap.add_argument("device", nargs="?", default="cuda")
    ap.add_argument("--freeze", default="",
                    help="comma list of groups to freeze, e.g. 'dnn,head,sparse'")
    ap.add_argument("--tag", default="", help="suffix for output artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=1,
                    help="the frozen (rx-only) setting needs several epochs")
    ap.add_argument("--vanilla-ckpt", default="cache/vanilla_pretrain.pt",
                    help="use cache/dcnv2_vanilla.pt to share the base with the "
                         "MoE V1/V2 experiments")
    ap.add_argument("--arch", default="dcnv2moe",
                    choices=["dcnv2moe", "capacity"],
                    help="dcnv2moe = historical zero-param MoE; "
                         "capacity = full-rank top-k sparse MoE")
    ap.add_argument("--top-k", type=int, default=2,
                    help="[capacity] active experts per token")
    ap.add_argument("--noise-scale", type=float, default=0.1,
                    help="[capacity] upcycle multiplicative perturbation std")
    ap.add_argument("--lb-alpha", type=float, default=0.001,
                    help="[capacity] load-balance loss coefficient")
    ap.add_argument("--warmup-epochs", type=int, default=2,
                    help="[capacity] epochs of full soft routing (top_k=K) "
                         "before sparsifying")
    return ap.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])
DEVICE = ARGS.device
K = ARGS.K
ALPHA = ARGS.alpha
BATCH_SIZE = ARGS.batch_size
OUTPUT_DIR = Path(__file__).parent / "cache"
OUTPUT_DIR.mkdir(exist_ok=True)
SUF = f"_{ARGS.tag}" if ARGS.tag else ""

torch.manual_seed(ARGS.seed)

SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]

print(f"[config] device={DEVICE} seed={ARGS.seed} K={K} alpha={ALPHA} lr={ARGS.lr} "
      f"epochs={ARGS.epochs} batch={BATCH_SIZE} freeze='{ARGS.freeze or 'none'}' "
      f"tag='{ARGS.tag}' vanilla={ARGS.vanilla_ckpt}")

# ============================================================
# Phase 0: Vanilla pretrain (or load cached)
# ============================================================
vanilla_path = Path(ARGS.vanilla_ckpt)
if vanilla_path.exists():
    print("Phase 0: Loading cached vanilla pretrain...")
    vanilla = DCNv2().to(DEVICE)
    vanilla.load_state_dict(torch.load(vanilla_path, weights_only=True))
    vanilla.requires_grad_(False)
else:
    print("Phase 0: Vanilla DCNv2 pretrain...")
    vanilla = DCNv2().to(DEVICE)
    from train import train
    train(vanilla, "all")
    vanilla.requires_grad_(False)
    torch.save(vanilla.state_dict(), vanilla_path)


def evaluate_auc(model: torch.nn.Module, scenario) -> float:
    """Evaluate AUC on the test set for a given scenario."""
    _, _, test = Split(scenario)
    met = BinaryAUROC().to(DEVICE)
    model.eval()
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            Dataset(test), batch_size=32768, shuffle=False
        ):
            batch = {k: v.to(DEVICE) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            model(batch)
            met.update(batch["logit"], batch["is_click"].float())
    return met.compute().item()


def routing_snapshot(model: torch.nn.Module, dataset) -> dict:
    """Collect router entropy + dispatch utilization on a dataset (eval mode).

    Reads the clean K-way softmax ``batch["_gate"]`` (3 layers) and the real
    dispatch counts ``batch["_dispatch_counts"]`` stored by
    ``DCNv2CapacityMoE.forward``. Entropy measures how far the router has left
    the uniform ``log K`` deadlock; dispatch utilization measures how tokens
    are spread across the K experts.

    Returns:
        entropy_per_layer: [h0, h1, h2] mean clean-gate entropy
        entropy_mean:      mean of the three layers
        dispatch_util:     3 × K fraction of dispatched tokens per expert
    """
    ent_acc = [0.0, 0.0, 0.0]
    disp_acc = [[0] * K for _ in range(3)]
    n_samples = 0
    for batch in infer(model, dataset):
        gates = batch.get("_gate", [])
        dc = batch.get("_dispatch_counts", [])
        B = batch["tab"].shape[0]
        for li in range(3):
            p = gates[li].clamp_min(1e-12)
            ent_acc[li] += (-(p * p.log()).sum(-1)).sum().item()
            if li < len(dc):
                for e in range(K):
                    disp_acc[li][e] += dc[li][e]
        n_samples += B
    entropy_per_layer = [e / max(n_samples, 1) for e in ent_acc]
    dispatch_util = []
    for li in range(3):
        total = sum(disp_acc[li])
        dispatch_util.append([c / max(total, 1) for c in disp_acc[li]])
    return {
        "entropy_per_layer": entropy_per_layer,
        "entropy_mean": sum(entropy_per_layer) / 3.0,
        "dispatch_util": dispatch_util,
    }


def train_adatask(mode: str) -> tuple[dict, dict]:
    """Train one MoE variant with AdaTaskOptimizer using per-scenario steps.

    Returns:
        per_scenario_auc: {scenario: AUC}
        au_dict: raw AU entries {(l,e,s): value}
    """
    print(f"\n{'=' * 60}")
    print(f"Training AdaTask MoE: mode = {mode}")
    print(f"{'=' * 60}")

    moe = DCNv2MoE(K=K).to(DEVICE)
    moe.load_pretrained(vanilla)

    freeze_info = apply_freeze(moe, ARGS.freeze)
    if ARGS.freeze:
        print(freeze_summary(freeze_info))

    opt = AdaTaskOptimizer(moe, lr=ARGS.lr, mode=mode,
                           alpha=ALPHA, beta=0.99, weight_decay=0.01)
    opt.register_hooks()

    train_set, valid_set = Split("all")[:2]
    all_data = pd.concat([train_set, valid_set])

    moe.train()
    n_batches = 0
    for epoch in range(ARGS.epochs):
        dl = torch.utils.data.DataLoader(Dataset(all_data), batch_size=BATCH_SIZE,
                                         shuffle=True)
        for batch in dl:
            tab = batch["tab"].to(DEVICE)
            for s_idx in tab.unique().tolist():
                mask = tab == s_idx
                if mask.sum() < 4:
                    continue
                mask_cpu = mask.cpu()
                sub = {k: v[mask_cpu].to(DEVICE)
                       for k, v in batch.items()
                       if isinstance(v, torch.Tensor)}

                opt.set_scenario(int(s_idx))
                moe(sub)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    sub["logit"], sub["is_click"].float())
                loss.backward()
                opt.modulate_and_zero_grad()

            n_batches += 1
            if n_batches % 20 == 0:
                print(f"  [{mode}] Batch {n_batches}, "
                      f"AU entries: {len(opt.AU)}, "
                      f"loss: {loss.item():.4f}")

    # Evaluate
    print(f"\nEvaluating {mode}...")
    per_scenario_auc = {}
    for s in SCENARIOS:
        auc = evaluate_auc(moe, s)
        per_scenario_auc[s] = auc
        print(f"  Scenario {s}: AUC = {auc:.4f}")

    # Save
    au_dict = {f"{li},{ei},{si}": v
               for (li, ei, si), v in opt.AU.items()}
    torch.save(moe.state_dict(), OUTPUT_DIR / f"adatask_moe_{mode}.pt")

    # Dominance summary
    print(opt.summary())

    opt.remove_hooks()
    del moe, opt
    torch.cuda.empty_cache()

    return per_scenario_auc, au_dict


def train_adatask_capacity(mode: str) -> tuple[dict, dict, list]:
    """Train one capacity-MoE variant with AdaTaskOptimizer (real sparse).

    Warmup (top_k=K soft routing) then sparse (top_k) dispatch. Records router
    entropy + dispatch utilization per epoch for the routing-specialization
    analysis — the new dimension that the historical all-compute MoE could not
    observe (unselected experts get no gradient, so their AU stays frozen).

    Returns:
        per_scenario_auc: {scenario: AUC}
        au_dict:          raw AU entries {(l,e,s): value} (e may be int or 'r')
        routing_history:  per-epoch {epoch, phase, top_k, entropy_mean,
                          entropy_per_layer, dispatch_util}
    """
    print(f"\n{'=' * 60}")
    print(f"Training AdaTask capacity-MoE: mode = {mode}")
    print(f"{'=' * 60}")

    moe = DCNv2CapacityMoE(
        dim=360, K=K, top_k=ARGS.top_k,
        noise_scale=ARGS.noise_scale, lb_alpha=ARGS.lb_alpha,
    ).to(DEVICE)
    moe.upcycle_from_dense(vanilla)

    opt = AdaTaskOptimizer(moe, lr=ARGS.lr, mode=mode,
                           alpha=ALPHA, beta=0.99, weight_decay=0.01)
    opt.register_hooks()

    train_set, valid_set = Split("all")[:2]
    all_data = pd.concat([train_set, valid_set])
    _, _, test_set = Split("all")

    routing_history = []
    n_batches = 0
    for epoch in range(1, ARGS.epochs + 1):
        cur_top_k = K if epoch <= ARGS.warmup_epochs else ARGS.top_k
        moe.set_top_k(cur_top_k)
        phase = "warmup" if cur_top_k >= K else "sparse"

        moe.train()
        dl = torch.utils.data.DataLoader(Dataset(all_data), batch_size=BATCH_SIZE,
                                         shuffle=True)
        for batch in dl:
            tab = batch["tab"].to(DEVICE)
            for s_idx in tab.unique().tolist():
                mask = tab == s_idx
                if mask.sum() < 4:
                    continue
                mask_cpu = mask.cpu()
                sub = {k: v[mask_cpu].to(DEVICE)
                       for k, v in batch.items()
                       if isinstance(v, torch.Tensor)}

                opt.set_scenario(int(s_idx))
                moe(sub)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    sub["logit"], sub["is_click"].float())
                lb = sub.get("_load_balance_loss",
                             torch.tensor(0.0, device=DEVICE))
                loss = loss + lb
                loss.backward()
                opt.modulate_and_zero_grad()

            n_batches += 1
            if n_batches % 20 == 0:
                print(f"  [{mode}][{phase}] Batch {n_batches}, "
                      f"AU entries: {len(opt.AU)}, loss: {loss.item():.4f}")

        snap = routing_snapshot(moe, test_set)
        print(f"  [{mode}][{phase}] Epoch {epoch} "
              f"entropy_mean={snap['entropy_mean']:.4f} (logK={math.log(K):.4f})")
        routing_history.append({
            "epoch": epoch, "phase": phase, "top_k": cur_top_k,
            "entropy_mean": snap["entropy_mean"],
            "entropy_per_layer": snap["entropy_per_layer"],
            "dispatch_util": snap["dispatch_util"],
        })

    # Evaluate per-scenario AUC on the final (sparse) state.
    print(f"\nEvaluating {mode}...")
    per_scenario_auc = {}
    for s in SCENARIOS:
        auc = evaluate_auc(moe, s)
        per_scenario_auc[s] = auc
        print(f"  Scenario {s}: AUC = {auc:.4f}")

    au_dict = {f"{li},{ei},{si}": v
               for (li, ei, si), v in opt.AU.items()}
    torch.save(moe.state_dict(),
               OUTPUT_DIR / f"adatask_capacity_moe_{mode}.pt")

    print(opt.summary())

    opt.remove_hooks()
    del moe, opt
    torch.cuda.empty_cache()

    return per_scenario_auc, au_dict, routing_history


# ============================================================
# Phase 1: Train all 3 variants
# ============================================================
MODES = ["none", "encourage", "suppress"]
all_auc: dict = {}
all_au: dict = {}
all_routing: dict = {}

for mode in MODES:
    if ARGS.arch == "capacity":
        aucs, aus, rhist = train_adatask_capacity(mode)
        all_routing[mode] = rhist
        with open(OUTPUT_DIR / f"adatask_capacity_routing_{mode}{SUF}.json",
                  "w") as f:
            json.dump(rhist, f)
    else:
        aucs, aus = train_adatask(mode)
    all_auc[mode] = aucs
    all_au[mode] = aus

    with open(OUTPUT_DIR / f"adatask_au_{mode}{SUF}.json", "w") as f:
        json.dump(aus, f)

# ============================================================
# Phase 2: Report results
# ============================================================
print(f"\n{'=' * 60}")
print("AdaTask MoE Results: Encourage vs Suppress Expert Specialization")
print(f"{'=' * 60}")

rows = []
for s in SCENARIOS:
    row = {"scenario": s}
    for mode in MODES:
        row[mode] = all_auc[mode][s]
    rows.append(row)
res_df = pd.DataFrame(rows)

print(res_df.to_string(index=False))

print(f"\n--- Mean AUC ---")
for mode in MODES:
    mean_auc = res_df[mode].mean()
    print(f"  {mode:10s}: {mean_auc:.4f}")

# Delta reporting
none_mean = res_df["none"].mean()
for mode in ["encourage", "suppress"]:
    delta = res_df[mode].mean() - none_mean
    print(f"  {mode:10s} Δ vs none: {delta:+.4f}")

res_df.to_csv(OUTPUT_DIR / f"adatask_results{SUF}.csv", index=False)
print(f"\nResults: {OUTPUT_DIR / f'adatask_results{SUF}.csv'}")
print(f"AU data:  {OUTPUT_DIR / f'adatask_au_*{SUF}.json'}")

# ============================================================
# Phase 3 (capacity only): routing specialization analysis
# ============================================================
if ARGS.arch == "capacity" and all_routing:
    print(f"\n{'=' * 60}")
    print("Routing specialization: entropy trajectory per mode "
          f"(log K = {math.log(K):.4f})")
    print(f"{'=' * 60}")
    for mode in MODES:
        traj = all_routing[mode]
        print(f"  {mode:10s}: " + "  ".join(
            f"e{r['epoch']}({r['phase'][0]})={r['entropy_mean']:.3f}"
            for r in traj))
    print("\n  dispatch utilization (final sparse epoch, layer 0):")
    for mode in MODES:
        last = all_routing[mode][-1]
        util = last["dispatch_util"][0]
        print(f"  {mode:10s}: " + "  ".join(f"E{e}:{u:.3f}"
                                            for e, u in enumerate(util)))
