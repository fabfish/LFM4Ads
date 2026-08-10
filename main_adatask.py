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


# ============================================================
# Phase 1: Train all 3 variants
# ============================================================
MODES = ["none", "encourage", "suppress"]
all_auc: dict = {}
all_au: dict = {}

for mode in MODES:
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
