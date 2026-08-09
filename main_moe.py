"""Zero-Parameter MoE Pretraining & Evaluation.

Flow:
  1. Pretrain vanilla DCNv2 on all scenarios (if not cached)
  2. Initialize DCNv2MoE from vanilla weights (split Cross layers)
  3. Train MoE with gradient tracking → detect expert specialization
  4. Compare per-scenario AUC: Vanilla vs MoE
  5. Run downstream eval (FeatureUsage/ModuleUsage/ModelUsage)
"""

import argparse
import json
import os
import sys
from copy import deepcopy

import pandas as pd
import torch

from dataset import Split
from model import (
    DCNv2,
    DCNv2MoE,
    FeatureUsage,
    GradientTracker,
    ModelUsage,
    ModuleUsage,
    SpecializationLoss,
    apply_freeze,
    freeze_summary,
)
from train import evaluate, infer, train, train_moe


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="MoE V1 pretraining & evaluation")
    ap.add_argument("device", nargs="?", default="cuda",
                    help="e.g. cuda:0 (positional, backward compatible)")
    ap.add_argument("--freeze", default="",
                    help="comma list of groups to freeze, e.g. 'dnn,head,sparse' "
                         "(router+experts-only ablation)")
    ap.add_argument("--tag", default="",
                    help="suffix for all output artifacts, e.g. 'rxonly'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=4,
                    help="number of experts; K=1 is architecturally identical to "
                         "vanilla DCNv2 (control for 'MoE split' vs 'which params "
                         "are trained')")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="AdamW lr; matters for the frozen (rx-only) setting")
    ap.add_argument("--spec-loss", action="store_true", default=True)
    ap.add_argument("--no-spec-loss", dest="spec_loss", action="store_false")
    ap.add_argument("--skip-downstream", action="store_true",
                    help="skip Step 5 (224 downstream trainings, very slow)")
    return ap.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])
DEVICE = ARGS.device
K = ARGS.K  # number of experts
CACHE_DIR = "cache"
SUF = f"_{ARGS.tag}" if ARGS.tag else ""

# K=1 collapses the router to a constant gate; the specialization loss would be
# trivially triggered (dominance == 1.0), so it is disabled automatically.
USE_SPEC_LOSS = ARGS.spec_loss and K > 1

torch.manual_seed(ARGS.seed)

os.makedirs(CACHE_DIR, exist_ok=True)
VANILLA_PATH = f"{CACHE_DIR}/dcnv2_vanilla.pt"
MOE_PATH = f"{CACHE_DIR}/dcnv2_moe_k{K}{SUF}.pt"
RESULT_CSV = f"result_moe_k{K}{SUF}.csv" if (K != 4 or SUF) else "result_moe.csv"
DOWNSTREAM_CSV = (f"result_moe_k{K}{SUF}_downstream.csv"
                  if (K != 4 or SUF) else "result_moe_downstream.csv")
DOMINANCE_JSON = f"{CACHE_DIR}/dominance_matrix_k{K}{SUF}.json" \
    if (K != 4 or SUF) else f"{CACHE_DIR}/dominance_matrix.json"
SUMMARY_JSON = f"{CACHE_DIR}/moe_v1_summary_k{K}{SUF}.json"

print(f"[config] device={DEVICE} seed={ARGS.seed} K={K} "
      f"freeze='{ARGS.freeze or 'none'}' tag='{ARGS.tag}' "
      f"spec_loss={USE_SPEC_LOSS} skip_downstream={ARGS.skip_downstream}")


# ===========================================================
#  Step 1: Pretrain vanilla DCNv2
# ===========================================================
print("=" * 60)
print("Step 1: Pretrain vanilla DCNv2 on all scenarios")
print("=" * 60)

if os.path.exists(VANILLA_PATH):
    print(f"  Loading cached vanilla model from {VANILLA_PATH}")
    vanilla = DCNv2().to(DEVICE)
    vanilla.load_state_dict(torch.load(VANILLA_PATH, map_location=DEVICE))
    vanilla_auc = evaluate(vanilla, Split("all")[2])  # test set
    print(f"  Vanilla test AUC (all): {vanilla_auc:.4f}")
else:
    vanilla = DCNv2().to(DEVICE)
    vanilla_auc = train(vanilla, "all")
    print(f"  Vanilla test AUC (all): {vanilla_auc:.4f}")
    torch.save(vanilla.state_dict(), VANILLA_PATH)


# ===========================================================
#  Step 2: Per-scenario vanilla AUC baseline
# ===========================================================
print()
print("=" * 60)
print("Step 2: Vanilla per-scenario AUC baseline")
print("=" * 60)

vanilla_per_scenario = {}
for s in [0, 1, 2, 3, 4, 5, 6, 8]:
    _, _, test_set = Split(s)
    vanilla_per_scenario[s] = evaluate(vanilla, test_set).item()
    print(f"  scenario {s}: {vanilla_per_scenario[s]:.4f}")


# ===========================================================
#  Step 3: Initialize MoE from vanilla + train with gradient tracking
# ===========================================================
print()
print("=" * 60)
print(f"Step 3: Train DCNv2MoE (K={K}) from vanilla init")
print("=" * 60)

moe = DCNv2MoE(dim=360, K=K).to(DEVICE)
moe.load_pretrained(vanilla)

# ---- router+experts-only ablation: freeze everything but the MoE itself ----
freeze_info = apply_freeze(moe, ARGS.freeze)
print()
print(freeze_summary(freeze_info))
print()

# Setup gradient tracker + specialization loss
tracker = GradientTracker(moe, beta=0.99)
tracker.register()
spec_loss = SpecializationLoss(threshold=0.3, lmbda=0.01) if USE_SPEC_LOSS else None

moe_auc = train_moe(moe, "all", tracker=tracker, spec_loss=spec_loss, lr=ARGS.lr)
print(f"  MoE test AUC (all): {moe_auc:.4f}")
torch.save(moe.state_dict(), MOE_PATH)

# Log dominance matrix
print()
print(tracker.summary())

# Save dominance data
dominance_data = {}
for li in range(3):
    dominance_data[f"layer_{li}"] = {
        f"E{ei}_S{s}": round(tracker.dominance_matrix(li).get((ei, s), 0), 4)
        for ei in range(K)
        for s in [0, 1, 2, 3, 4, 5, 6, 8]
    }
with open(DOMINANCE_JSON, "w") as f:
    json.dump(dominance_data, f, indent=2)

# Log specialization status
if spec_loss is None:
    print("\n  Specialization loss DISABLED (K=1 or --no-spec-loss)")
elif spec_loss.enabled:
    print(f"\n  Specialization loss ENABLED (threshold={spec_loss.threshold})")
else:
    print(f"\n  Specialization loss NOT triggered (no expert > {spec_loss.threshold})")

tracker.remove()


# ===========================================================
#  Step 4: Per-scenario MoE AUC comparison
# ===========================================================
print()
print("=" * 60)
print("Step 4: MoE per-scenario AUC vs Vanilla")
print("=" * 60)

moe_per_scenario = {}
for s in [0, 1, 2, 3, 4, 5, 6, 8]:
    _, _, test_set = Split(s)
    moe_per_scenario[s] = evaluate(moe, test_set).item()
    delta = moe_per_scenario[s] - vanilla_per_scenario[s]
    print(f"  scenario {s}: MoE={moe_per_scenario[s]:.4f}  Vanilla={vanilla_per_scenario[s]:.4f}  Δ={delta:+.4f}")

mean_vanilla = sum(vanilla_per_scenario.values()) / len(vanilla_per_scenario)
mean_moe = sum(moe_per_scenario.values()) / len(moe_per_scenario)
print(f"  Mean:   MoE={mean_moe:.4f}  Vanilla={mean_vanilla:.4f}  Δ={mean_moe-mean_vanilla:+.4f}")

# Write comparison CSV
with open(RESULT_CSV, "w") as f:
    f.write("Scenario,Vanilla_AUC,MoE_AUC,Delta\n")
    for s in [0, 1, 2, 3, 4, 5, 6, 8]:
        f.write(f"{s},{vanilla_per_scenario[s]:.4f},{moe_per_scenario[s]:.4f},"
                f"{moe_per_scenario[s]-vanilla_per_scenario[s]:+.4f}\n")
    f.write(f"Mean,{mean_vanilla:.4f},{mean_moe:.4f},{mean_moe-mean_vanilla:+.4f}\n")

with open(SUMMARY_JSON, "w") as f:
    json.dump({
        "config": {"device": DEVICE, "seed": ARGS.seed, "K": K, "lr": ARGS.lr,
                   "freeze": ARGS.freeze, "tag": ARGS.tag,
                   "spec_loss": USE_SPEC_LOSS},
        "freeze": freeze_info,
        "test_auc_all": float(moe_auc),
        "vanilla_test_auc_all": float(vanilla_auc),
        "per_scenario": {str(s): moe_per_scenario[s] for s in moe_per_scenario},
        "vanilla_per_scenario": {str(s): vanilla_per_scenario[s]
                                 for s in vanilla_per_scenario},
        "mean_moe": mean_moe, "mean_vanilla": mean_vanilla,
    }, f, indent=2)
print(f"  summary → {SUMMARY_JSON}")


# ===========================================================
#  Step 5: Downstream evaluation (Feature/Module/Model Usage)
# ===========================================================
print()
print("=" * 60)
if ARGS.skip_downstream:
    print("Step 5: Downstream evaluation — SKIPPED (--skip-downstream)")
    print("=" * 60)
    print()
    print("Done! Results saved to:")
    print(f"  {RESULT_CSV}      — per-scenario AUC comparison")
    print(f"  {DOMINANCE_JSON}  — expert dominance data")
    print(f"  {SUMMARY_JSON}    — config / freeze / AUC summary")
    print(f"  {MOE_PATH}        — cached MoE model")
    raise SystemExit(0)

print("Step 5: Downstream evaluation")
print("=" * 60)

# Aggregate CRs for MoE
moe.CRs = torch.zeros(1000, 6, 360).to(DEVICE)
train_valid_set = pd.concat(Split("all")[:2])
for batch in infer(moe, train_valid_set):
    pass
moe.CRs = torch.nn.functional.layer_norm(moe.CRs, [360])

vanilla.requires_grad_(False)
moe.requires_grad_(False)

# Aggregate vanilla CRs too (we need them for FeatureUsage)
if not hasattr(vanilla, "CRs"):
    vanilla.CRs = torch.zeros(1000, 6, 360).to(DEVICE)
    for batch in infer(vanilla, train_valid_set):
        pass
    vanilla.CRs = torch.nn.functional.layer_norm(vanilla.CRs, [360])


def run_downstream(Usage, LFM4Ads, method, scenario, tag):
    model = Usage(LFM4Ads, method).to(DEVICE)
    from train import train as train_fn
    auc = train_fn(model, scenario)
    line = f"{tag},{scenario},{method},{auc:.4f}\n"
    print(f"  [{tag}] scenario={scenario} method={method} AUC={auc:.4f}")
    return line


# Run downstream on both vanilla and MoE, writing to result_moe_downstream.csv
header = "Model,Scenario,Method,AUC\n"
with open(DOWNSTREAM_CSV, "w") as f:
    f.write(header)

for s in [1, 0, 4, 2, 6, 3, 8, 5]:
    with open(DOWNSTREAM_CSV, "a") as f:
        # Vanilla downstream
        f.write(run_downstream(FeatureUsage, vanilla, "SUM", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "concat CR_0", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "concat CR_1", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "concat CR_2", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "concat CR_3", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "concat CR_4", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "concat CR_5", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "gate CR_0", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "gate CR_1", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "gate CR_2", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "gate CR_3", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "gate CR_4", s, "Vanilla"))
        f.write(run_downstream(FeatureUsage, vanilla, "gate CR_5", s, "Vanilla"))
        f.write(run_downstream(ModuleUsage, vanilla, "Vanilla", s, "Vanilla"))

        # MoE downstream
        f.write(run_downstream(FeatureUsage, moe, "SUM", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "concat CR_0", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "concat CR_1", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "concat CR_2", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "concat CR_3", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "concat CR_4", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "concat CR_5", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "gate CR_0", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "gate CR_1", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "gate CR_2", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "gate CR_3", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "gate CR_4", s, "MoE"))
        f.write(run_downstream(FeatureUsage, moe, "gate CR_5", s, "MoE"))
        f.write(run_downstream(ModuleUsage, moe, "Vanilla", s, "MoE"))


print()
print("=" * 60)
print("Done! Results saved to:")
print(f"  {RESULT_CSV}      — per-scenario AUC comparison")
print(f"  {DOWNSTREAM_CSV}  — downstream eval comparison")
print(f"  {DOMINANCE_JSON}  — expert dominance data")
print(f"  {SUMMARY_JSON}    — config / freeze / AUC summary")
print(f"  {MOE_PATH}        — cached MoE model")
