"""Zero-Parameter MoE Pretraining & Evaluation.

Flow:
  1. Pretrain vanilla DCNv2 on all scenarios (if not cached)
  2. Initialize DCNv2MoE from vanilla weights (split Cross layers)
  3. Train MoE with gradient tracking → detect expert specialization
  4. Compare per-scenario AUC: Vanilla vs MoE
  5. Run downstream eval (FeatureUsage/ModuleUsage/ModelUsage)
"""

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
)
from train import evaluate, infer, train, train_moe

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
K = 4  # number of experts
CACHE_DIR = "cache"

os.makedirs(CACHE_DIR, exist_ok=True)
VANILLA_PATH = f"{CACHE_DIR}/dcnv2_vanilla.pt"
MOE_PATH = f"{CACHE_DIR}/dcnv2_moe_k{K}.pt"


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

# Setup gradient tracker + specialization loss
tracker = GradientTracker(moe, beta=0.99)
tracker.register()
spec_loss = SpecializationLoss(threshold=0.3, lmbda=0.01)

moe_auc = train_moe(moe, "all", tracker=tracker, spec_loss=spec_loss)
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
with open(f"{CACHE_DIR}/dominance_matrix.json", "w") as f:
    json.dump(dominance_data, f, indent=2)

# Log specialization status
if spec_loss.enabled:
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
with open("result_moe.csv", "w") as f:
    f.write("Scenario,Vanilla_AUC,MoE_AUC,Delta\n")
    for s in [0, 1, 2, 3, 4, 5, 6, 8]:
        f.write(f"{s},{vanilla_per_scenario[s]:.4f},{moe_per_scenario[s]:.4f},"
                f"{moe_per_scenario[s]-vanilla_per_scenario[s]:+.4f}\n")
    f.write(f"Mean,{mean_vanilla:.4f},{mean_moe:.4f},{mean_moe-mean_vanilla:+.4f}\n")


# ===========================================================
#  Step 5: Downstream evaluation (Feature/Module/Model Usage)
# ===========================================================
print()
print("=" * 60)
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
with open("result_moe_downstream.csv", "w") as f:
    f.write(header)

for s in [1, 0, 4, 2, 6, 3, 8, 5]:
    with open("result_moe_downstream.csv", "a") as f:
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
print("  result_moe.csv          — per-scenario AUC comparison")
print("  result_moe_downstream.csv — downstream eval comparison")
print(f"  {CACHE_DIR}/dominance_matrix.json — expert dominance data")
print(f"  {CACHE_DIR}/dcnv2_vanilla.pt — cached vanilla model")
print(f"  {CACHE_DIR}/dcnv2_moe_k{K}.pt — cached MoE model")
