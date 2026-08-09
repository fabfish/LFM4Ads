"""Advanced MoE (V2) Pretraining & Evaluation.

基于 DCNv2MoE_V2: shared expert + noisy top-k routing + load balancing.

升级对比 V1 (Zero-Parameter MoE):
  V1:  K experts, soft routing, scenario embedding router
  V2:  1 shared expert + K routed experts, noisy top-k routing,
       load balancing aux loss, warmup → sparsify

Flow:
  1. Pretrain/load vanilla DCNv2 on all scenarios
  2. Initialize DCNv2MoE_V2 from vanilla weights (split Cross layers)
  3. Train MoE V2 with warmup + load balance loss
  4. Compare per-scenario AUC: Vanilla vs MoE V2 vs MoE V1
  5. Run downstream eval (FeatureUsage/ModuleUsage/ModelUsage)
"""

import json
import os
import sys
from copy import deepcopy

import pandas as pd
import torch
from tqdm import tqdm

import fields
from dataset import Dataset, Split
from model import (
    DCNv2,
    DCNv2MoE,
    DCNv2MoE_V2,
    FeatureUsage,
    ModelUsage,
    ModuleUsage,
)
from train import evaluate, infer, train as train_fn

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
K = 4  # number of routed experts
TOP_K_TARGET = 2  # target sparsity (active routed experts at inference)
CACHE_DIR = "cache"

os.makedirs(CACHE_DIR, exist_ok=True)
VANILLA_PATH = f"{CACHE_DIR}/dcnv2_vanilla.pt"
MOE_V2_PATH = f"{CACHE_DIR}/dcnv2_moe_v2_k{K}.pt"
MOE_V1_PATH = f"{CACHE_DIR}/dcnv2_moe_k{K}.pt"


# ===========================================================
#  Step 1: Pretrain/load vanilla DCNv2
# ===========================================================
print("=" * 60)
print("Step 1: Pretrain/load vanilla DCNv2 on all scenarios")
print("=" * 60)

if os.path.exists(VANILLA_PATH):
    print(f"  Loading cached vanilla model from {VANILLA_PATH}")
    vanilla = DCNv2().to(DEVICE)
    vanilla.load_state_dict(torch.load(VANILLA_PATH, map_location=DEVICE))
    vanilla_auc = evaluate(vanilla, Split("all")[2])
    print(f"  Vanilla test AUC (all): {vanilla_auc:.4f}")
else:
    vanilla = DCNv2().to(DEVICE)
    vanilla_auc = train_fn(vanilla, "all")
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
#  Step 3: Train DCNv2MoE_V2 with warmup + load balance
# ===========================================================
print()
print("=" * 60)
print(f"Step 3: Train DCNv2MoE_V2 (K={K}, top_k={TOP_K_TARGET})")
print(f"        Shared expert ({360//(K+1)}d) + {K} routed ({360//(K+1)}d each)")
print("        Noisy top-k gating + load balance loss")
print("=" * 60)

moe_v2 = DCNv2MoE_V2(
    dim=360, K=K, top_k=K,  # start full-soft (warmup)
    routing='data', noise_scale=0.1, lb_alpha=0.01,
).to(DEVICE)
moe_v2.load_pretrained(vanilla)
print(f"\n  {DCNv2MoE_V2.param_summary(moe_v2)}")
print()

# Training with warmup + load balance loss
train_set, valid_set, test_set = Split("all")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(moe_v2.parameters())
device = torch.device(DEVICE)
auc_best = 0
epoch = 0

while True:
    epoch += 1
    loader = torch.utils.data.DataLoader(
        Dataset(train_set), batch_size=10000, num_workers=10,
    )
    total_steps = len(loader)
    lb_total_epoch = 0.0

    # ---- Warmup schedule ----
    # Epoch 1: top_k = K = 4 (soft routing, vanilla-equiv)
    # Epoch 2+: top_k = TOP_K_TARGET (sparse routing)
    if epoch == 1:
        moe_v2.set_top_k(K)
        print(f"  [Warmup] Epoch {epoch}: top_k={K} (soft routing)")
    else:
        moe_v2.set_top_k(TOP_K_TARGET)
        print(f"  [Sparse]  Epoch {epoch}: top_k={TOP_K_TARGET}")

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        moe_v2.train()
        for field_name in fields.all:
            batch[field_name] = batch[field_name].to(device).int()
        tab_batch = batch["tab"]

        # Per-scenario forward+backward (consistent with train_moe)
        for s in tab_batch.unique():
            mask = tab_batch == s
            sub = {k: v[mask] for k, v in batch.items()}

            moe_v2(sub)
            loss = criterion(sub["logit"], sub["is_click"].float())

            # Add load balance loss
            lb_loss = sub.get("_load_balance_loss", torch.tensor(0.0, device=device))
            loss = loss + lb_loss
            lb_total_epoch += lb_loss.item()

            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    auc = evaluate(moe_v2, valid_set)
    print(f"  Epoch {epoch} valid AUC: {auc:.4f}  "
          f"LB loss: {lb_total_epoch / max(total_steps, 1):.6f}")

    if auc_best < auc - 0.001:
        auc_best = auc
        state_dict = deepcopy(moe_v2.state_dict())
    else:
        moe_v2.load_state_dict(state_dict)
        break

moe_v2_auc = evaluate(moe_v2, test_set)
print(f"  MoE V2 test AUC (all): {moe_v2_auc:.4f}")
torch.save(moe_v2.state_dict(), MOE_V2_PATH)


# ===========================================================
#  Step 4: Per-scenario AUC comparison (Vanilla vs MoE V2 vs MoE V1)
# ===========================================================
print()
print("=" * 60)
print("Step 4: Per-scenario AUC comparison")
print("=" * 60)

moe_v2_per_scenario = {}
for s in [0, 1, 2, 3, 4, 5, 6, 8]:
    _, _, test_set_s = Split(s)
    moe_v2_per_scenario[s] = evaluate(moe_v2, test_set_s).item()

# Also load MoE V1 if available for direct comparison
moe_v1_per_scenario = {}
if os.path.exists(MOE_V1_PATH):
    moe_v1 = DCNv2MoE(dim=360, K=K).to(DEVICE)
    moe_v1.load_state_dict(torch.load(MOE_V1_PATH, map_location=DEVICE))
    for s in [0, 1, 2, 3, 4, 5, 6, 8]:
        _, _, test_set_s = Split(s)
        moe_v1_per_scenario[s] = evaluate(moe_v1, test_set_s).item()
    del moe_v1

print(f"{'Scenario':>8}  {'Vanilla':>8}  {'MoE V2':>8}  {'Δ V2':>8}", end="")
if moe_v1_per_scenario:
    print(f"  {'MoE V1':>8}  {'Δ V1':>8}", end="")
print()

for s in [0, 1, 2, 3, 4, 5, 6, 8]:
    delta_v2 = moe_v2_per_scenario[s] - vanilla_per_scenario[s]
    line = f"{s:>8}  {vanilla_per_scenario[s]:>8.4f}  {moe_v2_per_scenario[s]:>8.4f}  {delta_v2:>+8.4f}"
    if moe_v1_per_scenario:
        delta_v1 = moe_v1_per_scenario[s] - vanilla_per_scenario[s]
        line += f"  {moe_v1_per_scenario[s]:>8.4f}  {delta_v1:>+8.4f}"
    print(line)

mean_vanilla = sum(vanilla_per_scenario.values()) / len(vanilla_per_scenario)
mean_v2 = sum(moe_v2_per_scenario.values()) / len(moe_v2_per_scenario)
line = f"{'Mean':>8}  {mean_vanilla:>8.4f}  {mean_v2:>8.4f}  {mean_v2-mean_vanilla:>+8.4f}"
if moe_v1_per_scenario:
    mean_v1 = sum(moe_v1_per_scenario.values()) / len(moe_v1_per_scenario)
    line += f"  {mean_v1:>8.4f}  {mean_v1-mean_vanilla:>+8.4f}"
print(line)

# Write comparison CSV
with open("result_moe_v2.csv", "w") as f:
    header = "Scenario,Vanilla_AUC,MoE_V2_AUC,Delta_V2"
    if moe_v1_per_scenario:
        header += ",MoE_V1_AUC,Delta_V1"
    f.write(header + "\n")
    for s in [0, 1, 2, 3, 4, 5, 6, 8]:
        line = f"{s},{vanilla_per_scenario[s]:.4f},{moe_v2_per_scenario[s]:.4f},{moe_v2_per_scenario[s]-vanilla_per_scenario[s]:+.4f}"
        if moe_v1_per_scenario:
            line += f",{moe_v1_per_scenario[s]:.4f},{moe_v1_per_scenario[s]-vanilla_per_scenario[s]:+.4f}"
        f.write(line + "\n")
    mean_line = f"Mean,{mean_vanilla:.4f},{mean_v2:.4f},{mean_v2-mean_vanilla:+.4f}"
    if moe_v1_per_scenario:
        mean_line += f",{mean_v1:.4f},{mean_v1-mean_vanilla:+.4f}"
    f.write(mean_line + "\n")


# ===========================================================
#  Step 5: Load MoE V1 for downstream comparison (if available)
#  OR: just run V2 downstream standalone
# ===========================================================
print()
print("=" * 60)
print("Step 5: Downstream evaluation (FeatureUsage/ModuleUsage)")
print("=" * 60)

# Aggregate CRs for MoE V2
moe_v2.CRs = torch.zeros(1000, 6, 360).to(DEVICE)
train_valid_set = pd.concat(Split("all")[:2])
for _ in infer(moe_v2, train_valid_set):
    pass
moe_v2.CRs = torch.nn.functional.layer_norm(moe_v2.CRs, [360])

vanilla.requires_grad_(False)
moe_v2.requires_grad_(False)

# Aggregate vanilla CRs if needed
if not hasattr(vanilla, "CRs"):
    vanilla.CRs = torch.zeros(1000, 6, 360).to(DEVICE)
    for _ in infer(vanilla, train_valid_set):
        pass
    vanilla.CRs = torch.nn.functional.layer_norm(vanilla.CRs, [360])


def run_downstream(Usage, LFM4Ads_model, method, scenario, tag):
    model = Usage(LFM4Ads_model, method).to(DEVICE)
    auc = train_fn(model, scenario)
    print(f"  [{tag}] scenario={scenario} method={method:>12} AUC={auc:.4f}")
    return f"{tag},{scenario},{method},{auc:.4f}\n"


header = "Model,Scenario,Method,AUC\n"
with open("result_moe_v2_downstream.csv", "w") as f:
    f.write(header)

for s in [1, 0, 4, 2, 6, 3, 8, 5]:
    with open("result_moe_v2_downstream.csv", "a") as f:
        # Vanilla downstream (baseline)
        f.write(run_downstream(FeatureUsage, vanilla, "SUM", s, "Vanilla"))
        for layer_idx in range(6):
            f.write(run_downstream(FeatureUsage, vanilla, f"concat CR_{layer_idx}", s, "Vanilla"))
        for layer_idx in range(6):
            f.write(run_downstream(FeatureUsage, vanilla, f"gate CR_{layer_idx}", s, "Vanilla"))
        f.write(run_downstream(ModuleUsage, vanilla, "Vanilla", s, "Vanilla"))

        # MoE V2 downstream
        f.write(run_downstream(FeatureUsage, moe_v2, "SUM", s, "MoE_V2"))
        for layer_idx in range(6):
            f.write(run_downstream(FeatureUsage, moe_v2, f"concat CR_{layer_idx}", s, "MoE_V2"))
        for layer_idx in range(6):
            f.write(run_downstream(FeatureUsage, moe_v2, f"gate CR_{layer_idx}", s, "MoE_V2"))
        f.write(run_downstream(ModuleUsage, moe_v2, "Vanilla", s, "MoE_V2"))


# ===========================================================
#  Step 6: Expert specialization analysis
# ===========================================================
print()
print("=" * 60)
print("Step 6: Expert specialization analysis")
print("=" * 60)

# Collect per-scenario gate statistics on test set
gate_stats = {}  # {scenario: [gate_vectors]}
moe_v2.eval()
for s in [0, 1, 2, 3, 4, 5, 6, 8]:
    _, _, test_set_s = Split(s)
    gates_by_layer = {li: [] for li in range(3)}
    for batch in infer(moe_v2, test_set_s):
        gates_batch = batch.get("_gate", [])
        for li in range(min(3, len(gates_batch))):
            gates_by_layer[li].append(gates_batch[li].detach().cpu())
    gate_stats[s] = {li: torch.cat(g, 0) if g else None
                     for li, g in gates_by_layer.items()}

# Print mean gate per scenario
print("\n  Mean gate activation per scenario (layer 0):")
header = "    " + "".join(f"   E{i}   " for i in range(K))
print(header)
for s in [0, 1, 2, 3, 4, 5, 6, 8]:
    if gate_stats[s][0] is not None:
        means = gate_stats[s][0].mean(0)
        gate_str = " ".join(f"{m:.4f}" for m in means)
        print(f"  S{s}: {gate_str}")

# Save gate stats
gate_json = {}
for s in [0, 1, 2, 3, 4, 5, 6, 8]:
    gate_json[f"scenario_{s}"] = {}
    for li in range(3):
        if gate_stats[s][li] is not None:
            gate_json[f"scenario_{s}"][f"layer_{li}_mean"] = \
                gate_stats[s][li].mean(0).tolist()
            gate_json[f"scenario_{s}"][f"layer_{li}_std"] = \
                gate_stats[s][li].std(0).tolist()

with open(f"{CACHE_DIR}/gate_stats_v2.json", "w") as f:
    json.dump(gate_json, f, indent=2)


print()
print("=" * 60)
print("Done! Results saved to:")
print("  result_moe_v2.csv             — per-scenario AUC comparison")
print("  result_moe_v2_downstream.csv  — downstream eval comparison")
print(f"  {CACHE_DIR}/gate_stats_v2.json — expert gate statistics")
print(f"  {CACHE_DIR}/dcnv2_vanilla.pt  — cached vanilla model")
print(f"  {CACHE_DIR}/dcnv2_moe_v2_k{K}.pt — cached MoE V2 model")
