import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
  5. Run historical FeatureUsage exploration plus a random-init ModuleUsage
     placebo. This is not true MoE ModuleUsage/ModelUsage transfer evidence.
"""

import argparse
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
    apply_freeze,
    freeze_summary,
    trainable_parameters,
)
from train import evaluate, infer, train as train_fn


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="MoE V2 pretraining & evaluation")
    ap.add_argument("device", nargs="?", default="cuda",
                    help="e.g. cuda:0 (positional, backward compatible)")
    ap.add_argument("--freeze", default="",
                    help="comma list of groups to freeze, e.g. 'dnn,head,sparse' "
                         "(router+experts-only ablation)")
    ap.add_argument("--tag", default="",
                    help="suffix for all output artifacts, e.g. 'rxonly'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=4, help="number of routed experts")
    ap.add_argument("--top-k-target", type=int, default=2,
                    help="active routed experts after warmup")
    ap.add_argument("--routing", default="data", choices=["data", "scenario"])
    ap.add_argument("--noise-scale", type=float, default=0.1)
    ap.add_argument("--lb-alpha", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="AdamW lr; matters for the frozen (rx-only) setting")
    ap.add_argument("--beta2", type=float, default=0.999,
                    help="AdamW beta2; lower (e.g. 0.9) speeds up expert updates "
                         "diagnosed as under-trained (RMS(Expert)/RMS(DNN)<<1)")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the training loader; REQUIRED for --seed to "
                         "have any effect (weights come from a fixed ckpt and "
                         "the router is zero-init, so data order is the only "
                         "stochastic source)")
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--skip-downstream", action="store_true",
                    help="skip Step 5 (224 downstream trainings, very slow)")
    return ap.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])
DEVICE = ARGS.device
K = ARGS.K                       # number of routed experts
TOP_K_TARGET = ARGS.top_k_target  # target sparsity (active routed experts)
CACHE_DIR = "cache"
SUF = f"_{ARGS.tag}" if ARGS.tag else ""

torch.manual_seed(ARGS.seed)

os.makedirs(CACHE_DIR, exist_ok=True)
VANILLA_PATH = f"{CACHE_DIR}/dcnv2_vanilla.pt"
MOE_V2_PATH = f"{CACHE_DIR}/dcnv2_moe_v2_k{K}{SUF}.pt"
MOE_V1_PATH = f"{CACHE_DIR}/dcnv2_moe_k{K}.pt"
RESULT_CSV = f"result_moe_v2{SUF}.csv"
DOWNSTREAM_CSV = f"result_moe_v2_downstream{SUF}.csv"
GATE_JSON = f"{CACHE_DIR}/gate_stats_v2{SUF}.json"
HISTORY_JSON = f"{CACHE_DIR}/moe_v2_train_history{SUF}.json"

print(f"[config] device={DEVICE} seed={ARGS.seed} K={K} top_k_target={TOP_K_TARGET} "
      f"routing={ARGS.routing} freeze='{ARGS.freeze or 'none'}' tag='{ARGS.tag}' "
      f"skip_downstream={ARGS.skip_downstream}")


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
    routing=ARGS.routing, noise_scale=ARGS.noise_scale, lb_alpha=ARGS.lb_alpha,
).to(DEVICE)
moe_v2.load_pretrained(vanilla)
print(f"\n  {DCNv2MoE_V2.param_summary(moe_v2)}")

# ---- router+experts-only ablation: freeze everything but the MoE itself ----
freeze_info = apply_freeze(moe_v2, ARGS.freeze)
print()
print(freeze_summary(freeze_info))
print()

# Training with warmup + load balance loss
train_set, valid_set, test_set = Split("all")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(trainable_parameters(moe_v2), lr=ARGS.lr,
                              betas=(0.9, ARGS.beta2))
device = torch.device(DEVICE)
auc_best = 0.0
epoch = 0
history = []
warmup_auc = None
best_state, best_top_k = None, None

while True:
    epoch += 1
    loader = torch.utils.data.DataLoader(
        Dataset(train_set), batch_size=ARGS.batch_size, num_workers=ARGS.num_workers,
        shuffle=ARGS.shuffle,
    )
    total_steps = len(loader)
    lb_total_epoch = 0.0

    # ---- Warmup schedule ----
    # Epoch 1: top_k = K (soft routing, vanilla-equiv)
    # Epoch 2+: top_k = TOP_K_TARGET (sparse routing)
    cur_top_k = K if epoch == 1 else TOP_K_TARGET
    moe_v2.set_top_k(cur_top_k)
    phase = "Warmup" if cur_top_k >= K else "Sparse"
    print(f"  [{phase}] Epoch {epoch}: top_k={cur_top_k}")

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

    auc = float(evaluate(moe_v2, valid_set))
    lb_mean = lb_total_epoch / max(total_steps, 1)
    print(f"  Epoch {epoch} valid AUC: {auc:.4f}  LB loss: {lb_mean:.6f}")
    history.append({"epoch": epoch, "top_k": cur_top_k,
                    "valid_auc": auc, "lb_loss": lb_mean})

    # ---- Fix (2026-08-09): warmup(top_k=K) 与 sparse(top_k<K) 是两个不同的
    # 路由体制，用 warmup 的 AUC 去 early-stop 稀疏体制的第一个 epoch 会导致
    # 训练在 epoch 2 立刻中断，并且回滚到 warmup 权重却保留 top_k=TOP_K_TARGET
    # （权重/路由模式错配）。因此：体制切换时重置 early-stop 基线，
    # 并把 best 权重对应的 top_k 一起记录、恢复时一并还原。
    if epoch == 1 and TOP_K_TARGET < K:
        warmup_auc = auc
        print(f"  [regime switch] warmup valid AUC={auc:.4f} recorded; "
              f"early-stop baseline reset for the sparse regime")
        auc_best = 0.0
        continue

    if auc_best < auc - 0.001:
        auc_best = auc
        best_state = deepcopy(moe_v2.state_dict())
        best_top_k = cur_top_k
    else:
        break

    if epoch >= ARGS.max_epochs:
        print(f"  [max-epochs] stop at epoch {epoch}")
        break

if best_state is not None:
    moe_v2.load_state_dict(best_state)
    moe_v2.set_top_k(best_top_k)
    print(f"  restored best state (valid AUC={auc_best:.4f}, top_k={best_top_k})")

moe_v2_auc = evaluate(moe_v2, test_set)
print(f"  MoE V2 test AUC (all): {moe_v2_auc:.4f}")
if warmup_auc is not None:
    print(f"  (warmup-epoch valid AUC was {warmup_auc:.4f} vs best sparse "
          f"{auc_best:.4f})")
torch.save(moe_v2.state_dict(), MOE_V2_PATH)

with open(HISTORY_JSON, "w") as f:
    json.dump({
        "config": {
            "device": DEVICE, "seed": ARGS.seed, "K": K,
            "top_k_target": TOP_K_TARGET, "routing": ARGS.routing,
            "noise_scale": ARGS.noise_scale, "lb_alpha": ARGS.lb_alpha,
            "batch_size": ARGS.batch_size, "lr": ARGS.lr, "beta2": ARGS.beta2,
            "shuffle": ARGS.shuffle,
            "freeze": ARGS.freeze, "tag": ARGS.tag,
        },
        "freeze": freeze_info,
        "history": history,
        "warmup_valid_auc": warmup_auc,
        "best_valid_auc": auc_best,
        "best_top_k": best_top_k,
        "test_auc_all": float(moe_v2_auc),
        "vanilla_test_auc_all": float(vanilla_auc),
    }, f, indent=2)
print(f"  training history → {HISTORY_JSON}")


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
with open(RESULT_CSV, "w") as f:
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
#  Step 6: Expert specialization analysis
#  (moved BEFORE the expensive downstream step so the core
#   mechanism evidence is persisted even if Step 5 is skipped)
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

with open(GATE_JSON, "w") as f:
    json.dump(gate_json, f, indent=2)
print(f"\n  gate statistics → {GATE_JSON}")


# ===========================================================
#  Step 5: Downstream evaluation (FeatureUsage / ModuleUsage)
#  224 trainings — skip with --skip-downstream when only the
#  MoE mechanism question is being answered.
# ===========================================================
print()
print("=" * 60)
if ARGS.skip_downstream:
    print("Step 5: Downstream evaluation — SKIPPED (--skip-downstream)")
    print("=" * 60)
else:
    print("Step 5: Historical FeatureUsage + random-init placebo")
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

    with open(DOWNSTREAM_CSV, "w") as f:
        f.write("Model,Scenario,Method,AUC\n")

    for s in [1, 0, 4, 2, 6, 3, 8, 5]:
        with open(DOWNSTREAM_CSV, "a") as f:
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


print()
print("=" * 60)
print("Done! Results saved to:")
print(f"  {RESULT_CSV}                  — per-scenario AUC comparison")
if not ARGS.skip_downstream:
    print(f"  {DOWNSTREAM_CSV}           — downstream eval comparison")
print(f"  {GATE_JSON}    — expert gate statistics")
print(f"  {HISTORY_JSON} — training history / freeze summary")
print(f"  {MOE_V2_PATH}  — cached MoE V2 model")
