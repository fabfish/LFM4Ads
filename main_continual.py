"""Continual Learning: Base vs MoE on sequential scenario training.

Flow:
  1. Load pretrained vanilla DCNv2 and DCNv2MoE
  2. Sequential training: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 8
  3. After each task, evaluate on ALL scenarios
  4. Compute forgetting, forward transfer, and compare Base vs MoE
"""

import json
import os
import sys
from copy import deepcopy

import torch

from dataset import Split
from model import DCNv2, DCNv2MoE
from train import compute_forgetting, evaluate, train_continual

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
K = 4
CACHE_DIR = "cache"
SCENARIO_ORDER = [0, 1, 2, 3, 4, 5, 6, 8]

VANILLA_PATH = f"{CACHE_DIR}/dcnv2_vanilla.pt"
MOE_PATH = f"{CACHE_DIR}/dcnv2_moe_k{K}.pt"


def load_models():
    """Load pretrained vanilla and MoE models."""
    vanilla = DCNv2().to(DEVICE)
    vanilla.load_state_dict(torch.load(VANILLA_PATH, map_location=DEVICE))

    moe = DCNv2MoE(dim=360, K=K).to(DEVICE)
    moe.load_state_dict(torch.load(MOE_PATH, map_location=DEVICE))

    return vanilla, moe


def eval_all(model):
    """Evaluate on all test scenarios."""
    auc = {}
    for s in SCENARIO_ORDER:
        _, _, test_set = Split(s)
        auc[s] = evaluate(model, test_set).item()
    return auc


# ===========================================================
#  Main
# ===========================================================
print("=" * 60)
print("Continual Learning: Base vs MoE")
print("=" * 60)

# Check for cached models
if not os.path.exists(VANILLA_PATH):
    print("ERROR: Vanilla model not found. Run main_moe.py first.")
    sys.exit(1)
if not os.path.exists(MOE_PATH):
    print(f"ERROR: MoE model not found. Run main_moe.py first.")
    sys.exit(1)

vanilla, moe = load_models()

# Baseline evaluation (before any continual learning)
print("\n--- Pre-continual AUC (before any sequential training) ---")
vanilla_pre = eval_all(vanilla)
moe_pre = eval_all(moe)
for s in SCENARIO_ORDER:
    print(f"  scenario {s}: Vanilla={vanilla_pre[s]:.4f}  MoE={moe_pre[s]:.4f}")


# ===========================================================
#  Base: Continual learning on vanilla DCNv2
# ===========================================================
print()
print("=" * 60)
print("Base Continual Learning (Vanilla DCNv2)")
print("=" * 60)
print(f"Training order: {SCENARIO_ORDER}")

vanilla_results = train_continual(vanilla, "base", SCENARIO_ORDER)
vanilla_forgetting = compute_forgetting(vanilla_results)

print("\nBase forgetting summary:")
for f in vanilla_forgetting:
    print(f"  After training S{f['after_training_scenario']} (task {f['train_task']}), "
          f"S{f['eval_scenario']} forgetting: {f['forgetting']:+.4f}")


# ===========================================================
#  MoE: Continual learning on DCNv2MoE
# ===========================================================
print()
print("=" * 60)
print("MoE Continual Learning (DCNv2MoE)")
print("=" * 60)

moe_results = train_continual(moe, "moe", SCENARIO_ORDER)
moe_forgetting = compute_forgetting(moe_results)

print("\nMoE forgetting summary:")
for f in moe_forgetting:
    print(f"  After training S{f['after_training_scenario']} (task {f['train_task']}), "
          f"S{f['eval_scenario']} forgetting: {f['forgetting']:+.4f}")


# ===========================================================
#  Comparison
# ===========================================================
print()
print("=" * 60)
print("Base vs MoE Forgetting Comparison")
print("=" * 60)

# Align forgetting entries
base_f = {(f["train_task"], f["eval_task"]): f["forgetting"] for f in vanilla_forgetting}
moe_f = {(f["train_task"], f["eval_task"]): f["forgetting"] for f in moe_forgetting}

for key in base_f:
    bf = base_f[key]
    mf = moe_f.get(key, 0)
    if abs(bf) > 0.001 or abs(mf) > 0.001:
        delta = mf - bf  # negative = MoE forgets less
        print(f"  Task {key[0]}→{key[1]}: Base={bf:+.4f}  MoE={mf:+.4f}  Δ={delta:+.4f}  "
              f"({'MoE better' if delta > 0 else 'Base better'})")

mean_bf = sum(base_f.values()) / len(base_f)
mean_mf = sum(moe_f.values()) / len(moe_f)
print(f"\n  Mean forgetting: Base={mean_bf:+.4f}  MoE={mean_mf:+.4f}")
print(f"  Difference: {mean_mf - mean_bf:+.4f} "
      f"({'MoE has less forgetting' if mean_mf > mean_bf else 'Base has less forgetting'})")


# ===========================================================
#  Per-scenario trajectory
# ===========================================================
print()
print("=" * 60)
print("Per-scenario AUC trajectory over tasks")
print("=" * 60)

for s in SCENARIO_ORDER:
    print(f"\n  Scenario {s}:")
    for t in range(len(vanilla_results)):
        va = vanilla_results[t]["auc_per_scenario"].get(s, 0)
        ma = moe_results[t]["auc_per_scenario"].get(s, 0)
        train_s = vanilla_results[t]["train_scenario"]
        mark = "← current" if train_s == s else ""
        print(f"    After task {t} (train S{train_s}): Vanilla={va:.4f}  MoE={ma:.4f}  Δ={ma-va:+.4f}  {mark}")


# ===========================================================
#  Final AUC comparison
# ===========================================================
print()
print("=" * 60)
print("Final AUC after all continual learning")
print("=" * 60)

vanilla_final = eval_all(vanilla)
moe_final = eval_all(moe)

for s in SCENARIO_ORDER:
    vb, mb = vanilla_pre[s], moe_pre[s]
    vf, mf = vanilla_final[s], moe_final[s]
    print(f"  scenario {s}: "
          f"Vanilla={vf:.4f} (pre={vb:.4f})  "
          f"MoE={mf:.4f} (pre={mb:.4f})  "
          f"Δ_final={mf-vf:+.4f}")

mean_vf = sum(vanilla_final.values()) / len(vanilla_final)
mean_mf = sum(moe_final.values()) / len(moe_final)
print(f"  Mean final: Vanilla={mean_vf:.4f}  MoE={mean_mf:.4f}  Δ={mean_mf-mean_vf:+.4f}")


# ===========================================================
#  Save results
# ===========================================================
results = {
    "pre_continual": {
        "vanilla": vanilla_pre,
        "moe": moe_pre,
    },
    "vanilla_trajectory": [
        {"task": r["task"], "train_scenario": r["train_scenario"], "auc": r["auc_per_scenario"]}
        for r in vanilla_results
    ],
    "moe_trajectory": [
        {"task": r["task"], "train_scenario": r["train_scenario"], "auc": r["auc_per_scenario"]}
        for r in moe_results
    ],
    "vanilla_forgetting": [
        {"train_task": f["train_task"], "eval_task": f["eval_task"],
         "after_scenario": f["after_training_scenario"],
         "eval_scenario": f["eval_scenario"], "forgetting": f["forgetting"]}
        for f in vanilla_forgetting
    ],
    "moe_forgetting": [
        {"train_task": f["train_task"], "eval_task": f["eval_task"],
         "after_scenario": f["after_training_scenario"],
         "eval_scenario": f["eval_scenario"], "forgetting": f["forgetting"]}
        for f in moe_forgetting
    ],
    "final_auc": {
        "vanilla": vanilla_final,
        "moe": moe_final,
    },
}

with open(f"{CACHE_DIR}/continual_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {CACHE_DIR}/continual_results.json")
