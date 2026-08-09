"""Phase 3 standalone run: train DCNv2MoE WITH SpecializationLoss.

Fixes the defects that previously made Phase 3 a no-op:
  - D1-A: spec-loss gate no longer waits for epoch%5 (MoE trains only 2-3 epochs).
  - D1-B: dominance ratios keyed as 3-tuple (li, ei, s) so compute() matches.
  - D5:   specialization loss is added on EACH sub-batch's own gate+tab, and
          sub["_gate"] is the live (non-detached) gate so the gradient reaches
          the router.

Why a custom loop instead of train_moe(): train_moe() early-stops and restores
the BEST epoch's weights. But spec loss only becomes enabled at the END of
epoch 1, so the best (epoch-1) checkpoint was never spec-trained -> identical
to Phase-1. Here we train a FIXED number of epochs and save the FINAL weights,
so the specialization loss actually has an effect to measure.
"""
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fields
from dataset import Dataset, Split
from model import DCNv2, DCNv2MoE, GradientTracker, SpecializationLoss
from train import evaluate

DEVICE = "cuda"
K = 4
EPOCHS = 6
CACHE = "cache"
VANILLA_PATH = f"{CACHE}/dcnv2_vanilla.pt"
PHASE1_MOE = f"{CACHE}/dcnv2_moe_k{K}.pt"
SPEC_PATH = f"{CACHE}/dcnv2_moe_k{K}_spec.pt"
RESULT_PATH = f"{CACHE}/phase3_spec_results.json"


def train_phase3(moe, tracker, spec_loss, epochs=EPOCHS):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(moe.parameters())
    device = next(moe.parameters()).device
    train_set, valid_set, test_set = Split("all")
    ratios = {}
    all_gates, all_tabs = [], []
    traj = []

    for epoch in range(1, epochs + 1):
        moe.train()
        all_gates.clear()
        all_tabs.clear()
        loader = torch.utils.data.DataLoader(
            Dataset(train_set), batch_size=10000, num_workers=10
        )
        for batch in loader:
            moe.train()
            for field in fields.all:
                batch[field] = batch[field].to(device).int()
            tab_batch = batch["tab"]

            for s in tab_batch.unique():
                mask = tab_batch == s
                si = int(s.item())
                sub = {k: v[mask] for k, v in batch.items()}
                if tracker is not None:
                    tracker.set_scenario(si)
                moe(sub)
                loss = criterion(sub["logit"], sub["is_click"].float())
                if spec_loss is not None and spec_loss.enabled:
                    loss = loss + spec_loss.compute(sub["_gate"], sub["tab"], ratios)
                loss.backward()
                if spec_loss is not None and "_gate" in sub:
                    all_gates.append(sub["_gate"])
                    all_tabs.append(sub["tab"])
            optimizer.step()
            optimizer.zero_grad()

        # Epoch-end: dominance ratios (3-tuple keys, D1-B) + enable gate (D1-A)
        if tracker is not None and spec_loss is not None:
            for li in range(3):
                dm = tracker.dominance_matrix(li)
                for (ei, s), v in dm.items():
                    ratios[(li, ei, s)] = v
            if not spec_loss.enabled:
                if spec_loss.check_and_enable(tracker):
                    print(f"[Epoch {epoch}] Specialization detected! Enabling spec loss.")
        if spec_loss is not None and all_gates:
            for sub_gates, sub_tab in zip(all_gates, all_tabs):
                spec_loss.update(sub_gates, sub_tab)

        valid_auc = float(evaluate(moe, valid_set).item())
        traj.append({"epoch": epoch, "valid_auc": valid_auc,
                     "spec_enabled": spec_loss.enabled})
        print(f"Epoch {epoch} valid AUC: {valid_auc:.4f}  spec_enabled={spec_loss.enabled}")

    test_auc = float(evaluate(moe, test_set).item())
    return test_auc, traj


# --- fresh MoE from cached vanilla (NOT from trained Phase-1 MoE)
vanilla = DCNv2().to(DEVICE)
vanilla.load_state_dict(torch.load(VANILLA_PATH, map_location=DEVICE))

moe = DCNv2MoE(dim=360, K=K).to(DEVICE)
moe.load_pretrained(vanilla)

tracker = GradientTracker(moe, beta=0.99)
tracker.register()
spec_loss = SpecializationLoss(threshold=0.3, lmbda=0.01)

print("=== Phase 3: training MoE WITH SpecializationLoss (D1-A/D1-B/D5 fixed) ===")
moe_auc, traj = train_phase3(moe, tracker, spec_loss)
print(f"MoE(spec) FINAL test AUC (all): {moe_auc:.4f}")
print(f"spec_loss.enabled = {spec_loss.enabled}")
print(tracker.summary())

torch.save(moe.state_dict(), SPEC_PATH)
print(f"saved checkpoint -> {SPEC_PATH}")

# --- per-scenario AUC: spec-MoE vs vanilla baseline vs Phase-1 MoE
scenarios = [0, 1, 2, 3, 4, 5, 6, 8]
spec_ps, van_ps, p1_ps = {}, {}, {}
moe1 = DCNv2MoE(dim=360, K=K).to(DEVICE)
moe1.load_state_dict(torch.load(PHASE1_MOE, map_location=DEVICE))
for s in scenarios:
    _, _, test = Split(s)
    spec_ps[s] = float(evaluate(moe, test).item())
    van_ps[s] = float(evaluate(vanilla, test).item())
    p1_ps[s] = float(evaluate(moe1, test).item())

result = {
    "spec_enabled": spec_loss.enabled,
    "spec_test_auc_all": moe_auc,
    "threshold": spec_loss.threshold,
    "lmbda": spec_loss.lmbda,
    "epoch_trajectory": traj,
    "per_scenario": {
        str(s): {
            "vanilla": van_ps[s],
            "moe_phase1": p1_ps[s],
            "moe_spec": spec_ps[s],
            "spec_minus_vanilla": spec_ps[s] - van_ps[s],
            "spec_minus_phase1": spec_ps[s] - p1_ps[s],
        }
        for s in scenarios
    },
    "mean": {
        "vanilla": sum(van_ps.values()) / len(van_ps),
        "moe_phase1": sum(p1_ps.values()) / len(p1_ps),
        "moe_spec": sum(spec_ps.values()) / len(spec_ps),
    },
}
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print("\n=== Mean AUC ===")
print(json.dumps(result["mean"], indent=2))
print(f"\nsaved results -> {RESULT_PATH}")
tracker.remove()
