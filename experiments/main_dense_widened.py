import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Dense-widened 4x Cross-layer capacity control (no routing).

Decisive control for the capacity-MoE "capacity is the bottleneck?" question.

Three arms, all sharing the pretrained frozen embeddings + DNN + head and all
starting their Cross layers from scratch (reinit-cross regime — the only regime
where extra Cross capacity has real headroom):

  A   : frozen dense baseline (read-only)
  A'  : dense-continued (1x Cross, reinitialized)
  B   : DenseWidenedDCNv2 (4x Cross hidden width, no router / no sparse / no lb)

If B's AUC > A's AUC, then cross-layer capacity IS a bottleneck and the MoE
router is the culprit; if B ≈ A', capacity is NOT the bottleneck and the
capacity-MoE route is closed on this task.

Reuses the GPU-resident data path (dataset.GpuBatches) and the full-batch loss,
so both arms run in seconds per epoch at ~100% GPU utilization.
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import torch
from tqdm import tqdm

from dataset import Dataset, GpuBatches, Split
from model import (
    DCNv2,
    DenseWidenedDCNv2,
    apply_freeze,
    freeze_summary,
    trainable_parameters,
)
from train import evaluate, evaluate_gpu

CACHE_DIR = "cache"
VANILLA_PATH = f"{CACHE_DIR}/checkpoints/dcnv2_vanilla.pt"


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="Dense-widened capacity control")
    ap.add_argument("device", nargs="?", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--width", type=int, default=2,
                    help="Cross hidden multiplier; hidden = dim * width "
                         "(width=2 => 720 => ~4x single-layer params, matching "
                         "4 full-rank experts)")
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--freeze", default="sparse",
                    help="freeze groups applied identically to both arms")
    ap.add_argument("--tag", default="", help="suffix for output artifacts")
    return ap.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])
DEVICE = ARGS.device
SUF = f"_{ARGS.tag}" if ARGS.tag else ""
RESULT_DENSE_CSV = f"results/dense_widened/result_widen_dense{SUF}.csv"
RESULT_WIDE_CSV = f"results/dense_widened/result_widen_wide{SUF}.csv"
HISTORY_JSON = f"{CACHE_DIR}/archives/dense_widened/widen_history{SUF}.json"

torch.manual_seed(ARGS.seed)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"[config] device={DEVICE} seed={ARGS.seed} lr={ARGS.lr} beta2={ARGS.beta2} "
      f"width={ARGS.width} batch_size={ARGS.batch_size} "
      f"max_epochs={ARGS.max_epochs} freeze='{ARGS.freeze}' tag='{ARGS.tag or 'none'}'")

if not os.path.exists(VANILLA_PATH):
    raise FileNotFoundError(f"{VANILLA_PATH} not found; run main.py first")

vanilla = DCNv2().to(DEVICE)
vanilla.load_state_dict(torch.load(VANILLA_PATH, map_location=DEVICE))
vanilla.requires_grad_(False)

# --- GPU-resident data sources (single feather read) ---
train_set, valid_set, test_set = Split("all")
TRAIN_SRC = GpuBatches(train_set, ARGS.batch_size, DEVICE,
                       shuffle=True, seed=ARGS.seed)
VALID_SRC = GpuBatches(valid_set, ARGS.batch_size, DEVICE, shuffle=False)
TEST_SRC = GpuBatches(test_set, ARGS.batch_size, DEVICE, shuffle=False)


def _eval(model, src):
    return float(evaluate_gpu(model, src))


def train_dense(model, train_src, valid_src, test_src, lr, beta2, max_epochs,
                freeze, result_csv, label):
    """Train a dense-style model (no router / no lb) on GPU-resident batches."""
    model.requires_grad_(True)
    info = apply_freeze(model, freeze)
    if freeze:
        print(f"  [{label}] " + freeze_summary(info).replace(
            "\n", f"\n  [{label}] "))
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=lr, betas=(0.9, beta2))
    criterion = torch.nn.BCEWithLogitsLoss()
    auc_best = 0.0
    best_state = None
    best_epoch = 0
    hist = []
    with open(result_csv, "w") as f:
        f.write("epoch,valid_auc,wall_clock_sec\n")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        for batch in tqdm(train_src, desc=f"[{label}] Epoch {epoch}",
                          leave=False):
            model.train()
            model(batch)
            loss = criterion(batch["logit"], batch["is_click"].float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        valid_auc = _eval(model, valid_src)
        wall = time.time() - t0
        print(f"  [{label}] Epoch {epoch} valid AUC={valid_auc:.4f} "
              f"wall={wall:.1f}s")
        with open(result_csv, "a") as f:
            f.write(f"{epoch},{valid_auc:.4f},{wall:.1f}\n")
        hist.append({"epoch": epoch, "valid_auc": valid_auc,
                     "wall_clock_sec": wall})
        if valid_auc > auc_best:
            auc_best = valid_auc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    test_auc = _eval(model, test_src)
    return test_auc, auc_best, best_epoch, hist


# ============================================================
#  Arm A: frozen dense baseline
# ============================================================
print("=" * 60)
print("Arm A: frozen dense baseline")
print("=" * 60)
dense_test_auc = _eval(vanilla, TEST_SRC)
print(f"  dense test AUC (frozen): {dense_test_auc:.4f}")

# ============================================================
#  Arm A': dense-continued with reinitialized 1x Cross layers
# ============================================================
print()
print("=" * 60)
print(f"Arm A': dense-continued (1x Cross, reinit) — {ARGS.max_epochs} epochs")
print("=" * 60)
vanilla.requires_grad_(True)
for _i in range(3):
    vanilla.layers[_i].reset_parameters()
dense_cont_test, dense_cont_best, dense_cont_best_ep, dense_cont_hist = \
    train_dense(vanilla, TRAIN_SRC, VALID_SRC, TEST_SRC,
                lr=ARGS.lr, beta2=ARGS.beta2, max_epochs=ARGS.max_epochs,
                freeze=ARGS.freeze, result_csv=RESULT_DENSE_CSV, label="dense")
print(f"  [dense-cont] test AUC: {dense_cont_test:.4f} "
      f"(best valid {dense_cont_best:.4f} @ epoch {dense_cont_best_ep})")

# ============================================================
#  Arm B: DenseWidenedDCNv2 (4x Cross hidden width, no router)
# ============================================================
print()
print("=" * 60)
print(f"Arm B: dense-widened (width={ARGS.width}, no router) — "
      f"{ARGS.max_epochs} epochs")
print("=" * 60)
widened = DenseWidenedDCNv2(dim=360, width=ARGS.width).to(DEVICE)
widened.init_from_dense(vanilla)  # copy sparse/dnn/head (Cross stays random)
print(f"\n  {DenseWidenedDCNv2.param_summary(widened)}\n")
wide_test, wide_best, wide_best_ep, wide_hist = \
    train_dense(widened, TRAIN_SRC, VALID_SRC, TEST_SRC,
                lr=ARGS.lr, beta2=ARGS.beta2, max_epochs=ARGS.max_epochs,
                freeze=ARGS.freeze, result_csv=RESULT_WIDE_CSV, label="wide")
print(f"  [widened] test AUC: {wide_test:.4f} "
      f"(best valid {wide_best:.4f} @ epoch {wide_best_ep})")

# ============================================================
#  Verdict
# ============================================================
print()
print("=" * 60)
print("Capacity-control verdict")
print("=" * 60)
print(f"  A   frozen dense      : {dense_test_auc:.4f}")
print(f"  A'  dense-cont (1x)   : {dense_cont_test:.4f}")
print(f"  B   dense-widened (4x): {wide_test:.4f}")
print(f"  Δ_capacity = B - A'   : {wide_test - dense_cont_test:+.4f}")

with open(HISTORY_JSON, "w") as f:
    json.dump({
        "config": {
            "device": DEVICE, "seed": ARGS.seed, "lr": ARGS.lr,
            "beta2": ARGS.beta2, "width": ARGS.width,
            "batch_size": ARGS.batch_size, "max_epochs": ARGS.max_epochs,
            "freeze": ARGS.freeze, "tag": ARGS.tag,
        },
        "dense_test_auc": dense_test_auc,
        "dense_cont_test_auc": dense_cont_test,
        "dense_cont_best_valid": dense_cont_best,
        "dense_cont_best_epoch": dense_cont_best_ep,
        "widened_test_auc": wide_test,
        "widened_best_valid": wide_best,
        "widened_best_epoch": wide_best_ep,
        "delta_capacity": wide_test - dense_cont_test,
        "dense_cont_history": dense_cont_hist,
        "widened_history": wide_hist,
    }, f, indent=2)

print(f"\nDone! Outputs:")
print(f"  {RESULT_DENSE_CSV} / {RESULT_WIDE_CSV} — per-epoch valid AUC")
print(f"  {HISTORY_JSON} — full history + delta_capacity")
