"""Run a single downstream transfer trial (one arm x target x seed).

Protocol (docs/archive/drivers/20260812-2303-...):
  - load frozen source backbone (dense or moe) from cache/downstream_transfer/source
  - build the transfer arm, train on the frozen 3072 fit rows, early-stop on 1024 val
  - evaluate ONCE on the frozen target test set
  - write an immutable per-trial manifest (hashes + metrics)

Usage:
  python scripts/matrix/run_downstream_transfer_trial.py --arm moe-router --target 2 --seed 42 --device cuda:1
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC

from downstream_transfer_protocol import (
    TRANSFER_ARMS, TARGET_SCENARIOS, build_downstream_datasets,
    build_transfer_arm, load_source_backbone, trainable_parameter_count,
    frozen_parameter_sha256, source_checkpoint_path,
)

AUDIT_JSON = Path("cache/audit/downstream_transfer/protocol_invariants.json")
TRIALS_DIR = Path("cache/downstream_transfer/trials")


def _to_device(batch, device):
    return {
        k: (v.to(device) if k == "is_click" else v.to(device).int())
        for k, v in batch.items()
    }


def evaluate_auc(model, dataset, device, batch_size: int = 512) -> float:
    model.eval()
    metric = BinaryAUROC()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = _to_device(batch, device)
            model(x)
            metric.update(x["logit"].float().cpu(), x["is_click"].float().cpu())
    return float(metric.compute())


def train_trial(model, fit_ds, val_ds, device, seed, lr=5e-4, max_epochs=50,
                patience=5, min_delta=1e-4, batch_size=512):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(1000 + seed)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val = float("-inf")
    best_state = None
    patience_count = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in DataLoader(fit_ds, batch_size=batch_size, shuffle=True,
                                generator=generator):
            x = _to_device(batch, device)
            model(x)
            loss = criterion(x["logit"], x["is_click"].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_auc = evaluate_auc(model, val_ds, device)
        history.append({"epoch": epoch, "val_auc": val_auc})
        if val_auc > best_val + min_delta:
            best_val = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, history


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=TRANSFER_ARMS)
    ap.add_argument("--target", required=True, type=int, choices=TARGET_SCENARIOS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epochs", type=int, default=50)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    src_type = "dense" if args.arm.startswith("dense") else "moe"
    backbone = load_source_backbone(src_type, args.seed, device)
    if src_type == "dense":
        model = build_transfer_arm(args.arm, backbone, None)
    else:
        model = build_transfer_arm(args.arm, None, backbone)
    model.to(device)
    n_train = trainable_parameter_count(model)

    fit_ds, val_ds, test_ds, split = build_downstream_datasets(args.target, args.seed)
    t0 = time.time()
    best_val, history = train_trial(
        model, fit_ds, val_ds, device, args.seed,
        lr=args.lr, max_epochs=args.max_epochs,
    )
    test_auc = evaluate_auc(model, test_ds, device)

    result = {
        "arm": args.arm, "target": args.target, "seed": args.seed,
        "source_type": src_type, "source_seed": args.seed,
        "device": args.device,
        "trainable_param_count": n_train,
        "best_val_auc": best_val,
        "test_auc": test_auc,
        "seconds": round(time.time() - t0, 1),
        "epochs_run": len(history),
        "fit_sha256": split.fit_sha256,
        "validation_sha256": split.validation_sha256,
        "test_sha256": split.test_sha256,
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path(src_type, args.seed)),
        "frozen_param_sha256": frozen_parameter_sha256(backbone),
        "audit_protocol_sha256": sha256_file(AUDIT_JSON),
    }
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    out = TRIALS_DIR / f"{args.arm}_t{args.target}_s{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[trial] {args.arm} t{args.target} s{args.seed} -> "
          f"test_auc={test_auc:.4f} val={best_val:.4f} trainable={n_train} ({out.name})")


if __name__ == "__main__":
    main()
