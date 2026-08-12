"""Source-pretrain a frozen backbone on SOURCE_SCENARIOS for the downstream route.

Pre-registered protocol (docs/archive/drivers/20260812-2303-...):
  - train on source scenarios [0,1,3,4,8] only (targets [2,5,6] never touched)
  - pooled per-sample BCE, batch_size 10000, AdamW lr=1e-3 betas=(0.9,0.999)
  - early-stop on source validation AUC (no improvement by 0.001)
  - save backbone state_dict (no downstream head) for downstream transfer

Usage:
  python scripts/run_downstream_source_pretrain.py --model dense --device cuda:1 --seed 42
  python scripts/run_downstream_source_pretrain.py --model moe   --device cuda:1 --seed 42
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC

import fields
from dataset import DATASET_PATH
from model import DCNv2, DCNv2MoE_LowRank
from downstream_transfer_protocol import SOURCE_SCENARIOS, source_checkpoint_path


class _BatchDataset(torch.utils.data.Dataset):
    """A torch Dataset that keeps the raw columns as a batch dict."""

    def __init__(self, rows: pd.DataFrame):
        self.batch: dict[str, torch.Tensor] = {}
        for field in fields.user:
            self.batch[field] = torch.as_tensor(rows[field].to_numpy(), dtype=torch.long)
        for field in fields.video:
            self.batch[field] = torch.as_tensor(rows[field].to_numpy(), dtype=torch.long)
        self.batch["tab"] = torch.as_tensor(rows["tab"].to_numpy(), dtype=torch.long)
        self.batch["user_id"] = torch.as_tensor(rows["user_id"].to_numpy(), dtype=torch.long)
        self.batch["is_click"] = torch.as_tensor(rows["is_click"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.batch["is_click"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.batch.items()}


def _build_dataset(period: str) -> _BatchDataset:
    frame = pd.read_feather(DATASET_PATH, columns=list(fields.all))
    src = frame[frame["tab"].isin(SOURCE_SCENARIOS)]
    if period == "train":
        df = src[src["date"] < 20220503]
    elif period == "valid":
        df = src[(src["date"] >= 20220503) & (src["date"] < 20220506)]
    elif period == "test":
        df = src[src["date"] >= 20220506]
    else:
        raise ValueError(period)
    return _BatchDataset(df)


def evaluate_auc(model, dataset, device, batch_size: int = 10000) -> float:
    model.eval()
    metric = BinaryAUROC()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = {k: (v.to(device).int() if k != "is_click" else v.to(device))
                 for k, v in batch.items()}
            model(x)
            metric.update(x["logit"].float().cpu(), x["is_click"].float().cpu())
    return float(metric.compute())


def source_train(model, device, seed, lr, beta2, max_epochs) -> dict:
    train_set = _build_dataset("train")
    valid_set = _build_dataset("valid")
    test_set = _build_dataset("test")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, beta2),
    )
    generator = torch.Generator().manual_seed(seed)
    best_auc = 0.0
    best_state = None
    batch_size = 10000
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                generator=generator, num_workers=4, pin_memory=True):
            x = {k: (v.to(device).int() if k != "is_click" else v.to(device))
                 for k, v in batch.items()}
            model(x)
            loss = criterion(x["logit"], x["is_click"].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(x["is_click"])
            n += len(x["is_click"])
        val_auc = evaluate_auc(model, valid_set, device)
        print(f"[source][epoch {epoch}] train_loss={total_loss / n:.4f} val_auc={val_auc:.4f}")
        if val_auc > best_auc + 0.001:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())
        else:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_auc": best_auc, "test_auc": evaluate_auc(model, test_set, device)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dense", "moe"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--max-epochs", type=int, default=300)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.model == "dense":
        model = DCNv2()
    else:
        model = DCNv2MoE_LowRank(dim=360, K=4, r=45, routing="data")
    model.to(device)

    t0 = time.time()
    summary = source_train(model, device, args.seed, args.lr, args.beta2, args.max_epochs)
    ckpt = source_checkpoint_path(args.model, args.seed)
    torch.save(model.state_dict(), ckpt)
    summary.update({
        "model_type": args.model, "seed": args.seed, "device": args.device,
        "lr": args.lr, "beta2": args.beta2, "seconds": round(time.time() - t0, 1),
        "checkpoint": str(ckpt),
    })
    out = Path("cache/downstream_transfer/source") / f"{args.model}_seed{args.seed}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[source] {args.model} seed {args.seed} done -> {ckpt}")
    print(f"[source] test_auc={summary['test_auc']:.4f} best_val_auc={summary['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
