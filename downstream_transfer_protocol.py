"""Frozen protocol primitives for held-out-domain downstream transfer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset

import fields
from dataset import DATASET_PATH
from model import DCNv2, DCNv2MoE_LowRank


SOURCE_SCENARIOS = (0, 1, 3, 4, 8)
TARGET_SCENARIOS = (2, 5, 6)
LABEL_BUDGET = 4096
FIT_SIZE = 3072
VALIDATION_SIZE = 1024
TRANSFER_ARMS = (
    "dense-head",
    "dense-adapter-r2",
    "moe-head",
    "moe-router",
)


@dataclass(frozen=True)
class TargetBudgetSplit:
    target: int
    seed: int
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    fit_sha256: str
    validation_sha256: str
    test_sha256: str


def _indices_sha256(indices: Iterable[int]) -> str:
    payload = "\n".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def target_budget_split(frame: pd.DataFrame, target: int, seed: int) -> TargetBudgetSplit:
    """Select and split the frozen 4096-label target budget by stable row hashes."""
    if target not in TARGET_SCENARIOS:
        raise ValueError(f"target={target} is not pre-registered")
    candidates = frame.index[
        (frame["tab"] == target) & (frame["date"] < 20220503)
    ].tolist()
    if len(candidates) < LABEL_BUDGET:
        raise ValueError(
            f"target={target} has {len(candidates)} train rows; need {LABEL_BUDGET}"
        )

    def key(index: int) -> bytes:
        return hashlib.sha256(f"{seed}:{int(index)}".encode("ascii")).digest()

    selected = sorted(candidates, key=key)[:LABEL_BUDGET]
    fit = tuple(selected[:FIT_SIZE])
    validation = tuple(selected[FIT_SIZE:])
    test = tuple(sorted(frame.index[
        (frame["tab"] == target) & (frame["date"] >= 20220506)
    ].tolist()))
    if len(fit) != FIT_SIZE or len(validation) != VALIDATION_SIZE:
        raise AssertionError("invalid target budget partition")
    if len(test) < 1:
        raise ValueError(f"target={target} has no test rows")
    return TargetBudgetSplit(
        target=target,
        seed=seed,
        fit_indices=fit,
        validation_indices=validation,
        test_indices=test,
        fit_sha256=_indices_sha256(fit),
        validation_sha256=_indices_sha256(validation),
        test_sha256=_indices_sha256(test),
    )


def source_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return source-only temporal train/validation/test frames."""
    source = frame[frame["tab"].isin(SOURCE_SCENARIOS)]
    return (
        source[source["date"] < 20220503],
        source[(source["date"] >= 20220503) & (source["date"] < 20220506)],
        source[source["date"] >= 20220506],
    )


def assert_source_target_disjoint(frame: pd.DataFrame) -> None:
    """Verify the pre-registered scenario sets and temporal rows are disjoint."""
    if set(SOURCE_SCENARIOS) & set(TARGET_SCENARIOS):
        raise AssertionError("source and target scenario sets overlap")
    source_index = set(frame.index[frame["tab"].isin(SOURCE_SCENARIOS)])
    target_index = set(frame.index[frame["tab"].isin(TARGET_SCENARIOS)])
    if source_index & target_index:
        raise AssertionError("source and target rows overlap")


class RankResidualAdapter(nn.Module):
    """Bias-free rank-r residual on the cross transform, zero at initialization."""

    def __init__(self, dim: int = 360, rank: int = 2):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(value))


class DenseDownstreamTransfer(nn.Module):
    """Frozen dense source backbone with a binary head and optional adapters."""

    def __init__(self, backbone: DCNv2, adapter_rank: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.binary_head = nn.Linear(backbone.layers[0].in_features, 1)
        self.adapters = (
            nn.ModuleList([
                RankResidualAdapter(backbone.layers[0].in_features, adapter_rank)
                for _ in range(3)
            ])
            if adapter_rank is not None else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> None:
        x0 = self.backbone.sparse(batch)
        x = x0
        for layer_index in range(3):
            transform = self.backbone.layers[layer_index](x)
            if self.adapters is not None:
                transform = transform + self.adapters[layer_index](x)
            x = transform * x0 + x
        x = self.backbone.layers[3](x.relu())
        x = self.backbone.layers[4](x.relu())
        batch["logit"] = self.binary_head(x).squeeze(-1)


class MoEDownstreamTransfer(nn.Module):
    """Frozen full-dimensional MoE source backbone with optional router tuning."""

    def __init__(self, backbone: DCNv2MoE_LowRank, train_router: bool):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        if train_router:
            for layer in self.backbone.cross_layers:
                layer.router.requires_grad_(True)
        self.binary_head = nn.Linear(backbone.dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> None:
        x0 = self.backbone.sparse(batch)
        x = x0
        gates = []
        for layer in self.backbone.cross_layers:
            x, gate = layer(x, x0)
            gates.append(gate)
        for layer in self.backbone.dnn:
            x = layer(x.relu())
        batch["logit"] = self.binary_head(x).squeeze(-1)
        batch["_gate"] = gates


def build_transfer_arm(
    arm: str,
    dense_backbone: DCNv2,
    moe_backbone: DCNv2MoE_LowRank,
) -> nn.Module:
    if arm == "dense-head":
        return DenseDownstreamTransfer(dense_backbone)
    if arm == "dense-adapter-r2":
        return DenseDownstreamTransfer(dense_backbone, adapter_rank=2)
    if arm == "moe-head":
        return MoEDownstreamTransfer(moe_backbone, train_router=False)
    if arm == "moe-router":
        return MoEDownstreamTransfer(moe_backbone, train_router=True)
    raise ValueError(f"unknown transfer arm={arm!r}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def parameter_sha256(parameters: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash named tensors without device-dependent serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def frozen_parameter_sha256(model: nn.Module) -> str:
    return parameter_sha256(
        (name, parameter) for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )


SOURCE_CHECKPOINT_DIR = Path("cache/downstream_transfer/source")


def source_checkpoint_path(model_type: str, seed: int) -> Path:
    SOURCE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if model_type not in ("dense", "moe"):
        raise ValueError(model_type)
    return SOURCE_CHECKPOINT_DIR / f"{model_type}_seed{seed}.pt"


def load_source_backbone(
    model_type: str,
    seed: int,
    device: torch.device,
    routing: str = "data",
) -> DCNv2 | DCNv2MoE_LowRank:
    """Load a frozen source-pretrained backbone for downstream transfer."""
    if model_type == "dense":
        backbone: DCNv2 | DCNv2MoE_LowRank = DCNv2()
    elif model_type == "moe":
        backbone = DCNv2MoE_LowRank(dim=360, K=4, r=45, routing=routing)
    else:
        raise ValueError(model_type)
    path = source_checkpoint_path(model_type, seed)
    if not path.exists():
        raise FileNotFoundError(f"source checkpoint missing: {path}")
    backbone.load_state_dict(torch.load(path, map_location=device))
    backbone.to(device)
    backbone.requires_grad_(False)
    return backbone


class DownstreamDataset(TorchDataset):
    """A torch Dataset over a pre-selected, frozen subset of the raw frame."""

    def __init__(self, rows: pd.DataFrame):
        self.batch = _rows_to_batch(rows)

    def __len__(self) -> int:
        return len(self.batch["is_click"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.batch.items()}


def _rows_to_batch(rows: pd.DataFrame) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for field in fields.user:
        batch[field] = torch.as_tensor(rows[field].to_numpy(), dtype=torch.long)
    for field in fields.video:
        batch[field] = torch.as_tensor(rows[field].to_numpy(), dtype=torch.long)
    batch["tab"] = torch.as_tensor(rows["tab"].to_numpy(), dtype=torch.long)
    batch["user_id"] = torch.as_tensor(rows["user_id"].to_numpy(), dtype=torch.long)
    batch["is_click"] = torch.as_tensor(rows["is_click"].to_numpy(), dtype=torch.long)
    return batch


def _select_rows(target: int, period: str, indices: Iterable[int]) -> pd.DataFrame:
    frame = pd.read_feather(DATASET_PATH, columns=list(fields.all))
    if period in ("fit", "validation"):
        frame = frame[(frame["tab"] == target) & (frame["date"] < 20220503)]
    elif period == "test":
        frame = frame[(frame["tab"] == target) & (frame["date"] >= 20220506)]
    else:
        raise ValueError(period)
    selected = frame.loc[list(indices)].reset_index(drop=True)
    if len(selected) != len(list(indices)):
        raise AssertionError("downstream row selection dropped rows")
    return selected


def build_downstream_datasets(target: int, seed: int):
    """Return (fit, validation, test) DownstreamDatasets and the frozen split."""
    frame = pd.read_feather(DATASET_PATH, columns=list(fields.all))
    split = target_budget_split(frame, target, seed)
    fit_rows = _select_rows(target, "fit", split.fit_indices)
    val_rows = _select_rows(target, "validation", split.validation_indices)
    test_rows = _select_rows(target, "test", split.test_indices)
    return (
        DownstreamDataset(fit_rows),
        DownstreamDataset(val_rows),
        DownstreamDataset(test_rows),
        split,
    )


def forward_smoke(device: str = "cpu") -> dict[str, dict]:
    """One forward+backward for every arm on tiny random inputs (no data leakage)."""
    torch.manual_seed(0)
    dense = DCNv2().to(device)
    moe = DCNv2MoE_LowRank(dim=360, K=4, r=45, routing="data").to(device)
    arms = {
        "dense-head": DenseDownstreamTransfer(dense),
        "dense-adapter-r2": DenseDownstreamTransfer(dense, adapter_rank=2),
        "moe-head": MoEDownstreamTransfer(moe, train_router=False),
        "moe-router": MoEDownstreamTransfer(moe, train_router=True),
    }
    # Use in-bounds indices (zero) for every embedding field; cardinalities vary
    # widely (some are < 10), so random ints would raise IndexError. A zero batch
    # still exercises the full forward/backward path and gradient flow.
    batch = {f: torch.zeros(5, dtype=torch.long, device=device) for f in fields.user}
    for f in fields.video:
        batch[f] = torch.zeros(5, dtype=torch.long, device=device)
    batch["tab"] = torch.zeros(5, dtype=torch.long, device=device)
    batch["user_id"] = torch.zeros(5, dtype=torch.long, device=device)
    batch["is_click"] = torch.zeros(5, dtype=torch.long, device=device)
    results: dict[str, dict] = {}
    for name, arm in arms.items():
        arm.to(device)
        arm.zero_grad(set_to_none=True)
        arm(batch)
        logit = batch["logit"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, batch["is_click"].float()
        )
        loss.backward()
        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for p in arm.parameters() if p.grad is not None
        )
        results[name] = {
            "logit_shape": list(logit.shape),
            "loss": float(loss.item()),
            "trainable_grad_norm": grad_norm,
            "has_nan": bool(torch.isnan(logit).any()),
        }
        for p in arm.parameters():
            p.grad = None
    return results
