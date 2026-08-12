#!/usr/bin/env python
"""Verify downstream-transfer protocol invariants without launching training."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from downstream_transfer_protocol import (  # noqa: E402
    FIT_SIZE,
    LABEL_BUDGET,
    SOURCE_SCENARIOS,
    TARGET_SCENARIOS,
    VALIDATION_SIZE,
    DenseDownstreamTransfer,
    MoEDownstreamTransfer,
    RankResidualAdapter,
    assert_source_target_disjoint,
    source_frames,
    target_budget_split,
    trainable_parameter_count,
)
from model import DCNv2, DCNv2MoE_LowRank  # noqa: E402


AUDIT_PATH = ROOT / "cache/audit/downstream_transfer/protocol_invariants.json"
DRIVER_PATH = ROOT / "docs/archive/drivers/20260812-2303-MoE下游留出域参数高效迁移驱动.md"
SEEDS = (42, 123, 456)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def verify_parameter_budget(errors: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    with torch.device("meta"):
        dense_head = DenseDownstreamTransfer(DCNv2())
        dense_adapter = DenseDownstreamTransfer(DCNv2(), adapter_rank=2)
        moe_head = MoEDownstreamTransfer(
            DCNv2MoE_LowRank(dim=360, K=4, r=45), train_router=False,
        )
        moe_router = MoEDownstreamTransfer(
            DCNv2MoE_LowRank(dim=360, K=4, r=45), train_router=True,
        )
    counts = {
        "dense-head": trainable_parameter_count(dense_head),
        "dense-adapter-r2": trainable_parameter_count(dense_adapter),
        "moe-head": trainable_parameter_count(moe_head),
        "moe-router": trainable_parameter_count(moe_router),
    }
    total_counts = {
        "dense-head": sum(parameter.numel() for parameter in dense_head.parameters()),
        "dense-adapter-r2": sum(
            parameter.numel() for parameter in dense_adapter.parameters()
        ),
        "moe-head": sum(parameter.numel() for parameter in moe_head.parameters()),
        "moe-router": sum(parameter.numel() for parameter in moe_router.parameters()),
    }
    require(counts["dense-head"] == 361, "dense-head count != 361", errors)
    require(counts["dense-adapter-r2"] == 4681,
            "dense-adapter-r2 count != 4681", errors)
    require(counts["moe-head"] == 361, "moe-head count != 361", errors)
    require(counts["moe-router"] == 4693,
            "moe-router count != 4693", errors)
    difference = abs(counts["moe-router"] - counts["dense-adapter-r2"])
    require(difference / counts["dense-adapter-r2"] < 0.01,
            "primary arms differ by >=1% trainable parameters", errors)
    return counts, total_counts


def verify_adapter_identity(errors: list[str]) -> float:
    torch.manual_seed(7)
    adapter = RankResidualAdapter(dim=8, rank=2)
    value = torch.randn(11, 8)
    maximum = float(adapter(value).abs().max())
    require(maximum == 0.0, "zero-init adapter is not exactly zero", errors)
    return maximum


def verify_paired_head_initialization(errors: list[str]) -> bool:
    class TinyDense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sparse = torch.nn.Identity()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(8, 8) for _ in range(5)
            ])

    class TinyCross(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = torch.nn.Linear(8, 2)

    class TinyMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 8
            self.sparse = torch.nn.Identity()
            self.cross_layers = torch.nn.ModuleList([TinyCross() for _ in range(3)])
            self.dnn = torch.nn.ModuleList([
                torch.nn.Linear(8, 8) for _ in range(2)
            ])

    dense_head_backbone = TinyDense()
    dense_adapter_backbone = TinyDense()
    moe_head_backbone = TinyMoE()
    seed = 991
    torch.manual_seed(seed)
    dense_head = DenseDownstreamTransfer(dense_head_backbone)
    torch.manual_seed(seed)
    dense_adapter = DenseDownstreamTransfer(dense_adapter_backbone, adapter_rank=2)
    torch.manual_seed(seed)
    moe_head = MoEDownstreamTransfer(moe_head_backbone, train_router=False)
    paired = (
        torch.equal(dense_head.binary_head.weight, dense_adapter.binary_head.weight)
        and torch.equal(dense_head.binary_head.bias, dense_adapter.binary_head.bias)
        and torch.equal(dense_head.binary_head.weight, moe_head.binary_head.weight)
        and torch.equal(dense_head.binary_head.bias, moe_head.binary_head.bias)
    )
    require(paired, "binary-head initialization is not paired across arms", errors)
    return paired


def verify_data(errors: list[str]) -> dict:
    frame = pd.read_feather(
        ROOT / "dataset.feather", columns=["date", "tab", "is_click"],
    )
    assert_source_target_disjoint(frame)
    train, validation, test = source_frames(frame)
    source_counts = {
        "train": int(len(train)),
        "validation": int(len(validation)),
        "test": int(len(test)),
    }
    splits = {}
    for target in TARGET_SCENARIOS:
        splits[str(target)] = {}
        for seed in SEEDS:
            split = target_budget_split(frame, target, seed)
            fit = set(split.fit_indices)
            validation_indices = set(split.validation_indices)
            require(len(fit) == FIT_SIZE, "fit size mismatch", errors)
            require(len(validation_indices) == VALIDATION_SIZE,
                    "validation size mismatch", errors)
            require(not fit & validation_indices, "fit/validation overlap", errors)
            require(len(fit | validation_indices) == LABEL_BUDGET,
                    "target label budget mismatch", errors)
            fit_frame = frame.loc[list(split.fit_indices)]
            validation_frame = frame.loc[list(split.validation_indices)]
            require(set(fit_frame["tab"]) == {target}, "fit target mismatch", errors)
            require(set(validation_frame["tab"]) == {target},
                    "validation target mismatch", errors)
            splits[str(target)][str(seed)] = {
                "fit_count": len(split.fit_indices),
                "validation_count": len(split.validation_indices),
                "fit_positive_count": int(fit_frame["is_click"].sum()),
                "validation_positive_count": int(validation_frame["is_click"].sum()),
                "fit_sha256": split.fit_sha256,
                "validation_sha256": split.validation_sha256,
            }
    return {
        "dataset_sha256": file_sha256(ROOT / "dataset.feather"),
        "source_scenarios": list(SOURCE_SCENARIOS),
        "target_scenarios": list(TARGET_SCENARIOS),
        "source_counts": source_counts,
        "target_budget_splits": splits,
    }


def main() -> None:
    errors: list[str] = []
    counts, total_counts = verify_parameter_budget(errors)
    adapter_maximum = verify_adapter_identity(errors)
    paired_head_initialization = verify_paired_head_initialization(errors)
    data = verify_data(errors)
    frozen_files = [
        ROOT / "model.py",
        ROOT / "dataset.py",
        ROOT / "fields.py",
        ROOT / "downstream_transfer_protocol.py",
        Path(__file__).resolve(),
        DRIVER_PATH,
    ]
    report = {
        "status": "pass" if not errors else "fail",
        "gate": "downstream-transfer-protocol-invariants",
        "authorization_status": "not-started",
        "formal_training_authorized": False,
        "errors": errors,
        "trainable_parameter_counts": counts,
        "total_parameter_counts": total_counts,
        "primary_arm_parameter_difference": (
            counts["moe-router"] - counts["dense-adapter-r2"]
        ),
        "zero_init_adapter_max_abs": adapter_maximum,
        "paired_binary_head_initialization": paired_head_initialization,
        "data": data,
        "frozen_source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in frozen_files
        },
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
