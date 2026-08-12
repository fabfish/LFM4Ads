#!/usr/bin/env python3
"""Preflight G1/G2 invariants for the shared-residual continual experiment."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torcheval.metrics import BinaryAUROC

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))

import fields  # noqa: E402
from dataset import Split  # noqa: E402
from model import (  # noqa: E402
    DCNv2,
    DCNv2MoE_SharedResidual,
    SharedResidualCrossExpertLayer,
)
from shared_residual_continual_protocol import (  # noqa: E402
    FORMAL_SEEDS,
    continual_metrics,
    optimizer_groups,
    sha256_file,
    tensor_sha256,
    write_json_immutable_atomic,
)


DRIVER_GLOB = "*-共享残差混合专家-函数保持与持续学习-驱动.md"
REPORT_PATH = (
    REPOSITORY / "cache" / "audit" / "shared_residual_continual" /
    "shared_residual_experiment_invariants.json"
)
LOGIT_TOLERANCE = 1e-6
LOSS_TOLERANCE = 1e-7
AUC_TOLERANCE = 1e-12


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def fixed_batch(size=1024):
    frame = Split("all")[1].iloc[:size]
    return {
        field_name: torch.as_tensor(frame[field_name].to_numpy()).int()
        for field_name in fields.all
    }


def clone_batch(batch):
    return {name: value.clone() for name, value in batch.items()}


def auc(logits, labels):
    metric = BinaryAUROC()
    metric.update(logits, labels.float())
    return float(metric.compute())


def verify_upcycling(errors):
    batch = fixed_batch()
    reports = {}
    criterion = nn.BCEWithLogitsLoss()
    for seed in FORMAL_SEEDS:
        checkpoint = REPOSITORY / "cache" / f"vanilla_from_scratch_seed{seed}.pt"
        if not checkpoint.is_file():
            errors.append(f"missing dense checkpoint {checkpoint}")
            continue
        dense = DCNv2()
        dense.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        moe = DCNv2MoE_SharedResidual(
            dim=360, K=4, r=45, num_tasks=15,
            routing_mode="frozen-uniform-routing",
        )
        moe.load_pretrained(dense)
        dense.eval()
        moe.eval()
        dense_batch = clone_batch(batch)
        moe_batch = clone_batch(batch)
        with torch.inference_mode():
            dense(dense_batch)
            moe(moe_batch)
        logit_error = float(
            (dense_batch["logit"] - moe_batch["logit"]).abs().max()
        )
        dense_loss = criterion(
            dense_batch["logit"], dense_batch["is_click"].float(),
        )
        moe_loss = criterion(
            moe_batch["logit"], moe_batch["is_click"].float(),
        )
        loss_error = abs(float(dense_loss - moe_loss))
        auc_error = abs(auc(dense_batch["logit"], dense_batch["is_click"])
                        - auc(moe_batch["logit"], moe_batch["is_click"]))
        shared_copy_exact = all(
            torch.equal(moe.cross_layers[index].shared.weight,
                        dense.layers[index].weight)
            and torch.equal(moe.cross_layers[index].shared.bias,
                            dense.layers[index].bias)
            for index in range(3)
        )
        specialist_max_abs = max(
            float(expert.up.weight.abs().max())
            + float(expert.up.bias.abs().max())
            for layer in moe.cross_layers for expert in layer.experts
        )
        uniform_gate = all(
            torch.equal(gate, torch.full_like(gate, 0.25))
            for gate in moe_batch["_gate"]
        )
        require(logit_error <= LOGIT_TOLERANCE,
                f"seed {seed} logit error {logit_error} exceeds tolerance", errors)
        require(loss_error <= LOSS_TOLERANCE,
                f"seed {seed} loss error {loss_error} exceeds tolerance", errors)
        require(auc_error <= AUC_TOLERANCE,
                f"seed {seed} AUC error {auc_error} exceeds tolerance", errors)
        require(shared_copy_exact, f"seed {seed} shared copy is not exact", errors)
        require(specialist_max_abs == 0.0,
                f"seed {seed} specialists are not exactly zero", errors)
        require(uniform_gate, f"seed {seed} router is not exactly uniform", errors)
        reports[str(seed)] = {
            "dense_checkpoint": str(checkpoint.relative_to(REPOSITORY)),
            "dense_checkpoint_sha256": sha256_file(checkpoint),
            "logit_max_abs_error": logit_error,
            "loss_abs_error": loss_error,
            "pooled_probe_auc_abs_error": auc_error,
            "shared_parameter_copy_exact": shared_copy_exact,
            "specialist_up_projection_max_abs": specialist_max_abs,
            "router_exactly_uniform": uniform_gate,
        }
    return reports


def run_adam(parameter, gradient, learning_rate, steps=200):
    optimizer = torch.optim.AdamW([parameter], lr=learning_rate, weight_decay=0.0)
    start = parameter.detach().clone()
    for _ in range(steps):
        parameter.grad = torch.full_like(parameter, gradient)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float((parameter.detach() - start).abs())


def verify_optimizer_semantics(errors):
    unscaled = run_adam(nn.Parameter(torch.tensor(1.0)), 1.0, 1e-3)
    pre_scaled = run_adam(nn.Parameter(torch.tensor(1.0)), 0.1, 1e-3)
    cancellation_ratio = pre_scaled / unscaled
    require(abs(cancellation_ratio - 1.0) <= 1e-5,
            "constant pre-Adam gradient scaling did not cancel", errors)

    full_lr = run_adam(nn.Parameter(torch.tensor(1.0)), 1.0, 1e-3, steps=1)
    tenth_lr = run_adam(nn.Parameter(torch.tensor(1.0)), 1.0, 1e-4, steps=1)
    post_ratio = full_lr / tenth_lr
    require(abs(post_ratio - 10.0) <= 5e-3,
            "parameter-group post-preconditioner LR ratio is incorrect", errors)
    return {
        "pre_adam_constant_scaling_update_ratio": cancellation_ratio,
        "parameter_group_learning_rate_update_ratio": post_ratio,
    }


def verify_update_isolation(errors):
    torch.manual_seed(9)
    layer = SharedResidualCrossExpertLayer(
        dim=8, K=4, r=2, num_tasks=3,
        routing_mode="frozen-uniform-routing",
    )
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    selected = list(layer.experts[1].parameters())
    for parameter in selected:
        parameter.requires_grad_(True)
    before = {
        index: tensor_sha256(layer.experts[index].state_dict().items())
        for index in range(4)
    }
    optimizer = torch.optim.AdamW(selected, lr=1e-3, weight_decay=0.0)
    x = torch.randn(16, 8)
    task = torch.tensor([0, 1] * 8)
    output, _, _ = layer(x, x, task)
    output.square().mean().backward()
    optimizer.step()
    after = {
        index: tensor_sha256(layer.experts[index].state_dict().items())
        for index in range(4)
    }
    require(before[1] != after[1], "selected expert did not update", errors)
    require(all(before[index] == after[index] for index in (0, 2, 3)),
            "non-selected expert changed", errors)

    model = DCNv2MoE_SharedResidual(dim=360, K=4, r=45, num_tasks=15)
    inactive_before = tensor_sha256(model.head.rows[0].state_dict().items())
    optimizer = torch.optim.AdamW(model.head.rows[3].parameters(), lr=1e-3)
    feature = torch.randn(8, 360)
    model.head(feature, torch.full((8,), 3)).sum().backward()
    optimizer.step()
    inactive_after = tensor_sha256(model.head.rows[0].state_dict().items())
    require(inactive_before == inactive_after,
            "inactive independent task head changed", errors)
    formal_groups = optimizer_groups(
        model, task_id=0, arm="specialist-only",
        expert_learning_rate=2e-4, head_learning_rate=5e-4,
    )
    group_lrs = {group["name"]: group["lr"] for group in formal_groups}
    require(group_lrs.get("current-task-head") == 5e-4,
            "formal task-head LR is not fixed at 5e-4", errors)
    require(group_lrs.get("task-owned-specialist-0") == 2e-4,
            "formal specialist LR does not use the scanned value", errors)
    return {
        "selected_expert_changed": before[1] != after[1],
        "non_selected_experts_immutable": all(
            before[index] == after[index] for index in (0, 2, 3)
        ),
        "inactive_task_head_immutable": inactive_before == inactive_after,
        "formal_optimizer_group_learning_rates": group_lrs,
    }


def verify_sample_weighted_gradient(errors):
    feature = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5])
    label = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    task = torch.tensor([0, 0, 1, 1, 1])
    criterion = nn.BCEWithLogitsLoss()
    full_weight = nn.Parameter(torch.tensor(0.25))
    criterion(feature * full_weight, label).backward()
    full_gradient = full_weight.grad.detach().clone()
    split_weight = nn.Parameter(torch.tensor(0.25))
    for task_id in task.unique():
        mask = task == task_id
        weight = mask.sum() / task.numel()
        (criterion(feature[mask] * split_weight, label[mask]) * weight).backward()
    split_gradient = split_weight.grad.detach().clone()
    error = float((full_gradient - split_gradient).abs())
    require(error <= 1e-7, "sample-weighted gradient conservation failed", errors)
    return {"gradient_abs_error": error}


def verify_bwt(errors):
    trajectory = [
        {"auc_per_scenario": {"0": 0.8, "3": 0.5}},
        {"auc_per_scenario": {"0": 0.7, "3": 0.9}},
    ]
    metrics = continual_metrics((0, 3), trajectory)
    require(abs(metrics["backward_transfer"] + 0.1) <= 1e-12,
            "standard BWT uses an incorrect baseline", errors)
    require(abs(metrics["learning_accuracy"] - 0.85) <= 1e-12,
            "learning accuracy calculation is incorrect", errors)
    require(abs(metrics["average_forgetting"] - 0.1) <= 1e-12,
            "average forgetting sign/definition is incorrect", errors)
    require(abs(metrics["worst_task_forgetting"] - 0.1) <= 1e-12,
            "worst-task forgetting sign/definition is incorrect", errors)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    drivers = sorted((REPOSITORY / "docs").glob(DRIVER_GLOB))
    errors = []
    if not drivers:
        errors.append("shared-residual driver document is missing")
        driver = None
    else:
        driver = drivers[-1]
    report = {
        "status": "pass",
        "gate_verdict": "PASS",
        "tolerances": {
            "logit_max_abs_error": LOGIT_TOLERANCE,
            "loss_abs_error": LOSS_TOLERANCE,
            "auc_abs_error": AUC_TOLERANCE,
        },
        "checks": {
            "function_preserving_upcycling": verify_upcycling(errors),
            "optimizer_semantics": verify_optimizer_semantics(errors),
            "update_isolation": verify_update_isolation(errors),
            "sample_weighted_gradient_conservation": verify_sample_weighted_gradient(errors),
            "standard_bwt_matrix": verify_bwt(errors),
        },
        "provenance": {
            "driver_path": str(driver.relative_to(REPOSITORY)) if driver else None,
            "driver_sha256": sha256_file(driver) if driver else None,
            "source_sha256": {
                str(path.relative_to(REPOSITORY)): sha256_file(path)
                for path in (
                    REPOSITORY / "model.py",
                    REPOSITORY / "shared_residual_continual_protocol.py",
                    Path(__file__).resolve(),
                )
            },
        },
        "errors": errors,
    }
    if errors:
        report["status"] = "fail"
        report["gate_verdict"] = "BLOCKED"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if arguments.write_report:
        write_json_immutable_atomic(REPORT_PATH, report)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
