#!/usr/bin/env python3
"""验证按参数角色隔离子任务优化器的十三项启动前不变量。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import DCNv2MoE  # noqa: E402
from task_role_optimizer import (  # noqa: E402
    ALWAYS_ON_SHARED_EXPERT,
    ROUTER,
    SPARSE_EXPERT,
    TASK_HEAD,
    ParameterRoleRegistry,
    SharedAdamWBatchOptimizer,
    TaskGradient,
    TaskRoleOptimizer,
    atomic_torch_save,
    capture_random_state,
    collect_task_gradients,
    default_role_hyperparameters,
    restore_random_state,
)
from task_role_optimizer_protocol import (  # noqa: E402
    EXPECTED_SAMPLE_COUNTS_27K,
    MACRO_SCENARIOS,
)


class TinySparse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tables = nn.ModuleDict({"feature": nn.Embedding(11, 10)})

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.tables["feature"](batch["feature"])


def build_model(same_expert: bool = False) -> DCNv2MoE:
    torch.manual_seed(17)
    model = DCNv2MoE(dim=10, K=2, routing="scenario", top_k=1)
    model.sparse = TinySparse()
    model = model.double()
    with torch.no_grad():
        for layer in model.cross_layers:
            layer.router.embed.weight.zero_()
            layer.router.embed.weight[0] = torch.tensor([2.0, -2.0], dtype=torch.float64)
            if same_expert:
                layer.router.embed.weight[1] = torch.tensor([2.0, -2.0], dtype=torch.float64)
            else:
                layer.router.embed.weight[1] = torch.tensor([-2.0, 2.0], dtype=torch.float64)
    return model


def make_batch(tasks: tuple[int, ...] = (0, 1)) -> dict[str, torch.Tensor]:
    feature, tab, label = [], [], []
    for task_id in tasks:
        for offset in range(4):
            feature.append((task_id * 4 + offset) % 11)
            tab.append(task_id)
            label.append((task_id + offset) % 2)
    return {
        "feature": torch.tensor(feature, dtype=torch.long),
        "tab": torch.tensor(tab, dtype=torch.long),
        "is_click": torch.tensor(label, dtype=torch.float64),
    }


def balanced_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model(batch)
    per_row = F.binary_cross_entropy_with_logits(
        batch["logit"], batch["is_click"], reduction="none")
    return torch.stack([
        per_row[batch["tab"] == task_id].mean()
        for task_id in torch.unique(batch["tab"], sorted=True)
    ]).mean()


def max_parameter_difference(left: nn.Module, right: nn.Module) -> float:
    return max(
        float((a - b).abs().max())
        for a, b in zip(left.parameters(), right.parameters())
    )


def tensors_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def result(passed: bool, detail: object) -> dict[str, object]:
    return {"passed": bool(passed), "detail": detail}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blank_task_gradient(
    registry: ParameterRoleRegistry,
    task_id: int,
) -> TaskGradient:
    return TaskGradient(
        task_id=task_id,
        gradients={spec.name: None for spec in registry.specs},
        active_experts={layer: frozenset() for layer in range(3)},
        active_rows={},
    )


def verify(output_path: Path) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    learning_rate = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.01

    native_model = build_model()
    wrapped_model = copy.deepcopy(native_model)
    native_optimizer = torch.optim.AdamW(
        native_model.parameters(), lr=learning_rate, betas=betas,
        eps=eps, weight_decay=weight_decay)
    wrapped_registry = ParameterRoleRegistry.from_model(wrapped_model)
    wrapped_optimizer = SharedAdamWBatchOptimizer(
        wrapped_registry, learning_rate, betas, eps, weight_decay)
    native_batch = make_batch()
    wrapped_batch = make_batch()
    balanced_loss(native_model, native_batch).backward()
    native_optimizer.step()
    wrapped_optimizer.step(collect_task_gradients(
        wrapped_model, wrapped_batch, wrapped_registry))
    shared_diff = max_parameter_difference(native_model, wrapped_model)
    checks["01_shared_state_matches_native_adamw"] = result(
        shared_diff < 1e-12, {"max_abs_parameter_difference": shared_diff})

    hyper = default_role_hyperparameters(
        expert_learning_rate=learning_rate,
        router_learning_rate_ratio=0.05,
        shared_learning_rate_ratio=1.0,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    order_left = build_model()
    order_right = copy.deepcopy(order_left)
    left_registry = ParameterRoleRegistry.from_model(order_left)
    right_registry = ParameterRoleRegistry.from_model(order_right)
    left_gradients = collect_task_gradients(order_left, make_batch(), left_registry)
    right_gradients = collect_task_gradients(order_right, make_batch(), right_registry)
    TaskRoleOptimizer(left_registry, hyper).step(left_gradients)
    TaskRoleOptimizer(right_registry, hyper).step(list(reversed(right_gradients)))
    order_diff = max_parameter_difference(order_left, order_right)
    checks["02_task_extraction_order_invariant"] = result(
        order_diff < 1e-12, {"max_abs_parameter_difference": order_diff})

    isolation_model = build_model()
    isolation_registry = ParameterRoleRegistry.from_model(isolation_model)
    isolation_optimizer = TaskRoleOptimizer(isolation_registry, hyper)
    initial_gradients = collect_task_gradients(
        isolation_model, make_batch(), isolation_registry)
    task_zero = next(item for item in initial_gradients if item.task_id == 0)
    task_one = next(item for item in initial_gradients if item.task_id == 1)
    isolation_optimizer.step([task_one])
    task_one_before = {
        spec.name: isolation_optimizer.task_state(spec.name, 1)
        for spec in isolation_registry.specs
        if isolation_optimizer.task_state(spec.name, 1) is not None
    }
    isolation_optimizer.step([task_zero])
    task_one_after = {
        spec.name: isolation_optimizer.task_state(spec.name, 1)
        for spec in isolation_registry.specs
        if isolation_optimizer.task_state(spec.name, 1) is not None
    }
    isolated = task_one_before.keys() == task_one_after.keys() and all(
        tensors_equal(task_one_before[name], task_one_after[name])
        for name in task_one_before
    )
    checks["03_other_task_state_unchanged"] = result(
        isolated, {"task_one_state_count": len(task_one_before)})

    sparse_model = build_model()
    sparse_registry = ParameterRoleRegistry.from_model(sparse_model)
    sparse_optimizer = TaskRoleOptimizer(sparse_registry, hyper)
    sparse_task = collect_task_gradients(
        sparse_model, make_batch((0,)), sparse_registry)[0]
    inactive_specs = [
        spec for spec in sparse_registry.specs
        if spec.role == SPARSE_EXPERT
        and spec.expert_index not in sparse_task.active_experts[spec.layer_index]
    ]
    inactive_before = {
        spec.name: spec.parameter.detach().clone() for spec in inactive_specs
    }
    sparse_optimizer.step([sparse_task])
    inactive_unchanged = all(
        torch.equal(spec.parameter, inactive_before[spec.name])
        and sparse_optimizer.task_state(spec.name, 0) is None
        for spec in inactive_specs
    )
    checks["04_unselected_expert_fully_static"] = result(
        inactive_unchanged,
        {"unselected_parameter_tensors": len(inactive_specs)})

    shared_expert_model = build_model(same_expert=True)
    shared_expert_registry = ParameterRoleRegistry.from_model(shared_expert_model)
    shared_expert_optimizer = TaskRoleOptimizer(shared_expert_registry, hyper)
    shared_expert_gradients = collect_task_gradients(
        shared_expert_model, make_batch(), shared_expert_registry)
    shared_expert_metrics = shared_expert_optimizer.step(shared_expert_gradients)
    selected_specs = [
        spec for spec in shared_expert_registry.specs
        if spec.role == SPARSE_EXPERT and spec.expert_index == 0
    ]
    two_states = all(
        shared_expert_optimizer.task_state(spec.name, 0) is not None
        and shared_expert_optimizer.task_state(spec.name, 1) is not None
        and shared_expert_metrics["parameter_writes"].get(spec.name) == 1
        for spec in selected_specs
    )
    checks["05_shared_expert_has_separate_states_one_write"] = result(
        two_states, {"checked_parameter_tensors": len(selected_specs)})

    row_model = build_model()
    row_registry = ParameterRoleRegistry.from_model(row_model)
    row_optimizer = TaskRoleOptimizer(row_registry, hyper)
    row_task = collect_task_gradients(row_model, make_batch((0,)), row_registry)[0]
    router_specs = [spec for spec in row_registry.specs if spec.role == ROUTER]
    router_other_before = {
        spec.name: spec.parameter[1:].detach().clone() for spec in router_specs
    }
    row_optimizer.step([row_task])
    router_rows_static = all(
        torch.equal(spec.parameter[1:], router_other_before[spec.name])
        for spec in router_specs
    )
    checks["06_router_only_current_row_changes"] = result(
        router_rows_static, {"router_tables": len(router_specs)})

    head_specs = [spec for spec in row_registry.specs if spec.role == TASK_HEAD]
    head_model = build_model()
    head_registry = ParameterRoleRegistry.from_model(head_model)
    head_optimizer = TaskRoleOptimizer(head_registry, hyper)
    head_specs = [spec for spec in head_registry.specs if spec.role == TASK_HEAD]
    head_other_before = {
        spec.name: spec.parameter[1:].detach().clone() for spec in head_specs
    }
    head_task = collect_task_gradients(head_model, make_batch((0,)), head_registry)[0]
    head_optimizer.step([head_task])
    head_rows_static = all(
        torch.equal(spec.parameter[1:], head_other_before[spec.name])
        for spec in head_specs
    )
    checks["07_head_only_current_row_changes"] = result(
        head_rows_static, {"head_parameter_tensors": len(head_specs)})

    one_task_model = build_model()
    two_task_model = copy.deepcopy(one_task_model)
    one_registry = ParameterRoleRegistry.from_model(one_task_model)
    two_registry = ParameterRoleRegistry.from_model(two_task_model)
    shared_name = next(
        spec.name for spec in one_registry.specs
        if spec.role == "shared_backbone")
    one_gradient = blank_task_gradient(one_registry, 0)
    one_gradient.gradients[shared_name] = torch.ones_like(
        one_registry.by_name[shared_name].parameter)
    two_gradient_zero = blank_task_gradient(two_registry, 0)
    two_gradient_one = blank_task_gradient(two_registry, 1)
    two_gradient_zero.gradients[shared_name] = torch.ones_like(
        two_registry.by_name[shared_name].parameter)
    two_gradient_one.gradients[shared_name] = torch.ones_like(
        two_registry.by_name[shared_name].parameter)
    TaskRoleOptimizer(one_registry, hyper).step([one_gradient])
    TaskRoleOptimizer(two_registry, hyper).step([
        two_gradient_zero, two_gradient_one])
    count_diff = float((
        one_registry.by_name[shared_name].parameter
        - two_registry.by_name[shared_name].parameter
    ).abs().max())
    checks["08_task_count_does_not_scale_update"] = result(
        count_diff < 1e-12, {"max_abs_parameter_difference": count_diff})

    decay_model = build_model()
    decay_registry = ParameterRoleRegistry.from_model(decay_model)
    decay_optimizer = TaskRoleOptimizer(decay_registry, hyper)
    active_spec = next(
        spec for spec in decay_registry.specs
        if spec.role == SPARSE_EXPERT
        and spec.layer_index == 0 and spec.expert_index == 0)
    inactive_spec = next(
        spec for spec in decay_registry.specs
        if spec.role == SPARSE_EXPERT
        and spec.layer_index == 0 and spec.expert_index == 1
        and spec.name.endswith("weight"))
    active_before = active_spec.parameter.detach().clone()
    inactive_before_decay = inactive_spec.parameter.detach().clone()
    decay_tasks = []
    for task_id in (0, 1):
        item = blank_task_gradient(decay_registry, task_id)
        item.gradients[active_spec.name] = torch.zeros_like(active_spec.parameter)
        item.active_experts = {0: frozenset({0}), 1: frozenset(), 2: frozenset()}
        decay_tasks.append(item)
    decay_optimizer.step(decay_tasks)
    expected_active = active_before * (
        1.0 - learning_rate * weight_decay)
    decay_once = torch.allclose(
        active_spec.parameter, expected_active, atol=0.0, rtol=1e-12)
    inactive_no_decay = torch.equal(inactive_spec.parameter, inactive_before_decay)
    checks["09_weight_decay_once_and_only_when_active"] = result(
        decay_once and inactive_no_decay,
        {"active_decayed_once": decay_once, "inactive_static": inactive_no_decay})

    checkpoint_model = build_model()
    checkpoint_registry = ParameterRoleRegistry.from_model(checkpoint_model)
    checkpoint_optimizer = TaskRoleOptimizer(checkpoint_registry, hyper)
    checkpoint_optimizer.step(collect_task_gradients(
        checkpoint_model, make_batch(), checkpoint_registry))
    checkpoint_path = output_path.with_suffix(".checkpoint.pt")
    checkpoint = {
        "model": checkpoint_model.state_dict(),
        "optimizer": checkpoint_optimizer.state_dict(),
        "random_state": capture_random_state(),
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    complete_checkpoint = (
        set(loaded) == {"model", "optimizer", "random_state"}
        and loaded["optimizer"]["registry"] == checkpoint_registry.metadata()
    )
    checks["10_checkpoint_is_complete"] = result(
        complete_checkpoint,
        {"checkpoint": str(checkpoint_path.relative_to(ROOT))})

    restored_model = build_model()
    restored_model.load_state_dict(loaded["model"])
    restored_registry = ParameterRoleRegistry.from_model(restored_model)
    restored_optimizer = TaskRoleOptimizer(restored_registry, hyper)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restore_random_state(loaded["random_state"])
    checkpoint_optimizer.step(collect_task_gradients(
        checkpoint_model, make_batch(), checkpoint_registry))
    restored_optimizer.step(collect_task_gradients(
        restored_model, make_batch(), restored_registry))
    resume_diff = max_parameter_difference(checkpoint_model, restored_model)
    checks["11_resume_matches_uninterrupted_next_step"] = result(
        resume_diff < 1e-12, {"max_abs_parameter_difference": resume_diff})
    checkpoint_path.unlink(missing_ok=True)

    role_model = build_model()
    role_registry = ParameterRoleRegistry.from_model(role_model)
    counts = role_registry.role_counts()
    complete_roles = (
        sum(counts.values()) == sum(
            parameter.numel() for parameter in role_model.parameters()
            if parameter.requires_grad)
        and counts[ALWAYS_ON_SHARED_EXPERT] == 0
        and all(counts[role] > 0 for role in (
            "shared_embedding", "shared_backbone", SPARSE_EXPERT,
            ROUTER, TASK_HEAD))
    )
    checks["12_parameter_roles_exclusive_complete"] = result(
        complete_roles, {"role_parameter_counts": counts})

    with (ROOT / "cache/sample_counts_27k.json").open(encoding="utf-8") as handle:
        actual_counts = {key: int(value) for key, value in json.load(handle).items()}
    data_frozen = (
        actual_counts == EXPECTED_SAMPLE_COUNTS_27K
        and MACRO_SCENARIOS == (0, 1, 2, 3, 4, 5, 6, 8)
    )
    checks["13_data_counts_and_endpoint_frozen"] = result(
        data_frozen,
        {"sample_counts": actual_counts,
         "macro_scenarios": list(MACRO_SCENARIOS)})

    passed = all(check["passed"] for check in checks.values())
    payload = {
        "status": "通过" if passed else "失败",
        "long_run_authorized": passed,
        "checks": checks,
        "implementation": {
            "site": os.environ.get("LFM_SITE", "B"),
            "sha256": {
                "task_role_optimizer.py": file_sha256(
                    ROOT / "task_role_optimizer.py"),
                "task_role_optimizer_protocol.py": file_sha256(
                    ROOT / "task_role_optimizer_protocol.py"),
                "scripts/verify/verify_task_role_optimizer.py": file_sha256(
                    ROOT / "scripts/verify/verify_task_role_optimizer.py"),
                "experiments/main_task_role_optimizer.py": file_sha256(
                    ROOT / "experiments/main_task_role_optimizer.py"),
                "scripts/matrix/run_task_role_optimizer_matrix.py": file_sha256(
                    ROOT / "scripts/matrix/run_task_role_optimizer_matrix.py"),
                "scripts/summarize/summarize_task_role_optimizer.py": file_sha256(
                    ROOT / "scripts/summarize/summarize_task_role_optimizer.py"),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="cache/task_role_optimizer_27k_siteB/invariant_audit.json")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    payload = verify(output)
    for name, check in payload["checks"].items():
        print(f"[{name}] {'通过' if check['passed'] else '失败'}: {check['detail']}")
    print(f"总判定：{payload['status']}")
    print(f"证据：{output}")
    if not payload["long_run_authorized"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
