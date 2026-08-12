"""Shared-residual MoE continual-learning protocol and machine-checkable metrics."""

import hashlib
import json
import os
import random
from pathlib import Path
from time import time_ns

import torch


REPORT_SCENARIOS = (0, 1, 2, 3, 4, 5, 6, 8)
DEFAULT_ORDERS = ((0, 3), (3, 0))
FORMAL_SEEDS = (42, 123, 456)
SPECIALIST_LEARNING_RATES = (2e-4, 5e-4, 1e-3)


def set_reproducible_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time_ns()}.tmp"
    )
    try:
        with open(temporary, "x") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_immutable_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time_ns()}.tmp"
    )
    try:
        with open(temporary, "x") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_order(value):
    order = tuple(int(item) for item in value.split(","))
    if len(order) < 2 or len(set(order)) != len(order):
        raise ValueError("order must contain at least two distinct scenario ids")
    unknown = set(order) - set(REPORT_SCENARIOS)
    if unknown:
        raise ValueError(f"order contains unsupported scenarios: {sorted(unknown)}")
    return order


def order_name(order):
    return "-then-".join(f"scenario-{scenario}" for scenario in order)


def owner_expert(task_id, expert_count):
    """Stable task ownership independent of presentation order."""
    if task_id not in REPORT_SCENARIOS:
        raise ValueError(f"unsupported task id {task_id}")
    return REPORT_SCENARIOS.index(task_id) % expert_count


def continual_metrics(order, trajectory):
    """Compute standard BWT and learning accuracy from a complete A matrix."""
    if len(trajectory) != len(order):
        raise ValueError("trajectory length does not match task order")
    matrix = []
    for row in trajectory:
        auc = row["auc_per_scenario"]
        matrix.append([float(auc[str(scenario)]) for scenario in order])
    diagonal = [matrix[index][index] for index in range(len(order))]
    final = matrix[-1]
    backward_differences = [
        final[index] - diagonal[index]
        for index in range(len(order) - 1)
    ]
    forgetting = [
        max(row[index] for row in matrix[index:]) - final[index]
        for index in range(len(order) - 1)
    ]
    return {
        "auc_matrix": matrix,
        "learning_accuracy": sum(diagonal) / len(diagonal),
        "backward_transfer": (
            sum(backward_differences) / len(backward_differences)
            if backward_differences else 0.0
        ),
        "average_forgetting": (
            sum(forgetting) / len(forgetting) if forgetting else 0.0
        ),
        "worst_task_forgetting": max(forgetting) if forgetting else 0.0,
        "backward_differences": backward_differences,
    }


def optimizer_groups(model, task_id, arm, expert_learning_rate,
                     head_learning_rate=5e-4,
                     shared_learning_rate_ratio=0.0, weight_decay=0.0):
    """Build disjoint AdamW groups; inactive task heads are never registered."""
    if arm not in {"task-head-only", "specialist-only", "slow-shared"}:
        raise ValueError(f"unsupported arm {arm!r}")
    groups = [{
        "name": "current-task-head",
        "params": model.task_head_parameters(task_id),
        "lr": head_learning_rate,
        "weight_decay": 0.0,
    }]
    expert_index = owner_expert(task_id, model.K)
    if arm in {"specialist-only", "slow-shared"}:
        groups.append({
            "name": f"task-owned-specialist-{expert_index}",
            "params": model.specialist_parameters(expert_index),
            "lr": expert_learning_rate,
            "weight_decay": weight_decay,
        })
    if arm == "slow-shared":
        shared_parameters = [
            parameter
            for layer in model.cross_layers
            for parameter in layer.shared.parameters()
        ]
        groups.append({
            "name": "always-on-shared-expert",
            "params": shared_parameters,
            "lr": expert_learning_rate * shared_learning_rate_ratio,
            "weight_decay": weight_decay,
        })
    return groups


def snapshot_blocks(model):
    snapshots = {}
    for block_name, parameters in model.parameter_blocks().items():
        snapshots[block_name] = [
            parameter.detach().cpu().clone() for parameter in parameters
        ]
    return snapshots


def block_drift(before, model):
    result = {}
    current_blocks = model.parameter_blocks()
    for block_name, previous in before.items():
        squared = 0.0
        maximum = 0.0
        changed = 0
        for old, new in zip(previous, current_blocks[block_name]):
            difference = new.detach().cpu() - old
            squared += float(difference.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
            changed += int(torch.count_nonzero(difference))
        result[block_name] = {
            "l2": squared ** 0.5,
            "max_abs": maximum,
            "changed_elements": changed,
        }
    return result
