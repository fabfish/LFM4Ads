#!/usr/bin/env python
"""CPU smoke checks for Task-Conditioned Mixture Routing (TCMR)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPOSITORY = Path(__file__).resolve().parent.parent
DRIVER_GLOB = "*-Task-Conditioned-Mixture-Routing-驱动.md"
SOURCE_PATHS = (
    REPOSITORY / "model.py",
    REPOSITORY / "task_conditioned_mixture_routing_protocol.py",
    Path(__file__).resolve(),
)
sys.path.insert(0, str(REPOSITORY))

from model import (  # noqa: E402
    TASK_CONDITIONED_ROUTING_MODES,
    TaskConditionedLowRankCrossExpertLayer,
    TaskConditionedRouter,
)
from task_conditioned_mixture_routing_protocol import (  # noqa: E402
    task_conditioned_moe_batch_step,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def verify_router_modes(errors):
    torch.manual_seed(7)
    x_first = torch.randn(6, 8)
    x_second = torch.randn(6, 8)
    task_first = torch.tensor([0, 0, 1, 1, 2, 2])
    task_second = torch.tensor([2, 2, 1, 1, 0, 0])

    frozen = TaskConditionedRouter(8, 4, 3, "frozen-uniform-routing")
    frozen_first, frozen_auxiliary = frozen(x_first, task_first)
    frozen_second, _ = frozen(x_second, task_second)
    expected_uniform = torch.full_like(frozen_first, 0.25)
    require(torch.equal(frozen_first, frozen_second),
            "frozen uniform routing is not deterministic", errors)
    require(torch.allclose(frozen_first, expected_uniform),
            "frozen uniform routing is not exactly uniform", errors)
    require(float(frozen_auxiliary) == 0.0,
            "frozen uniform routing produced auxiliary loss", errors)
    require(not any(parameter.requires_grad for parameter in frozen.parameters()),
            "frozen uniform router has trainable parameters", errors)

    data_only = TaskConditionedRouter(8, 4, 3, "data-only-routing")
    with torch.no_grad():
        data_only.data_projection.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32,
        )
    data_first, _ = data_only(x_first, task_first)
    data_changed_task, _ = data_only(x_first, task_second)
    require(torch.equal(data_first, data_changed_task),
            "data-only routing depends on task identifiers", errors)
    require(not data_only.task_embedding.weight.requires_grad,
            "data-only routing left task pathway trainable", errors)

    task_only = TaskConditionedRouter(8, 4, 3, "task-only-routing")
    with torch.no_grad():
        task_only.task_embedding.weight.copy_(
            torch.tensor([
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ])
        )
    task_first_gate, _ = task_only(x_first, task_first)
    task_changed_data, _ = task_only(x_second, task_first)
    require(torch.equal(task_first_gate, task_changed_data),
            "task-only routing depends on data representations", errors)
    require(not any(parameter.requires_grad
                    for parameter in task_only.data_projection.parameters()),
            "task-only routing left data pathway trainable", errors)

    combined = TaskConditionedRouter(8, 4, 3, "data-and-task-routing")
    with torch.no_grad():
        combined.data_projection.weight.copy_(data_only.data_projection.weight)
        combined.task_embedding.weight.copy_(task_only.task_embedding.weight)
    combined_gate, _ = combined(x_first, task_first)
    combined_changed_data, _ = combined(x_second, task_first)
    combined_changed_task, _ = combined(x_first, task_second)
    require(not torch.allclose(combined_gate, combined_changed_data),
            "data-and-task routing ignores data representations", errors)
    require(not torch.allclose(combined_gate, combined_changed_task),
            "data-and-task routing ignores task identifiers", errors)

    consistency = TaskConditionedRouter(
        8, 4, 3, "data-and-task-consistency-routing",
    )
    with torch.no_grad():
        consistency.data_projection.weight.copy_(data_only.data_projection.weight)
        consistency.task_embedding.weight.copy_(task_only.task_embedding.weight)
    _, consistency_loss = consistency(x_first, task_first)
    consistency_loss.backward()
    require(float(consistency_loss) > 0.0,
            "symmetric KL consistency loss is not positive", errors)
    require(consistency.data_projection.weight.grad is not None,
            "consistency loss does not reach data pathway", errors)
    require(consistency.task_embedding.weight.grad is not None,
            "consistency loss does not reach task pathway", errors)


class CountingStochasticGradientDescent(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.0)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class TinyTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, batch):
        batch["logit"] = batch["feature"].float() * self.weight
        batch["_routing_auxiliary_loss"] = self.weight.square() * 0.0


def verify_single_step_and_sample_weighting(errors):
    model = TinyTaskModel()
    optimizer = CountingStochasticGradientDescent(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    batch = {
        "feature": torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5]),
        "is_click": torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]),
        "tab": torch.tensor([0, 0, 1, 1, 1]),
    }
    expected = float(criterion(
        batch["feature"] * model.weight.detach(), batch["is_click"],
    ))
    metrics = task_conditioned_moe_batch_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        criterion=criterion,
        consistency_weight=0.01,
    )
    require(optimizer.step_count == 1,
            "one original batch did not produce exactly one optimizer step", errors)
    require(metrics["optimizer_steps"] == 1,
            "batch-step metric did not report exactly one optimizer step", errors)
    require(abs(metrics["base_loss"] - expected) <= 1e-7,
            "sample-weighted task losses do not equal full-batch mean BCE", errors)


def verify_shared_initialization(errors):
    hashes = {}
    for mode in TASK_CONDITIONED_ROUTING_MODES:
        torch.manual_seed(123)
        layer = TaskConditionedLowRankCrossExpertLayer(
            dim=8, K=4, r=2, num_tasks=3, routing_mode=mode,
        )
        hashes[mode] = tensor_hash(
            (name, tensor) for name, tensor in layer.state_dict().items()
            if not name.startswith("router.")
        )
    require(len(set(hashes.values())) == 1,
            "expert initialization differs across routing modes", errors)
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    errors = []
    verify_router_modes(errors)
    verify_single_step_and_sample_weighting(errors)
    shared_hashes = verify_shared_initialization(errors)
    drivers = sorted((REPOSITORY / "docs").glob(DRIVER_GLOB))
    if not drivers:
        errors.append("Task-Conditioned Mixture Routing driver document is missing")
        driver_path = None
    else:
        driver_path = drivers[-1]
    report = {
        "status": "pass" if not errors else "fail",
        "experiment": "Task-Conditioned Mixture Routing (TCMR)",
        "routing_modes": list(TASK_CONDITIONED_ROUTING_MODES),
        "provenance": {
            "source_sha256": {
                str(path.relative_to(REPOSITORY)): sha256_file(path)
                for path in SOURCE_PATHS
            },
            "driver_path": (
                str(driver_path.relative_to(REPOSITORY)) if driver_path else None
            ),
            "driver_sha256": sha256_file(driver_path) if driver_path else None,
        },
        "checks": {
            "frozen_uniform_deterministic": "pass" if not errors else "see errors",
            "data_and_task_pathway_isolation": "pass" if not errors else "see errors",
            "symmetric_kl_gradient_flow": "pass" if not errors else "see errors",
            "sample_weighted_single_optimizer_step": "pass" if not errors else "see errors",
            "shared_expert_initialization": "pass" if len(set(shared_hashes.values())) == 1 else "fail",
        },
        "shared_expert_initialization_hashes": shared_hashes,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if arguments.write_report:
        output = (
            REPOSITORY / "cache" / "audit" /
            "task_conditioned_mixture_routing" /
            "task_conditioned_mixture_routing_invariants.json"
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite invariant report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "x") as stream:
            json.dump(report, stream, indent=2)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
