import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python
"""Run one Task-Conditioned Mixture Routing (TCMR) experiment.

TCMR is the explicit English abbreviation used by the driver document. Formal
run names remain fully descriptive and are validated before any artifact is
created.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

try:
    # Force the fork start method so DataLoader workers inherit the parent instead
    # of re-executing this script's __main__, which would recursively spawn the full
    # trial under the spawn start method and starve the GPUs (fork-bomb-like).
    mp.set_start_method("fork")
except RuntimeError:
    pass

import torch.nn.functional as F

import fields
from dataset import Dataset, Split
from model import (
    DCNv2MoE_TaskConditionedLowRank,
    TASK_CONDITIONED_ROUTING_MODES,
)

REPOSITORY = Path(__file__).resolve().parent
CACHE_ROOT = REPOSITORY / "cache" / "task_conditioned_mixture_routing"
DRIVER_GLOB = "*-Task-Conditioned-Mixture-Routing-驱动.md"
AUTHORIZATION_PATH = (
    REPOSITORY / "scripts" /
    "task_conditioned_mixture_routing_authorization.json"
)
MATRIX_EXECUTION_ENVIRONMENT_VARIABLE = (
    "TASK_CONDITIONED_MIXTURE_ROUTING_MATRIX_EXECUTION"
)
REPORT_SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
FORMAL_SEEDS = (42, 123, 456)
CONSISTENCY_WEIGHT = 0.01


def expected_run_name(routing_mode, seed):
    return f"task-conditioned-mixture-routing-{routing_mode}-seed-{seed}"


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Run one Task-Conditioned Mixture Routing experiment",
    )
    parser.add_argument("--routing-mode", required=True,
                        choices=TASK_CONDITIONED_ROUTING_MODES)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adam-beta-two", type=float, default=0.999)
    parser.add_argument("--maximum-epochs", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers; safe now that the fork start "
                             "method is forced to avoid re-exec fork bombs")
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--expert-rank", type=int, default=45)
    parser.add_argument("--probe-size", type=int, default=4096)
    arguments = parser.parse_args(argv)
    required_name = expected_run_name(arguments.routing_mode, arguments.seed)
    if arguments.run_name != required_name:
        parser.error(
            "--run-name must be the fully descriptive immutable name "
            f"{required_name!r}; simple stage codes are forbidden"
        )
    return arguments


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_shared_initialization(model):
    """Hash all non-router tensors to prove paired core initialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if ".router." in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def git_information():
    def run(*arguments):
        completed = subprocess.run(
            arguments, cwd=REPOSITORY, capture_output=True, text=True,
            timeout=30, check=False,
        )
        return completed.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status": run("git", "status", "--porcelain"),
    }


def verify_execution_authorization():
    drivers = sorted((REPOSITORY / "docs").glob(DRIVER_GLOB))
    if not drivers or not AUTHORIZATION_PATH.is_file():
        raise PermissionError("TCMR driver or machine authorization is missing")
    driver_path = drivers[-1]
    with open(AUTHORIZATION_PATH) as stream:
        authorization = json.load(stream)
    if authorization.get("status") != "authorized":
        raise PermissionError("TCMR formal execution is not authorized")
    if authorization.get("driver_sha256") != sha256_file(driver_path):
        raise PermissionError("TCMR authorization driver SHA-256 is stale")
    if authorization.get("authorized_scope") != (
            "fifteen-trial-multi-device-seed-group-parallel"):
        raise PermissionError("TCMR authorization scope does not permit this matrix")
    if os.environ.get(MATRIX_EXECUTION_ENVIRONMENT_VARIABLE) != "authorized":
        raise PermissionError("TCMR trial must be launched by the authorized matrix runner")


def routing_semantics(mode):
    return {
        "frozen-uniform-routing": (
            "deterministic uniform gate; both data and task router pathways frozen"
        ),
        "data-only-routing": "learnable gate from data representation only",
        "task-only-routing": "learnable gate from task identifier only",
        "data-and-task-routing": (
            "learnable gate from additive data logits and task-identifier logits"
        ),
        "data-and-task-consistency-routing": (
            "data-and-task gate with sample-weighted symmetric KL to data-only gate"
        ),
    }[mode]


class FixedRoutingProbe:
    """Stream routing diagnostics on a fixed validation prefix."""

    def __init__(self, data_frame, device, batch_size, num_workers, probe_size):
        self.data_frame = data_frame.iloc[:probe_size].copy()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.previous_assignments = None

    def __call__(self, model, epoch):
        diagnostics, assignments = collect_routing_diagnostics(
            model=model,
            data_frame=self.data_frame,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            previous_assignments=self.previous_assignments,
        )
        self.previous_assignments = assignments
        diagnostics["epoch"] = epoch
        diagnostics["probe_samples"] = len(self.data_frame)
        return diagnostics


def collect_routing_diagnostics(model, data_frame, device, batch_size,
                                num_workers, previous_assignments=None):
    layer_count = len(model.cross_layers)
    expert_count = model.K
    task_count = model.num_tasks
    gate_sums = [torch.zeros(expert_count, dtype=torch.float64)
                 for _ in range(layer_count)]
    entropy_sums = [0.0 for _ in range(layer_count)]
    joint_sums = [torch.zeros(task_count, expert_count, dtype=torch.float64)
                  for _ in range(layer_count)]
    assignments = [[] for _ in range(layer_count)]
    sample_count = 0

    loader = torch.utils.data.DataLoader(
        Dataset(data_frame), batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, shuffle=False,
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            for field_name in fields.all:
                batch[field_name] = batch[field_name].to(
                    device, non_blocking=True,
                ).int()
            model(batch)
            task_ids = batch["tab"].long()
            task_one_hot = F.one_hot(
                task_ids, num_classes=task_count,
            ).to(torch.float64)
            sample_count += len(task_ids)
            for layer_index, gate in enumerate(model.last_gates):
                gate_cpu = gate.detach().to(torch.float64).cpu()
                gate_sums[layer_index] += gate_cpu.sum(0)
                entropy_sums[layer_index] += float(
                    -(gate_cpu * gate_cpu.clamp_min(1e-12).log()).sum()
                )
                joint_sums[layer_index] += task_one_hot.cpu().T @ gate_cpu
                assignments[layer_index].append(gate_cpu.argmax(-1))

    if sample_count == 0:
        raise RuntimeError("routing diagnostic probe contains no samples")

    layer_results = []
    concatenated_assignments = []
    for layer_index in range(layer_count):
        mean_gate = gate_sums[layer_index] / sample_count
        mean_entropy = entropy_sums[layer_index] / sample_count
        joint = joint_sums[layer_index] / joint_sums[layer_index].sum()
        task_probability = joint.sum(1, keepdim=True)
        expert_probability = joint.sum(0, keepdim=True)
        independent = task_probability @ expert_probability
        nonzero = joint > 0
        mutual_information = float(
            (joint[nonzero] * (joint[nonzero] / independent[nonzero]).log()).sum()
        )
        current_assignment = torch.cat(assignments[layer_index])
        concatenated_assignments.append(current_assignment)
        route_churn = None
        if previous_assignments is not None:
            route_churn = float(
                (current_assignment != previous_assignments[layer_index])
                .to(torch.float32).mean()
            )
        layer_results.append({
            "layer": layer_index,
            "mean_gate": [float(value) for value in mean_gate],
            "mean_entropy": mean_entropy,
            "maximum_entropy": math.log(expert_count),
            "soft_task_expert_mutual_information_nats": mutual_information,
            "hard_argmax_route_churn": route_churn,
            "collapsed": (
                mean_entropy < 0.25 * math.log(expert_count)
                or float(mean_gate.max()) > 0.95
                or float(mean_gate.min()) < 0.01
            ),
        })
    return {"layers": layer_results}, concatenated_assignments


def evaluate_report(model, device, batch_size, num_workers, evaluate_function):
    per_scenario = {}
    for scenario in REPORT_SCENARIOS:
        per_scenario[str(scenario)] = float(evaluate_function(
            model, Split(scenario)[2], batch_size=batch_size,
            num_workers=num_workers,
        ))
    pooled_auc = float(evaluate_function(
        model, Split("all")[2], batch_size=batch_size,
        num_workers=num_workers,
    ))
    macro_auc = sum(per_scenario.values()) / len(per_scenario)
    test_set = Split("all")[2]
    routing_diagnostics, _ = collect_routing_diagnostics(
        model=model,
        data_frame=test_set,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return pooled_auc, macro_auc, per_scenario, routing_diagnostics


def summarize_probe_churn(history):
    per_layer = {}
    for epoch_record in history:
        probe = epoch_record.get("routing_probe", {})
        for layer in probe.get("layers", []):
            churn = layer.get("hard_argmax_route_churn")
            if churn is not None:
                per_layer.setdefault(str(layer["layer"]), []).append(float(churn))
    return {
        layer: {
            "mean_hard_argmax_route_churn": sum(values) / len(values),
            "maximum_hard_argmax_route_churn": max(values),
            "last_hard_argmax_route_churn": values[-1],
            "measured_epoch_transitions": len(values),
        }
        for layer, values in per_layer.items()
    }


def write_trial_csv(path, summary):
    config = summary["config"]
    row = {
        "run_name": config["run_name"],
        "routing_mode": config["routing_mode"],
        "seed": config["seed"],
        "pooled_auc": summary["results"]["pooled_auc"],
        "macro_auc_eight_report_scenarios": summary["results"]["macro_auc"],
        "scenario_loss_weighting": "sample",
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "device": config["device"],
        "git_commit": summary["provenance"]["git"]["commit"],
    }
    with open(path, "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run(arguments):
    verify_execution_authorization()
    from train import evaluate, train_task_conditioned_moe

    run_directory = CACHE_ROOT / arguments.run_name
    if run_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable run directory: {run_directory}"
        )

    drivers = sorted((REPOSITORY / "docs").glob(DRIVER_GLOB))
    if not drivers:
        raise FileNotFoundError("Task-Conditioned Mixture Routing driver is missing")
    driver_path = drivers[-1]
    dataset_path = REPOSITORY / "dataset.feather"
    source_paths = [
        REPOSITORY / "model.py",
        REPOSITORY / "train.py",
        REPOSITORY / "task_conditioned_mixture_routing_protocol.py",
        REPOSITORY / "dataset.py",
        REPOSITORY / "fields.py",
        Path(__file__).resolve(),
        AUTHORIZATION_PATH,
        driver_path,
    ]
    missing_sources = [str(path) for path in source_paths + [dataset_path]
                       if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing frozen inputs: {missing_sources}")

    train_set, validation_set, _ = Split("all")
    training_task_counts = {
        str(int(task_id)): int(count)
        for task_id, count in train_set["tab"].value_counts().sort_index().items()
    }
    if len(training_task_counts) != 15:
        raise RuntimeError(
            f"expected 15 training task identifiers, found {len(training_task_counts)}"
        )

    run_directory.mkdir(parents=True)
    manifest_path = run_directory / "manifest.json"
    summary_path = run_directory / "summary.json"
    result_path = run_directory / "trial_result.csv"
    checkpoint_path = run_directory / "model_state.pt"
    manifest = {
        "status": "running",
        "run_name": arguments.run_name,
        "created_at": datetime.datetime.now().isoformat(),
        "config": {
            "routing_mode": arguments.routing_mode,
            "router_semantics": routing_semantics(arguments.routing_mode),
            "seed": arguments.seed,
            "device": arguments.device,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "adam_beta_two": arguments.adam_beta_two,
            "maximum_epochs": arguments.maximum_epochs,
            "num_workers": arguments.num_workers,
            "expert_count": arguments.expert_count,
            "expert_rank": arguments.expert_rank,
            "probe_size": arguments.probe_size,
            "scenario_loss_weighting": "sample",
            "optimizer_steps_per_original_batch": 1,
            "consistency_weight": CONSISTENCY_WEIGHT,
            "training_task_count": len(training_task_counts),
            "training_task_sample_counts": training_task_counts,
            "report_scenarios": REPORT_SCENARIOS,
        },
        "provenance": {
            "git": git_information(),
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "source_sha256": {
                str(path.relative_to(REPOSITORY)): sha256_file(path)
                for path in source_paths
            },
        },
        "paths": {
            "summary": str(summary_path),
            "trial_result": str(result_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    with open(manifest_path, "x") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    model = DCNv2MoE_TaskConditionedLowRank(
        dim=360,
        K=arguments.expert_count,
        r=arguments.expert_rank,
        num_tasks=15,
        routing_mode=arguments.routing_mode,
    ).to(arguments.device)
    manifest["provenance"]["shared_initialization_sha256"] = (
        hash_shared_initialization(model)
    )
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)

    probe = FixedRoutingProbe(
        data_frame=validation_set,
        device=arguments.device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        probe_size=arguments.probe_size,
    )
    history = train_task_conditioned_moe(
        model=model,
        scenario="all",
        lr=arguments.learning_rate,
        beta2=arguments.adam_beta_two,
        batch_size=arguments.batch_size,
        shuffle=True,
        max_epochs=arguments.maximum_epochs,
        consistency_weight=CONSISTENCY_WEIGHT,
        num_workers=arguments.num_workers,
        epoch_probe=probe,
    )
    pooled_auc, macro_auc, per_scenario, routing_diagnostics = evaluate_report(
        model=model,
        device=arguments.device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        evaluate_function=evaluate,
    )
    torch.save(model.state_dict(), checkpoint_path)
    routing_churn_summary = summarize_probe_churn(history)

    summary = {
        "config": {"run_name": arguments.run_name, **manifest["config"]},
        "results": {
            "pooled_auc": pooled_auc,
            "macro_auc": macro_auc,
            "per_scenario_auc": per_scenario,
            "routing_diagnostics": routing_diagnostics,
            "fixed_validation_probe_hard_argmax_route_churn": routing_churn_summary,
            "training_history": history,
        },
        "provenance": manifest["provenance"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
    }
    with open(summary_path, "x") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    write_trial_csv(result_path, summary)

    completion_source_hashes = {
        str(path.relative_to(REPOSITORY)): sha256_file(path)
        for path in source_paths
    }
    completion_dataset_hash = sha256_file(dataset_path)
    if completion_source_hashes != manifest["provenance"]["source_sha256"]:
        raise RuntimeError("frozen source files changed during trial execution")
    if completion_dataset_hash != manifest["provenance"]["dataset_sha256"]:
        raise RuntimeError("dataset.feather changed during trial execution")

    manifest["status"] = "done"
    manifest["completed_at"] = datetime.datetime.now().isoformat()
    manifest["summary_sha256"] = sha256_file(summary_path)
    manifest["trial_result_sha256"] = sha256_file(result_path)
    manifest["checkpoint_sha256"] = sha256_file(checkpoint_path)
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print(f"[done] {arguments.run_name} -> {summary_path}")


def main(argv=None):
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        run(arguments)
    except Exception as error:
        run_directory = CACHE_ROOT / arguments.run_name
        manifest_path = run_directory / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as stream:
                manifest = json.load(stream)
            manifest["status"] = "blocked"
            manifest["blocked_at"] = datetime.datetime.now().isoformat()
            manifest["error"] = f"{type(error).__name__}: {error}"
            with open(manifest_path, "w") as stream:
                json.dump(manifest, stream, indent=2, ensure_ascii=False)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
