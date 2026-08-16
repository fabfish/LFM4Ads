import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""Run one immutable specialist-only shared-residual continual-learning trial."""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import fields
from dataset import Dataset, Split
from model import DCNv2, DCNv2MoE_SharedResidual
from shared_residual_continual_protocol import (
    FORMAL_SEEDS,
    block_drift,
    continual_metrics,
    optimizer_groups,
    order_name,
    owner_expert,
    parse_order,
    set_reproducible_seed,
    sha256_file,
    snapshot_blocks,
    tensor_sha256,
    write_json_atomic,
    write_json_immutable_atomic,
)
from train import evaluate


REPOSITORY = Path(__file__).resolve().parent
RUN_ROOT = REPOSITORY / "cache" / "shared_residual_continual"
DRIVER_GLOB = "*-共享残差混合专家-函数保持与持续学习-驱动.md"
AUTHORIZATION_PATH = REPOSITORY / "scripts" / "shared_residual_experiment_authorization.json"
MATRIX_ENVIRONMENT = "SHARED_RESIDUAL_EXPERIMENT_MATRIX_EXECUTION"


def learning_rate_slug(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def expected_run_name(arm, learning_rate, order, seed):
    return (
        "shared-residual-specialist-screen-"
        f"{arm}-lr-{learning_rate_slug(learning_rate)}-"
        f"{order_name(order)}-seed-{seed}"
    )


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arm", required=True,
                        choices=("task-head-only", "specialist-only", "slow-shared"))
    parser.add_argument("--expert-learning-rate", required=True, type=float)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--shared-learning-rate-ratio", type=float, default=0.0)
    parser.add_argument("--order", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--expert-rank", type=int, default=45)
    parser.add_argument("--adam-beta-two", type=float, default=0.999)
    arguments = parser.parse_args(argv)
    arguments.order = parse_order(arguments.order)
    required = expected_run_name(
        arguments.arm, arguments.expert_learning_rate,
        arguments.order, arguments.seed,
    )
    if arguments.run_name != required:
        parser.error(f"--run-name must equal immutable canonical name {required!r}")
    if arguments.arm != "slow-shared" and arguments.shared_learning_rate_ratio != 0.0:
        parser.error("shared LR ratio is only valid for slow-shared")
    if arguments.arm == "slow-shared" and arguments.shared_learning_rate_ratio <= 0.0:
        parser.error("slow-shared requires a positive shared LR ratio")
    return arguments


def git_information():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
        capture_output=True, text=True, check=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY,
        capture_output=True, text=True, check=True,
    )
    return {"commit": completed.stdout.strip(), "status": status.stdout.strip()}


def verify_authorization():
    drivers = sorted((REPOSITORY / "docs").glob(DRIVER_GLOB))
    if not drivers or not AUTHORIZATION_PATH.is_file():
        raise PermissionError("driver or machine-readable authorization is missing")
    driver = drivers[-1]
    with open(AUTHORIZATION_PATH) as stream:
        authorization = json.load(stream)
    if authorization.get("status") != "authorized":
        raise PermissionError("experiment authorization is not active")
    if authorization.get("driver_path") != str(driver.relative_to(REPOSITORY)):
        raise PermissionError("authorization driver path mismatch")
    if authorization.get("driver_sha256") != sha256_file(driver):
        raise PermissionError("authorization is stale for the current driver")
    if os.environ.get(MATRIX_ENVIRONMENT) != "authorized":
        raise PermissionError("formal trial must be launched by the matrix runner")
    return driver, authorization


def move_batch(batch, device):
    for field_name in fields.all:
        batch[field_name] = batch[field_name].to(
            device, non_blocking=True,
        ).int()
    return batch


def train_task(model, scenario, arguments, task_index):
    train_set, _, _ = Split(scenario)
    model.requires_grad_(False)
    groups = optimizer_groups(
        model=model,
        task_id=scenario,
        arm=arguments.arm,
        expert_learning_rate=arguments.expert_learning_rate,
        head_learning_rate=arguments.head_learning_rate,
        shared_learning_rate_ratio=arguments.shared_learning_rate_ratio,
        weight_decay=0.0,
    )
    for group in groups:
        for parameter in group["params"]:
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, arguments.adam_beta_two),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, arguments.epochs_per_task + 1):
        generator = torch.Generator().manual_seed(
            arguments.seed * 1000 + task_index * 100 + epoch
        )
        loader = DataLoader(
            Dataset(train_set), batch_size=arguments.batch_size,
            num_workers=arguments.num_workers, pin_memory=True,
            shuffle=True, generator=generator,
        )
        loss_sum = 0.0
        optimizer_steps = 0
        model.train()
        for batch in loader:
            batch = move_batch(batch, arguments.device)
            optimizer.zero_grad(set_to_none=True)
            model(batch)
            loss = criterion(batch["logit"], batch["is_click"].float())
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            optimizer_steps += 1
        history.append({
            "epoch": epoch,
            "loss_sum": loss_sum,
            "optimizer_steps": optimizer_steps,
        })
    return history, [{
        "name": group["name"],
        "learning_rate": group["lr"],
        "weight_decay": group["weight_decay"],
        "parameter_count": sum(parameter.numel() for parameter in group["params"]),
    } for group in groups]


def evaluate_order(model, order, arguments):
    return {
        str(scenario): float(evaluate(
            model, Split(scenario)[2], batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
        ))
        for scenario in order
    }


def parameter_identity(model):
    return tensor_sha256(model.state_dict().items())


def write_trial_csv(path, summary):
    config = summary["config"]
    metrics = summary["results"]["metrics"]
    row = {
        "run_name": config["run_name"],
        "arm": config["arm"],
        "expert_learning_rate": config["expert_learning_rate"],
        "head_learning_rate": config["head_learning_rate"],
        "shared_learning_rate_ratio": config["shared_learning_rate_ratio"],
        "order": ",".join(str(value) for value in config["order"]),
        "seed": config["seed"],
        "device": config["device"],
        "backward_transfer": metrics["backward_transfer"],
        "learning_accuracy": metrics["learning_accuracy"],
        "average_forgetting": metrics["average_forgetting"],
        "worst_task_forgetting": metrics["worst_task_forgetting"],
    }
    with open(path, "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run(arguments):
    driver, authorization = verify_authorization()
    run_directory = RUN_ROOT / arguments.run_name
    if run_directory.exists():
        raise FileExistsError(f"refusing to overwrite run directory {run_directory}")
    run_directory.mkdir(parents=True)
    manifest_path = run_directory / "manifest.json"
    summary_path = run_directory / "summary.json"
    result_path = run_directory / "trial_result.csv"
    checkpoint_path = run_directory / "model_state.pt"
    dense_checkpoint = (
        REPOSITORY / "cache" /
        f"vanilla_from_scratch_seed{arguments.seed}.pt"
    )
    source_paths = [
        REPOSITORY / "model.py",
        REPOSITORY / "train.py",
        REPOSITORY / "dataset.py",
        REPOSITORY / "fields.py",
        REPOSITORY / "shared_residual_continual_protocol.py",
        Path(__file__).resolve(),
        driver,
        AUTHORIZATION_PATH,
    ]
    missing = [str(path) for path in source_paths + [dense_checkpoint]
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")
    provenance = {
        "git": git_information(),
        "driver_path": str(driver.relative_to(REPOSITORY)),
        "driver_sha256": sha256_file(driver),
        "authorization": authorization,
        "dense_checkpoint": str(dense_checkpoint),
        "dense_checkpoint_sha256": sha256_file(dense_checkpoint),
        "dataset_sha256": sha256_file(REPOSITORY / "dataset.feather"),
        "source_sha256": {
            str(path.relative_to(REPOSITORY)): sha256_file(path)
            for path in source_paths
        },
    }
    manifest = {
        "status": "running",
        "run_name": arguments.run_name,
        "created_at": datetime.datetime.now().isoformat(),
        "config": {
            "arm": arguments.arm,
            "expert_learning_rate": arguments.expert_learning_rate,
            "head_learning_rate": arguments.head_learning_rate,
            "shared_learning_rate_ratio": arguments.shared_learning_rate_ratio,
            "order": list(arguments.order),
            "seed": arguments.seed,
            "device": arguments.device,
            "batch_size": arguments.batch_size,
            "epochs_per_task": arguments.epochs_per_task,
            "num_workers": arguments.num_workers,
            "expert_count": arguments.expert_count,
            "expert_rank": arguments.expert_rank,
            "adam_beta_two": arguments.adam_beta_two,
            "fixed_step_budget": True,
            "early_stopping": False,
            "routing": "frozen-uniform-fully-routed",
            "scenario_loss_weighting": "single-task-per-trial-step",
        },
        "provenance": provenance,
        "paths": {
            "summary": str(summary_path),
            "trial_result": str(result_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    write_json_atomic(manifest_path, manifest)

    set_reproducible_seed(arguments.seed)
    dense = DCNv2().to(arguments.device)
    dense.load_state_dict(torch.load(dense_checkpoint, map_location=arguments.device))
    dense.eval()
    model = DCNv2MoE_SharedResidual(
        dim=360, K=arguments.expert_count, r=arguments.expert_rank,
        num_tasks=15, routing_mode="frozen-uniform-routing",
    ).to(arguments.device)
    model.load_pretrained(dense)
    del dense

    initial_identity = parameter_identity(model)
    pre_continual = evaluate_order(model, arguments.order, arguments)
    trajectory = []
    task_audits = []
    started = time.monotonic()
    for task_index, scenario in enumerate(arguments.order):
        before = snapshot_blocks(model)
        inactive_head_hashes = {
            str(task): tensor_sha256(
                (f"head.{task}.{name}", parameter)
                for name, parameter in model.head.rows[task].named_parameters()
            )
            for task in arguments.order if task != scenario
        }
        history, parameter_groups = train_task(
            model, scenario, arguments, task_index,
        )
        after_inactive_head_hashes = {
            str(task): tensor_sha256(
                (f"head.{task}.{name}", parameter)
                for name, parameter in model.head.rows[task].named_parameters()
            )
            for task in arguments.order if task != scenario
        }
        if inactive_head_hashes != after_inactive_head_hashes:
            raise RuntimeError("inactive task head changed during optimizer step")
        auc_per_scenario = evaluate_order(model, arguments.order, arguments)
        trajectory.append({
            "task_index": task_index,
            "train_scenario": scenario,
            "auc_per_scenario": auc_per_scenario,
        })
        task_audits.append({
            "task_index": task_index,
            "train_scenario": scenario,
            "owner_expert": owner_expert(scenario, arguments.expert_count),
            "parameter_groups": parameter_groups,
            "training_history": history,
            "block_drift": block_drift(before, model),
            "inactive_head_immutability": "pass",
        })

    metrics = continual_metrics(arguments.order, trajectory)
    torch.save(model.state_dict(), checkpoint_path)
    summary = {
        "config": {"run_name": arguments.run_name, **manifest["config"]},
        "results": {
            "pre_continual_auc": pre_continual,
            "trajectory": trajectory,
            "metrics": metrics,
            "task_update_audits": task_audits,
            "wall_clock_seconds": time.monotonic() - started,
            "initial_model_sha256": initial_identity,
            "final_model_sha256": parameter_identity(model),
        },
        "provenance": provenance,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
    }
    write_json_immutable_atomic(summary_path, summary)
    write_trial_csv(result_path, summary)

    current_sources = {
        str(path.relative_to(REPOSITORY)): sha256_file(path)
        for path in source_paths
    }
    if current_sources != provenance["source_sha256"]:
        raise RuntimeError("frozen source changed during trial")
    if sha256_file(REPOSITORY / "dataset.feather") != provenance["dataset_sha256"]:
        raise RuntimeError("dataset changed during trial")
    if sha256_file(dense_checkpoint) != provenance["dense_checkpoint_sha256"]:
        raise RuntimeError("dense checkpoint changed during trial")
    manifest.update({
        "status": "done",
        "completed_at": datetime.datetime.now().isoformat(),
        "summary_sha256": sha256_file(summary_path),
        "trial_result_sha256": sha256_file(result_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    })
    write_json_atomic(manifest_path, manifest)
    print(f"[done] {arguments.run_name}")


def main(argv=None):
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        run(arguments)
    except Exception as error:
        manifest_path = RUN_ROOT / arguments.run_name / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as stream:
                manifest = json.load(stream)
            manifest.update({
                "status": "blocked",
                "blocked_at": datetime.datetime.now().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            })
            write_json_atomic(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
