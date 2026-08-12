#!/usr/bin/env python3
"""Validate, summarize, and gate the two-task specialist-only screen."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))

from shared_residual_continual_protocol import (  # noqa: E402
    DEFAULT_ORDERS,
    FORMAL_SEEDS,
    SPECIALIST_LEARNING_RATES,
    order_name,
    sha256_file,
)

MATRIX_STATE = (
    REPOSITORY / "cache" / "manifests" / "shared_residual_continual" /
    "matrix_state.json"
)
RUN_ROOT = REPOSITORY / "cache" / "shared_residual_continual"


def learning_rate_slug(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def run_name(arm, learning_rate, order, seed):
    return (
        "shared-residual-specialist-screen-"
        f"{arm}-lr-{learning_rate_slug(learning_rate)}-"
        f"{order_name(order)}-seed-{seed}"
    )


RESULT_CSV = REPOSITORY / "result_shared_residual_specialist_screen.csv"
DECISION_PATH = RUN_ROOT / "specialist_screen_gate_decision.json"
RESULT_MANIFEST = RUN_ROOT / "specialist_screen_result_manifest.json"


def load_json(path):
    with open(path) as stream:
        return json.load(stream)


def expected_trials():
    expected = []
    for order in DEFAULT_ORDERS:
        expected.extend(
            [("task-head-only", 5e-4, order, seed) for seed in FORMAL_SEEDS]
        )
        for learning_rate in SPECIALIST_LEARNING_RATES:
            expected.extend(
                [("specialist-only", learning_rate, order, seed)
                 for seed in FORMAL_SEEDS]
            )
    return expected


def load_and_validate():
    summaries = {}
    errors = []
    for arm, learning_rate, order, seed in expected_trials():
        name = run_name(arm, learning_rate, order, seed)
        directory = RUN_ROOT / name
        manifest_path = directory / "manifest.json"
        summary_path = directory / "summary.json"
        result_path = directory / "trial_result.csv"
        checkpoint_path = directory / "model_state.pt"
        if not manifest_path.is_file():
            errors.append(f"missing manifest: {name}")
            continue
        manifest = load_json(manifest_path)
        if manifest.get("status") != "done":
            errors.append(f"trial {name} status={manifest.get('status')}")
            continue
        artifact_error = False
        for path, hash_field in (
            (summary_path, "summary_sha256"),
            (result_path, "trial_result_sha256"),
            (checkpoint_path, "checkpoint_sha256"),
        ):
            if not path.is_file() or manifest.get(hash_field) != sha256_file(path):
                errors.append(f"artifact/hash mismatch for {name}: {path.name}")
                artifact_error = True
        if artifact_error:
            continue
        summary = load_json(summary_path)
        config = summary.get("config", {})
        if (config.get("run_name") != name
                or config.get("arm") != arm
                or float(config.get("expert_learning_rate", -1)) != learning_rate
                or tuple(config.get("order", [])) != tuple(order)
                or config.get("seed") != seed):
            errors.append(f"identity mismatch for {name}")
            continue
        if summary.get("provenance") != manifest.get("provenance"):
            errors.append(f"provenance mismatch for {name}")
            continue
        summaries[(arm, learning_rate, tuple(order), seed)] = summary
    return summaries, errors


def fairness_errors(summaries, matrix):
    errors = []
    devices = {}
    dense_hashes = {}
    for (_, _, _, seed), summary in summaries.items():
        config = summary["config"]
        devices.setdefault(seed, set()).add(config["device"])
        dense_hashes.setdefault(seed, set()).add(
            summary["provenance"]["dense_checkpoint_sha256"]
        )
    if any(len(values) != 1 for values in devices.values()):
        errors.append("device is not constant within a paired seed")
    expected_devices = {42: {"cuda:0"}, 123: {"cuda:1"}, 456: {"cuda:0"}}
    if devices != expected_devices:
        errors.append("seed-device map differs from the preregistered mapping")
    if matrix.get("seed_device_map") != {
            "42": "cuda:0", "123": "cuda:1", "456": "cuda:0"}:
        errors.append("matrix seed-device map differs from the preregistered mapping")
    if any(len(values) != 1 for values in dense_hashes.values()):
        errors.append("dense checkpoint is not constant within a paired seed")
    source_maps = {
        json.dumps(summary["provenance"]["source_sha256"], sort_keys=True)
        for summary in summaries.values()
    }
    if len(source_maps) != 1:
        errors.append("source hashes differ across trials")
    frozen_hashes = matrix.get("frozen_source_sha256", {})
    for summary in summaries.values():
        for relative, digest in summary["provenance"]["source_sha256"].items():
            if frozen_hashes.get(relative) != digest:
                errors.append(f"trial source is not bound to matrix: {relative}")
        seed = summary["config"]["seed"]
        g1_checkpoint = (
            REPOSITORY / "cache" / "audit" / "shared_residual_continual" /
            "shared_residual_experiment_invariants.json"
        )
        invariant = load_json(g1_checkpoint)
        expected_dense = invariant["checks"]["function_preserving_upcycling"][
            str(seed)
        ]["dense_checkpoint_sha256"]
        if summary["provenance"]["dense_checkpoint_sha256"] != expected_dense:
            errors.append(f"trial dense checkpoint is not bound to G1 for seed {seed}")
    fixed_fields = {
        "batch_size", "epochs_per_task", "num_workers", "expert_count",
        "expert_rank", "adam_beta_two", "head_learning_rate", "fixed_step_budget",
        "early_stopping", "routing", "scenario_loss_weighting",
    }
    frozen_configs = {
        json.dumps({field: summary["config"][field] for field in fixed_fields},
                   sort_keys=True)
        for summary in summaries.values()
    }
    if len(frozen_configs) != 1:
        errors.append("frozen training configuration differs across trials")
    return errors


def paired_values(summaries, learning_rate, metric, order):
    return [
        summaries[("specialist-only", learning_rate, order, seed)]["results"][
            "metrics"][metric]
        - summaries[("task-head-only", 5e-4, order, seed)]["results"][
            "metrics"][metric]
        for seed in FORMAL_SEEDS
    ]


def decide(summaries):
    comparisons = {}
    passing = []
    for learning_rate in SPECIALIST_LEARNING_RATES:
        by_order = {}
        pass_orders = []
        for order in DEFAULT_ORDERS:
            bwt = paired_values(
                summaries, learning_rate, "backward_transfer", tuple(order),
            )
            learning_accuracy = paired_values(
                summaries, learning_rate, "learning_accuracy", tuple(order),
            )
            baseline = [
                summaries[("task-head-only", 5e-4, tuple(order), seed)][
                    "results"]["metrics"]["backward_transfer"]
                for seed in FORMAL_SEEDS
            ]
            noise_floor = max(baseline) - min(baseline)
            mean_bwt = sum(bwt) / len(bwt)
            order_pass = (
                all(value > 0 for value in bwt)
                and mean_bwt > noise_floor
                and not all(value < 0 for value in learning_accuracy)
            )
            pass_orders.append(order_pass)
            by_order[order_name(order)] = {
                "bwt_paired_differences": bwt,
                "bwt_directions": [
                    "positive" if value > 0 else "negative" if value < 0 else "zero"
                    for value in bwt
                ],
                "mean_bwt_difference": mean_bwt,
                "baseline_three_seed_range_noise_floor": noise_floor,
                "learning_accuracy_paired_differences": learning_accuracy,
                "order_pass": order_pass,
            }
        comparisons[str(learning_rate)] = by_order
        if all(pass_orders):
            overall = sum(
                item["mean_bwt_difference"] for item in by_order.values()
            ) / len(by_order)
            passing.append((overall, learning_rate))
    if passing:
        passing.sort(reverse=True)
        verdict = "PASS"
        winner = passing[0][1]
        reason = "one or more specialist learning rates pass both orders"
    else:
        all_differences = [
            value
            for comparison in comparisons.values()
            for details in comparison.values()
            for value in details["bwt_paired_differences"]
        ]
        verdict = "FAIL" if all(value <= 0 for value in all_differences) else "INCONCLUSIVE"
        winner = None
        reason = (
            "all specialist runs are non-positive against head-only"
            if verdict == "FAIL"
            else "paired BWT direction/practical-significance rules are not jointly met"
        )
    return {
        "status": "done",
        "gate": "Specialist-Only Continual Adaptation Screen",
        "verdict": verdict,
        "reason": reason,
        "winning_specialist_learning_rate": winner,
        "comparisons_against_task_head_only": comparisons,
        "unlock_shared_path_necessity_gate": verdict == "PASS",
        "claim_boundary": (
            "PASS only authorizes the Shared-Path Necessity Gate; it does not "
            "establish a full continual-learning or sparse-scaling claim"
        ),
    }


def write_csv(path, summaries):
    fields = [
        "run_name", "arm", "expert_learning_rate", "order", "seed", "device",
        "backward_transfer", "learning_accuracy", "average_forgetting",
        "worst_task_forgetting", "wall_clock_seconds",
    ]
    with open(path, "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm, learning_rate, order, seed in expected_trials():
            summary = summaries[(arm, learning_rate, tuple(order), seed)]
            metrics = summary["results"]["metrics"]
            writer.writerow({
                "run_name": summary["config"]["run_name"],
                "arm": arm,
                "expert_learning_rate": learning_rate,
                "order": ",".join(str(value) for value in order),
                "seed": seed,
                "device": summary["config"]["device"],
                "backward_transfer": metrics["backward_transfer"],
                "learning_accuracy": metrics["learning_accuracy"],
                "average_forgetting": metrics["average_forgetting"],
                "worst_task_forgetting": metrics["worst_task_forgetting"],
                "wall_clock_seconds": summary["results"]["wall_clock_seconds"],
            })


def write_results(summaries, decision):
    conflicts = [path for path in (RESULT_CSV, DECISION_PATH, RESULT_MANIFEST)
                 if path.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite aggregate results: {conflicts}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    csv_tmp = RESULT_CSV.with_suffix(".csv.tmp")
    decision_tmp = DECISION_PATH.with_suffix(".json.tmp")
    manifest_tmp = RESULT_MANIFEST.with_suffix(".json.tmp")
    try:
        write_csv(csv_tmp, summaries)
        with open(decision_tmp, "x") as stream:
            json.dump(decision, stream, indent=2, ensure_ascii=False)
        manifest = {
            "status": "done",
            "aggregate_csv": str(RESULT_CSV),
            "aggregate_csv_sha256": sha256_file(csv_tmp),
            "gate_decision": str(DECISION_PATH),
            "gate_decision_sha256": sha256_file(decision_tmp),
            "matrix_state": str(MATRIX_STATE),
            "matrix_state_sha256": sha256_file(MATRIX_STATE),
            "summarizer_sha256": sha256_file(Path(__file__).resolve()),
            "input_summary_sha256": {
                summary["config"]["run_name"]: sha256_file(
                    RUN_ROOT / summary["config"]["run_name"] / "summary.json"
                )
                for summary in summaries.values()
            },
        }
        with open(manifest_tmp, "x") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
        os.replace(csv_tmp, RESULT_CSV)
        os.replace(decision_tmp, DECISION_PATH)
        os.replace(manifest_tmp, RESULT_MANIFEST)
    except Exception:
        csv_tmp.unlink(missing_ok=True)
        decision_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-results", action="store_true")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    matrix = load_json(MATRIX_STATE) if MATRIX_STATE.exists() else None
    summaries, errors = load_and_validate()
    if matrix is None or matrix.get("status") not in {"done", "done-with-failures"}:
        errors.append("matrix state is absent or not terminal")
    if matrix:
        expected_names = {
            run_name(arm, learning_rate, order, seed)
            for arm, learning_rate, order, seed in expected_trials()
        }
        if set(matrix.get("trials", [])) != expected_names:
            errors.append("matrix trial set differs from the preregistered 24 trials")
        if set(matrix.get("completed_trials", [])) != expected_names:
            errors.append("matrix completed trial set is not exactly the 24 trials")
        current_self = sha256_file(Path(__file__).resolve())
        frozen_self = matrix.get("frozen_source_sha256", {}).get(
            "scripts/summarize_shared_residual_experiment.py"
        )
        if current_self != frozen_self:
            errors.append("summarizer source hash differs from matrix freeze")
    if matrix and matrix.get("failed_trials"):
        errors.append("matrix contains failed trials")
    errors.extend(fairness_errors(summaries, matrix) if summaries and matrix else [])
    if errors or len(summaries) != len(expected_trials()):
        payload = {
            "status": "blocked",
            "gate_verdict": "BLOCKED",
            "completed": len(summaries),
            "required": len(expected_trials()),
            "errors": errors,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1
    decision = decide(summaries)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    if arguments.write_results:
        write_results(summaries, decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
