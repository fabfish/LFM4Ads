#!/usr/bin/env python
"""Summarize and gate Task-Conditioned Mixture Routing (TCMR) trials."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
RUN_ROOT = REPOSITORY / "cache" / "task_conditioned_mixture_routing"
MATRIX_STATE_PATH = (
    REPOSITORY / "cache" / "manifests" /
    "task_conditioned_mixture_routing" / "matrix_state.json"
)
ROUTING_MODES = (
    "frozen-uniform-routing",
    "data-only-routing",
    "task-only-routing",
    "data-and-task-routing",
    "data-and-task-consistency-routing",
)
SEEDS = (42, 123, 456)
PRIMARY_MODE = "data-and-task-routing"
PRIMARY_ANCHORS = ("frozen-uniform-routing", "data-only-routing")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(mode, seed):
    return f"task-conditioned-mixture-routing-{mode}-seed-{seed}"


def load_json(path):
    with open(path) as stream:
        return json.load(stream)


def validate_completed_trial(mode, seed, manifest_path, summary_path):
    name = run_name(mode, seed)
    run_directory = RUN_ROOT / name
    manifest = load_json(manifest_path)
    if manifest.get("status") != "done":
        raise ValueError(f"trial status is {manifest.get('status')!r}, not done")
    if manifest.get("run_name") != name:
        raise ValueError("manifest run_name does not match immutable directory")
    config = manifest.get("config", {})
    if config.get("routing_mode") != mode or config.get("seed") != seed:
        raise ValueError("manifest routing_mode or seed does not match matrix key")

    expected_paths = {
        "summary": summary_path,
        "trial_result": run_directory / "trial_result.csv",
        "checkpoint": run_directory / "model_state.pt",
    }
    for label, expected_path in expected_paths.items():
        recorded_value = manifest.get("paths", {}).get(label)
        if not recorded_value:
            raise ValueError(f"manifest {label} path is missing")
        recorded_path = Path(recorded_value)
        if recorded_path.resolve() != expected_path.resolve():
            raise ValueError(f"manifest {label} path does not match run directory")
        if not expected_path.is_file():
            raise ValueError(f"missing {label} artifact")

    expected_hashes = {
        "summary_sha256": sha256_file(expected_paths["summary"]),
        "trial_result_sha256": sha256_file(expected_paths["trial_result"]),
        "checkpoint_sha256": sha256_file(expected_paths["checkpoint"]),
    }
    for field, actual_hash in expected_hashes.items():
        if manifest.get(field) != actual_hash:
            raise ValueError(f"{field} mismatch")

    summary = load_json(summary_path)
    summary_config = summary.get("config", {})
    if (summary_config.get("run_name") != name
            or summary_config.get("routing_mode") != mode
            or summary_config.get("seed") != seed):
        raise ValueError("summary identity does not match matrix key")
    if summary.get("provenance") != manifest.get("provenance"):
        raise ValueError("summary provenance does not match manifest provenance")
    summary_checkpoint = summary.get("checkpoint", {})
    if Path(summary_checkpoint.get("path", "")).resolve() != expected_paths[
            "checkpoint"].resolve():
        raise ValueError("summary checkpoint path mismatch")
    if summary_checkpoint.get("sha256") != expected_hashes[
            "checkpoint_sha256"]:
        raise ValueError("summary checkpoint hash mismatch")

    with open(expected_paths["trial_result"], newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("trial result CSV must contain exactly one data row")
    row = rows[0]
    if (row.get("run_name") != name
            or row.get("routing_mode") != mode
            or int(row.get("seed", -1)) != seed):
        raise ValueError("trial result CSV identity mismatch")
    if abs(float(row["pooled_auc"]) - summary["results"]["pooled_auc"]) > 1e-12:
        raise ValueError("trial result CSV pooled AUC mismatch")
    if abs(float(row["macro_auc_eight_report_scenarios"])
           - summary["results"]["macro_auc"]) > 1e-12:
        raise ValueError("trial result CSV macro AUC mismatch")
    return manifest, summary


def load_summaries():
    summaries = {}
    manifests = {}
    missing = []
    active = []
    blocked = []
    corrupt = []
    for mode in ROUTING_MODES:
        for seed in SEEDS:
            name = run_name(mode, seed)
            summary_path = RUN_ROOT / name / "summary.json"
            manifest_path = RUN_ROOT / name / "manifest.json"
            if not manifest_path.exists():
                missing.append(name)
                continue
            try:
                manifest = load_json(manifest_path)
            except Exception as error:
                corrupt.append({"run_name": name, "error": str(error)})
                continue
            status = manifest.get("status")
            if status == "blocked":
                blocked.append({"run_name": name, "error": manifest.get("error")})
                continue
            if status != "done":
                active.append({"run_name": name, "status": status})
                continue
            if not summary_path.exists():
                corrupt.append({"run_name": name, "error": "done trial missing summary"})
                continue
            try:
                manifest, summary = validate_completed_trial(
                    mode, seed, manifest_path, summary_path,
                )
            except Exception as error:
                corrupt.append({"run_name": name, "error": str(error)})
                continue
            manifests[(mode, seed)] = manifest
            summaries[(mode, seed)] = summary
    return summaries, manifests, {
        "missing": missing,
        "active": active,
        "blocked": blocked,
        "corrupt": corrupt,
    }


def paired_differences(summaries, metric, mode, anchor):
    return [
        summaries[(mode, seed)]["results"][metric]
        - summaries[(anchor, seed)]["results"][metric]
        for seed in SEEDS
    ]


def directions(values):
    return ["positive" if value > 0 else "negative" if value < 0 else "zero"
            for value in values]


def routing_collapsed(summary):
    layers = summary["results"]["routing_diagnostics"]["layers"]
    return any(layer["collapsed"] for layer in layers)


def systematic_per_scenario_degradation(summaries, anchor):
    degraded = []
    scenario_names = summaries[(PRIMARY_MODE, SEEDS[0])]["results"][
        "per_scenario_auc"
    ]
    for scenario in scenario_names:
        deltas = [
            summaries[(PRIMARY_MODE, seed)]["results"]["per_scenario_auc"][scenario]
            - summaries[(anchor, seed)]["results"]["per_scenario_auc"][scenario]
            for seed in SEEDS
        ]
        if all(delta < 0 for delta in deltas) and sum(deltas) / len(deltas) <= -0.001:
            degraded.append({"scenario": scenario, "paired_differences": deltas})
    return degraded


def practical_significance(summaries, pooled_differences,
                           exploratory_gate_verdict):
    details = {}
    for anchor in PRIMARY_ANCHORS:
        anchor_values = [
            summaries[(anchor, seed)]["results"]["pooled_auc"]
            for seed in SEEDS
        ]
        noise_floor = max(anchor_values) - min(anchor_values)
        mean_difference = sum(pooled_differences[anchor]) / len(SEEDS)
        details[anchor] = {
            "anchor_seed_range_noise_floor": noise_floor,
            "mean_paired_difference": mean_difference,
            "exceeds_anchor_seed_range": mean_difference > noise_floor,
        }
    return {
        "rule": (
            "a stable improvement claim additionally requires the mean paired "
            "difference to exceed each anchor's three-seed range"
        ),
        "comparisons": details,
        "stable_improvement_claim": (
            "SUPPORTED"
            if (exploratory_gate_verdict == "PASS"
                and all(item["exceeds_anchor_seed_range"]
                        for item in details.values()))
            else "NOT_SUPPORTED"
        ),
    }


def decide_gate(summaries):
    pooled = {
        anchor: paired_differences(
            summaries, "pooled_auc", PRIMARY_MODE, anchor,
        )
        for anchor in PRIMARY_ANCHORS
    }
    macro = {
        anchor: paired_differences(
            summaries, "macro_auc", PRIMARY_MODE, anchor,
        )
        for anchor in PRIMARY_ANCHORS
    }
    collapsed_runs = [
        run_name(mode, seed)
        for mode in ROUTING_MODES
        for seed in SEEDS
        if routing_collapsed(summaries[(mode, seed)])
        and mode != "frozen-uniform-routing"
    ]
    degraded_scenarios = {
        anchor: systematic_per_scenario_degradation(summaries, anchor)
        for anchor in PRIMARY_ANCHORS
    }
    comparison_directions = {
        anchor: directions(values) for anchor, values in pooled.items()
    }

    if any("positive" in value and "negative" in value
           for value in comparison_directions.values()):
        verdict = "INCONCLUSIVE"
        reason = "a primary same-seed comparison changes sign across seeds"
    elif any(all(value <= 0 for value in values) for values in pooled.values()):
        verdict = "FAIL"
        reason = "the primary candidate is non-positive for every seed against an anchor"
    elif not all(all(value > 0 for value in values) for values in pooled.values()):
        verdict = "INCONCLUSIVE"
        reason = "the primary paired differences are not strictly positive for all seeds"
    elif collapsed_runs:
        verdict = "FAIL"
        reason = "a trainable routing variant violates the preregistered collapse safeguard"
    elif any(all(value < 0 for value in values) for values in macro.values()):
        verdict = "FAIL"
        reason = "macro AUC degrades for every seed against an anchor"
    elif any(degraded_scenarios.values()):
        verdict = "FAIL"
        reason = "at least one report scenario degrades consistently beyond 0.001"
    else:
        verdict = "PASS"
        reason = (
            "exploratory routing gate: both primary comparisons are positive "
            "for all seeds and safeguards pass"
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "primary_mode": PRIMARY_MODE,
        "primary_paired_pooled_auc_differences": pooled,
        "primary_paired_macro_auc_differences": macro,
        "directions": comparison_directions,
        "practical_significance": practical_significance(
            summaries, pooled, verdict,
        ),
        "collapsed_runs": collapsed_runs,
        "systematically_degraded_scenarios": degraded_scenarios,
        "secondary_modes_cannot_replace_primary_endpoint": [
            "task-only-routing",
            "data-and-task-consistency-routing",
        ],
    }


def validate_cross_trial_fairness(summaries, manifests):
    errors = []
    source_hash_sets = {
        json.dumps(summary["provenance"]["source_sha256"], sort_keys=True)
        for summary in summaries.values()
    }
    dataset_hashes = {
        summary["provenance"]["dataset_sha256"]
        for summary in summaries.values()
    }
    git_commits = {
        summary["provenance"]["git"]["commit"]
        for summary in summaries.values()
    }
    if len(source_hash_sets) != 1:
        errors.append("source SHA-256 maps differ across trials")
    if len(dataset_hashes) != 1:
        errors.append("dataset SHA-256 differs across trials")
    if len(git_commits) != 1:
        errors.append("git commit differs across trials")

    # `device` varies by design: the multi-device matrix runner pins a whole
    # seed's five routing modes to one device (per-seed device constancy is the
    # fairness guarantee), so device is an execution location, not a training
    # hyperparameter, and must be excluded from the frozen-config equality check.
    ignored_config_fields = {"routing_mode", "router_semantics", "seed", "device"}
    base_configs = {
        json.dumps({key: value for key, value in manifest["config"].items()
                    if key not in ignored_config_fields}, sort_keys=True)
        for manifest in manifests.values()
    }
    if len(base_configs) != 1:
        errors.append("frozen training configuration differs across trials")

    # Audit: within each seed the device must be constant (the dispatch rule).
    seed_devices = {}
    for manifest in manifests.values():
        cfg = manifest["config"]
        seed_devices.setdefault(str(cfg["seed"]), set()).add(cfg["device"])
    inconsistent = [s for s, devs in seed_devices.items() if len(devs) != 1]
    if inconsistent:
        errors.append(
            f"device is not constant within seed(s) {inconsistent} "
            f"(per-seed device pinning violated)")

    shared_hashes = {}
    for (mode, seed), summary in summaries.items():
        shared_hashes.setdefault(seed, set()).add(
            summary["provenance"]["shared_initialization_sha256"]
        )
    mismatched_seeds = [seed for seed, values in shared_hashes.items()
                        if len(values) != 1]
    if mismatched_seeds:
        errors.append(f"shared initialization hash mismatch for seeds {mismatched_seeds}")
    return errors


def write_aggregate_csv(path, summaries):
    fieldnames = [
        "run_name", "routing_mode", "seed", "pooled_auc", "macro_auc",
        "mean_layer_entropy", "mean_soft_task_expert_mutual_information_nats",
        "mean_last_hard_argmax_route_churn",
    ]
    with open(path, "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for mode in ROUTING_MODES:
            for seed in SEEDS:
                summary = summaries[(mode, seed)]
                layers = summary["results"]["routing_diagnostics"]["layers"]
                churn = summary["results"].get(
                    "fixed_validation_probe_hard_argmax_route_churn", {},
                )
                last_churn_values = [
                    values["last_hard_argmax_route_churn"]
                    for values in churn.values()
                ]
                writer.writerow({
                    "run_name": run_name(mode, seed),
                    "routing_mode": mode,
                    "seed": seed,
                    "pooled_auc": summary["results"]["pooled_auc"],
                    "macro_auc": summary["results"]["macro_auc"],
                    "mean_layer_entropy": sum(
                        layer["mean_entropy"] for layer in layers
                    ) / len(layers),
                    "mean_soft_task_expert_mutual_information_nats": sum(
                        layer["soft_task_expert_mutual_information_nats"]
                        for layer in layers
                    ) / len(layers),
                    "mean_last_hard_argmax_route_churn": (
                        sum(last_churn_values) / len(last_churn_values)
                        if last_churn_values else None
                    ),
                })


def write_results_transactionally(summaries, decision):
    csv_path = REPOSITORY / "results" / "tcmr" / "result_task_conditioned_mixture_routing.csv"
    decision_path = RUN_ROOT / "gate_decision.json"
    result_manifest_path = RUN_ROOT / "aggregate_result_manifest.json"
    if any(path.exists() for path in (
            csv_path, decision_path, result_manifest_path)):
        raise FileExistsError(
            "refusing to overwrite aggregate CSV, gate decision, or result manifest"
        )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    decision_temporary = decision_path.with_suffix(".json.tmp")
    manifest_temporary = result_manifest_path.with_suffix(".json.tmp")
    try:
        write_aggregate_csv(csv_temporary, summaries)
        with open(decision_temporary, "x") as stream:
            json.dump(decision, stream, indent=2)
        result_manifest = {
            "status": "done",
            "aggregate_csv": str(csv_path),
            "aggregate_csv_sha256": sha256_file(csv_temporary),
            "gate_decision": str(decision_path),
            "gate_decision_sha256": sha256_file(decision_temporary),
            "matrix_state": str(MATRIX_STATE_PATH),
            "matrix_state_sha256": sha256_file(MATRIX_STATE_PATH),
        }
        with open(manifest_temporary, "x") as stream:
            json.dump(result_manifest, stream, indent=2)
        os.replace(csv_temporary, csv_path)
        os.replace(decision_temporary, decision_path)
        os.replace(manifest_temporary, result_manifest_path)
    except Exception:
        csv_temporary.unlink(missing_ok=True)
        decision_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-results", action="store_true")
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)

    matrix_state = load_json(MATRIX_STATE_PATH) if MATRIX_STATE_PATH.exists() else None
    summaries, manifests, states = load_summaries()
    if states["blocked"] or states["corrupt"]:
        print(json.dumps({
            "status": "blocked",
            "trial_states": states,
            "matrix_state": matrix_state,
            "gate_verdict": "INCONCLUSIVE",
        }, indent=2))
        return 1
    if states["missing"] or states["active"]:
        status = "not-started" if not summaries and not states["active"] else "running"
        if matrix_state and matrix_state.get("status") == "blocked":
            status = "blocked"
        print(json.dumps({
            "status": status,
            "completed": len(summaries),
            "required": len(ROUTING_MODES) * len(SEEDS),
            "trial_states": states,
            "matrix_state": matrix_state,
            "gate_verdict": "INCONCLUSIVE",
        }, indent=2))
        return 1 if status == "blocked" else 2
    if not matrix_state or matrix_state.get("status") != "done":
        print(json.dumps({
            "status": "blocked",
            "reason": "fifteen trials exist without a done matrix state",
            "matrix_state": matrix_state,
            "gate_verdict": "INCONCLUSIVE",
        }, indent=2))
        return 1

    expected_trials = [run_name(mode, seed)
                       for mode in ROUTING_MODES for seed in SEEDS]
    matrix_errors = []
    # Order-insensitive: under multi-device parallel execution the completion
    # order diverges from the canonical (ROUTING_MODES-major) order, but the
    # validation intent is completeness + identity, not ordering.
    expected_set = set(expected_trials)
    if set(matrix_state.get("trials", [])) != expected_set:
        matrix_errors.append("matrix trial list does not match preregistered trials")
    if set(matrix_state.get("completed_trials", [])) != expected_set:
        matrix_errors.append("matrix completed trial list is incomplete (missing or extra)")
    first_provenance = next(iter(summaries.values()))["provenance"]
    matrix_source_hashes = matrix_state.get("frozen_source_sha256", {})
    for source_name, source_hash in first_provenance["source_sha256"].items():
        if matrix_source_hashes.get(source_name) != source_hash:
            matrix_errors.append(
                f"trial source hash is not bound to matrix state: {source_name}"
            )
    if matrix_source_hashes.get("dataset.feather") != first_provenance[
            "dataset_sha256"]:
        matrix_errors.append("trial dataset hash is not bound to matrix state")
    if matrix_errors:
        print(json.dumps({
            "status": "blocked",
            "reason": "matrix state identity validation failed",
            "errors": matrix_errors,
            "gate_verdict": "INCONCLUSIVE",
        }, indent=2))
        return 1

    fairness_errors = validate_cross_trial_fairness(summaries, manifests)
    if fairness_errors:
        print(json.dumps({
            "status": "blocked",
            "reason": "cross-trial fairness validation failed",
            "errors": fairness_errors,
            "gate_verdict": "INCONCLUSIVE",
        }, indent=2))
        return 1

    decision = decide_gate(summaries)
    decision["status"] = "done"
    print(json.dumps(decision, indent=2))
    if arguments.write_results:
        write_results_transactionally(summaries, decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
