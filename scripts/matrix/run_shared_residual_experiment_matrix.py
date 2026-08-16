#!/usr/bin/env python3
"""Dual-GPU, seed-pinned, failure-isolated shared-residual experiment runner."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))

from shared_residual_continual_protocol import (  # noqa: E402
    DEFAULT_ORDERS,
    FORMAL_SEEDS,
    SPECIALIST_LEARNING_RATES,
    order_name,
    sha256_file,
    write_json_atomic,
)


DRIVER_GLOB = "*-共享残差混合专家-函数保持与持续学习-驱动.md"
AUTHORIZATION_PATH = REPOSITORY / "scripts" / "shared_residual_experiment_authorization.json"
INVARIANT_REPORT = (
    REPOSITORY / "cache" / "audit" / "shared_residual_continual" /
    "shared_residual_experiment_invariants.json"
)
MATRIX_STATE = (
    REPOSITORY / "cache" / "manifests" / "shared_residual_continual" /
    "matrix_state.json"
)
RUN_ROOT = REPOSITORY / "cache" / "shared_residual_continual"
POLL_SECONDS = 10


def lr_slug(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def run_name(arm, learning_rate, order, seed):
    return (
        "shared-residual-specialist-screen-"
        f"{arm}-lr-{lr_slug(learning_rate)}-{order_name(order)}-seed-{seed}"
    )


def build_trials(devices):
    device_by_seed = {
        seed: devices[index % len(devices)]
        for index, seed in enumerate(FORMAL_SEEDS)
    }
    trials = []
    for order in DEFAULT_ORDERS:
        configurations = [("task-head-only", 5e-4)] + [
            ("specialist-only", learning_rate)
            for learning_rate in SPECIALIST_LEARNING_RATES
        ]
        for arm, learning_rate in configurations:
            for seed in FORMAL_SEEDS:
                name = run_name(arm, learning_rate, order, seed)
                trials.append({
                    "run_name": name,
                    "arm": arm,
                    "learning_rate": learning_rate,
                    "order": list(order),
                    "seed": seed,
                    "device": device_by_seed[seed],
                    "command": [
                        sys.executable,
                        "experiments/run_shared_residual_continual_trial.py",
                        "--run-name", name,
                        "--arm", arm,
                        "--expert-learning-rate", str(learning_rate),
                        "--order", ",".join(str(value) for value in order),
                        "--seed", str(seed),
                        "--device", device_by_seed[seed],
                        "--num-workers", "4",
                    ],
                })
    return trials, device_by_seed


def driver_path():
    drivers = sorted((REPOSITORY / "docs").glob(DRIVER_GLOB))
    if not drivers:
        raise FileNotFoundError("shared-residual driver document is missing")
    return drivers[-1]


def verify_authorization(driver):
    if not AUTHORIZATION_PATH.is_file():
        raise PermissionError("machine-readable authorization is missing")
    with open(AUTHORIZATION_PATH) as stream:
        authorization = json.load(stream)
    if authorization.get("status") != "authorized":
        raise PermissionError("authorization status must be authorized")
    if authorization.get("driver_path") != str(driver.relative_to(REPOSITORY)):
        raise PermissionError("authorization driver path mismatch")
    if authorization.get("driver_sha256") != sha256_file(driver):
        raise PermissionError("authorization driver hash is stale")
    if authorization.get("authorized_scope") != (
            "function-and-lr-invariants-plus-two-task-specialist-screen"):
        raise PermissionError("authorization scope does not permit this matrix")
    return authorization


def frozen_paths(driver):
    return (
        REPOSITORY / "model.py",
        REPOSITORY / "train.py",
        REPOSITORY / "dataset.py",
        REPOSITORY / "fields.py",
        REPOSITORY / "shared_residual_continual_protocol.py",
        REPOSITORY / "experiments/run_shared_residual_continual_trial.py",
        REPOSITORY / "scripts" / "verify_shared_residual_experiment.py",
        REPOSITORY / "scripts" / "shared_residual_experiment.sh",
        Path(__file__).resolve(),
        REPOSITORY / "scripts" / "summarize_shared_residual_experiment.py",
        AUTHORIZATION_PATH,
        driver,
        REPOSITORY / "dataset.feather",
        *(REPOSITORY / "cache" / f"vanilla_from_scratch_seed{seed}.pt"
          for seed in FORMAL_SEEDS),
    )


def source_hashes(driver):
    paths = frozen_paths(driver)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")
    return {
        str(path.relative_to(REPOSITORY)): sha256_file(path)
        for path in paths
    }


def verify_invariants(driver):
    if not INVARIANT_REPORT.exists():
        subprocess.run(
            [sys.executable, "scripts/verify/verify_shared_residual_experiment.py",
             "--write-report"],
            cwd=REPOSITORY, check=True,
        )
    with open(INVARIANT_REPORT) as stream:
        report = json.load(stream)
    if report.get("status") != "pass" or report.get("gate_verdict") != "PASS":
        raise RuntimeError("G1/G2 invariant report is not PASS")
    provenance = report.get("provenance", {})
    if provenance.get("driver_sha256") != sha256_file(driver):
        raise RuntimeError("invariant report is stale for the driver")
    expected = {
        str(path.relative_to(REPOSITORY)): sha256_file(path)
        for path in (
            REPOSITORY / "model.py",
            REPOSITORY / "shared_residual_continual_protocol.py",
            REPOSITORY / "scripts" / "verify_shared_residual_experiment.py",
        )
    }
    if provenance.get("source_sha256") != expected:
        raise RuntimeError("invariant report is stale for current source")
    upcycling = report.get("checks", {}).get("function_preserving_upcycling", {})
    for seed in FORMAL_SEEDS:
        checkpoint = REPOSITORY / "cache" / f"vanilla_from_scratch_seed{seed}.pt"
        recorded = upcycling.get(str(seed), {}).get("dense_checkpoint_sha256")
        if recorded != sha256_file(checkpoint):
            raise RuntimeError(f"G1 report dense checkpoint hash mismatch for seed {seed}")
    return report


def read_run_status(name):
    manifest = RUN_ROOT / name / "manifest.json"
    if not manifest.exists():
        return None
    try:
        with open(manifest) as stream:
            return json.load(stream).get("status", "corrupt")
    except Exception:
        return "corrupt"


def validate_existing_done_trial(trial, frozen_hashes, invariant_report):
    directory = RUN_ROOT / trial["run_name"]
    manifest_path = directory / "manifest.json"
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "done" or manifest.get("run_name") != trial["run_name"]:
        raise ValueError("manifest identity/status mismatch")
    config = manifest.get("config", {})
    expected = {
        "arm": trial["arm"],
        "expert_learning_rate": trial["learning_rate"],
        "head_learning_rate": 5e-4,
        "order": trial["order"],
        "seed": trial["seed"],
        "device": trial["device"],
    }
    if any(config.get(field) != value for field, value in expected.items()):
        raise ValueError("manifest frozen trial configuration mismatch")
    for relative, digest in manifest.get("provenance", {}).get(
            "source_sha256", {}).items():
        if frozen_hashes.get(relative) != digest:
            raise ValueError(f"trial source hash mismatch: {relative}")
    dense_digest = invariant_report["checks"]["function_preserving_upcycling"][
        str(trial["seed"])
    ]["dense_checkpoint_sha256"]
    if manifest.get("provenance", {}).get("dense_checkpoint_sha256") != dense_digest:
        raise ValueError("trial dense checkpoint is not bound to G1")
    for filename, hash_field in (
        ("summary.json", "summary_sha256"),
        ("trial_result.csv", "trial_result_sha256"),
        ("model_state.pt", "checkpoint_sha256"),
    ):
        path = directory / filename
        if not path.is_file() or manifest.get(hash_field) != sha256_file(path):
            raise ValueError(f"trial artifact hash mismatch: {filename}")


def initial_state(trials, devices, device_by_seed, driver, authorization,
                  frozen_hashes, invariant_report):
    expected_identity = {
        "driver": str(driver.relative_to(REPOSITORY)),
        "driver_sha256": sha256_file(driver),
        "authorization": authorization,
        "devices": devices,
        "seed_device_map": {str(key): value for key, value in device_by_seed.items()},
        "frozen_source_sha256": frozen_hashes,
        "invariant_report": str(INVARIANT_REPORT.relative_to(REPOSITORY)),
        "invariant_report_sha256": sha256_file(INVARIANT_REPORT),
        "trials": [trial["run_name"] for trial in trials],
    }
    existing = {}
    if MATRIX_STATE.exists():
        with open(MATRIX_STATE) as stream:
            existing = json.load(stream)
        mismatches = [
            field for field, value in expected_identity.items()
            if existing.get(field) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"existing matrix provenance mismatch: {mismatches}"
            )
    state = {
        **existing,
        **expected_identity,
        "status": "running",
        "completed_trials": existing.get("completed_trials", []),
        "failed_trials": existing.get("failed_trials", []),
        "skipped_existing_trials": existing.get("skipped_existing_trials", []),
    }
    MATRIX_STATE.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(MATRIX_STATE, state)
    return state


def execute(trials, devices, driver):
    if devices != ["cuda:0", "cuda:1"]:
        raise ValueError(
            "formal execute requires exactly --devices cuda:0 cuda:1"
        )
    authorization = verify_authorization(driver)
    invariant_report = verify_invariants(driver)
    frozen_hashes = source_hashes(driver)
    _, device_by_seed = build_trials(devices)
    state = initial_state(
        trials, devices, device_by_seed, driver, authorization,
        frozen_hashes, invariant_report,
    )
    completed = set(state["completed_trials"])
    failed = {item["run_name"]: item for item in state["failed_trials"]}
    skipped = set(state["skipped_existing_trials"])
    queues = defaultdict(list)
    for trial in trials:
        status = read_run_status(trial["run_name"])
        if status == "done":
            try:
                validate_existing_done_trial(trial, frozen_hashes, invariant_report)
            except Exception as error:
                failed[trial["run_name"]] = {
                    "run_name": trial["run_name"],
                    "return_code": None,
                    "reason": f"invalid pre-existing done trial: {error}",
                }
            else:
                completed.add(trial["run_name"])
                skipped.add(trial["run_name"])
        elif status is not None:
            failed[trial["run_name"]] = {
                "run_name": trial["run_name"],
                "return_code": None,
                "reason": f"pre-existing immutable run status={status}",
            }
        else:
            queues[trial["device"]].append(trial)

    running = {}
    log_handles = {}
    try:
        while any(queues.values()) or running:
            busy = {information["device"] for information in running.values()}
            for device in devices:
                if device in busy or not queues[device]:
                    continue
                if source_hashes(driver) != frozen_hashes:
                    raise RuntimeError("frozen source changed during matrix execution")
                trial = queues[device].pop(0)
                log_path = REPOSITORY / "logs" / f"{trial['run_name']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if log_path.exists():
                    failed[trial["run_name"]] = {
                        "run_name": trial["run_name"],
                        "return_code": None,
                        "reason": "immutable log path already exists",
                    }
                    continue
                environment = os.environ.copy()
                environment["SHARED_RESIDUAL_EXPERIMENT_MATRIX_EXECUTION"] = "authorized"
                handle = open(log_path, "x")
                try:
                    process = subprocess.Popen(
                        trial["command"], cwd=REPOSITORY,
                        stdout=handle, stderr=subprocess.STDOUT,
                        env=environment,
                    )
                except Exception:
                    handle.close()
                    raise
                running[process] = trial
                log_handles[process] = handle
                state["active_trials"] = [
                    item["run_name"] for item in running.values()
                ]
                write_json_atomic(MATRIX_STATE, state)

            for process in list(running):
                return_code = process.poll()
                if return_code is None:
                    continue
                trial = running.pop(process)
                log_handles.pop(process).close()
                if return_code == 0 and read_run_status(trial["run_name"]) == "done":
                    completed.add(trial["run_name"])
                else:
                    failed[trial["run_name"]] = {
                        "run_name": trial["run_name"],
                        "return_code": return_code,
                        "reason": "trial process failed or did not produce done manifest",
                    }
                state["completed_trials"] = sorted(completed)
                state["failed_trials"] = sorted(
                    failed.values(), key=lambda item: item["run_name"]
                )
                state["active_trials"] = [
                    item["run_name"] for item in running.values()
                ]
                write_json_atomic(MATRIX_STATE, state)
            if running or any(queues.values()):
                time.sleep(POLL_SECONDS)
    except Exception as error:
        aborted = []
        for process, trial in list(running.items()):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            aborted.append({
                "run_name": trial["run_name"],
                "return_code": process.returncode,
                "reason": "aborted by matrix-level failure",
            })
        state["status"] = "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        state["completed_trials"] = sorted(completed)
        state["failed_trials"] = sorted(
            [*failed.values(), *aborted], key=lambda item: item["run_name"]
        )
        state["active_trials"] = []
        write_json_atomic(MATRIX_STATE, state)
        raise
    finally:
        for handle in log_handles.values():
            handle.close()

    if source_hashes(driver) != frozen_hashes:
        state["status"] = "blocked"
        state["error"] = "frozen source changed before matrix completion"
        write_json_atomic(MATRIX_STATE, state)
        raise RuntimeError(state["error"])

    state["completed_trials"] = sorted(completed)
    state["failed_trials"] = sorted(
        failed.values(), key=lambda item: item["run_name"]
    )
    state["skipped_existing_trials"] = sorted(skipped)
    state["active_trials"] = []
    state["status"] = "done" if not failed else "done-with-failures"
    write_json_atomic(MATRIX_STATE, state)
    if failed:
        return 1
    aggregate = subprocess.run(
        [sys.executable, "scripts/summarize/summarize_shared_residual_experiment.py",
         "--write-results"],
        cwd=REPOSITORY, check=False,
    )
    if aggregate.returncode != 0:
        state["status"] = "blocked"
        state["error"] = f"aggregation failed with code {aggregate.returncode}"
        write_json_atomic(MATRIX_STATE, state)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "execute", "status"))
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    trials, device_by_seed = build_trials(arguments.devices)
    if arguments.action == "status":
        payload = {"status": "not-started"}
        if MATRIX_STATE.exists():
            with open(MATRIX_STATE) as stream:
                payload = json.load(stream)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    driver = driver_path()
    if arguments.action == "plan":
        print(json.dumps({
            "driver": str(driver.relative_to(REPOSITORY)),
            "trial_count": len(trials),
            "seed_device_map": {str(key): value for key, value in device_by_seed.items()},
            "trials": trials,
        }, indent=2, ensure_ascii=False))
        return 0
    return execute(trials, arguments.devices, driver)


if __name__ == "__main__":
    sys.exit(main())
