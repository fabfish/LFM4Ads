#!/usr/bin/env python
"""Plan or execute the Task-Conditioned Mixture Routing (TCMR) matrix.

Execution requires a machine-readable authorization bound to the checked-in
driver hash. The fifteen trials run with one device per seed group so that all
routing modes of a seed share identical hardware (preserving the paired same-seed
AUC comparison); distinct seed groups run in parallel across the requested
devices.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
ROUTING_MODES = (
    "frozen-uniform-routing",
    "data-only-routing",
    "task-only-routing",
    "data-and-task-routing",
    "data-and-task-consistency-routing",
)
SEEDS = (42, 123, 456)
DRIVER_GLOB = "*-Task-Conditioned-Mixture-Routing-驱动.md"
AUTHORIZATION_PATH = (
    REPOSITORY / "scripts" /
    "task_conditioned_mixture_routing_authorization.json"
)
AUTHORIZED_SCOPE = "fifteen-trial-multi-device-seed-group-parallel"
MATRIX_POLL_INTERVAL_SECONDS = 30
INVARIANT_REPORT_PATH = (
    REPOSITORY / "cache" / "audit" / "task_conditioned_mixture_routing" /
    "task_conditioned_mixture_routing_invariants.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(routing_mode, seed):
    return f"task-conditioned-mixture-routing-{routing_mode}-seed-{seed}"


def seed_device_map(devices):
    """Map each seed to one device so all its routing modes share hardware.

    This keeps every same-seed paired comparison (DATR vs FUR / DOR) on
    identical hardware, which is required for the pooled-AUC pairing. Distinct
    seed groups may land on different devices and therefore run in parallel.
    """
    return {seed: devices[i % len(devices)] for i, seed in enumerate(SEEDS)}


def trial_commands(devices):
    device_by_seed = seed_device_map(devices)
    commands = []
    for routing_mode in ROUTING_MODES:
        for seed in SEEDS:
            device = device_by_seed[seed]
            name = run_name(routing_mode, seed)
            commands.append([
                sys.executable,
                "experiments/run_task_conditioned_mixture_routing.py",
                "--routing-mode", routing_mode,
                "--seed", str(seed),
                "--run-name", name,
                "--device", device,
                "--num-workers", "8",
            ])
    return commands


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "execute"))
    parser.add_argument(
        "--device", nargs="+", default=["cuda:0"],
        help="one or more devices; each seed group is pinned to one device",
    )
    parser.add_argument(
        "--acknowledge-document-driver",
        action="store_true",
        help="required in addition to machine-readable authorization",
    )
    return parser.parse_args(argv)


def verify_driver_exists():
    drivers = sorted((REPOSITORY / "docs").glob(DRIVER_GLOB))
    if not drivers:
        raise FileNotFoundError(
            "no Task-Conditioned Mixture Routing driver document found"
        )
    return drivers[-1]


def frozen_source_paths(driver):
    return (
        REPOSITORY / "model.py",
        REPOSITORY / "train.py",
        REPOSITORY / "task_conditioned_mixture_routing_protocol.py",
        REPOSITORY / "dataset.py",
        REPOSITORY / "fields.py",
        REPOSITORY / "dataset.feather",
        REPOSITORY / "experiments/run_task_conditioned_mixture_routing.py",
        REPOSITORY / "scripts" / "verify_task_conditioned_mixture_routing.py",
        Path(__file__).resolve(),
        REPOSITORY / "scripts" / "summarize_task_conditioned_mixture_routing.py",
        AUTHORIZATION_PATH,
        driver,
    )


def source_hashes(driver):
    paths = frozen_source_paths(driver)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen source files: {missing}")
    return {
        str(path.relative_to(REPOSITORY)): sha256_file(path)
        for path in paths
    }


def verify_authorization(driver):
    if not AUTHORIZATION_PATH.is_file():
        raise PermissionError(
            f"machine-readable authorization is missing: {AUTHORIZATION_PATH}"
        )
    with open(AUTHORIZATION_PATH) as stream:
        authorization = json.load(stream)
    expected_driver_hash = sha256_file(driver)
    if authorization.get("status") != "authorized":
        raise PermissionError(
            "TCMR matrix is not authorized; authorization status must be 'authorized'"
        )
    if authorization.get("driver_sha256") != expected_driver_hash:
        raise PermissionError(
            "authorization is not bound to the current driver SHA-256"
        )
    if authorization.get("driver_path") != str(driver.relative_to(REPOSITORY)):
        raise PermissionError("authorization driver path does not match current driver")
    if authorization.get("authorized_scope") != AUTHORIZED_SCOPE:
        raise PermissionError(
            f"authorization scope must be '{AUTHORIZED_SCOPE}' for the formal matrix"
        )
    return authorization


def verify_or_create_invariant_report(driver):
    if not INVARIANT_REPORT_PATH.exists():
        subprocess.run([
            sys.executable,
            "scripts/verify_task_conditioned_mixture_routing.py",
            "--write-report",
        ], cwd=REPOSITORY, check=True)
    with open(INVARIANT_REPORT_PATH) as stream:
        report = json.load(stream)
    if report.get("status") != "pass":
        raise RuntimeError("TCMR invariant report is not pass")
    provenance = report.get("provenance", {})
    if provenance.get("driver_sha256") != sha256_file(driver):
        raise RuntimeError("TCMR invariant report is stale for the current driver")
    expected_sources = {
        str(path.relative_to(REPOSITORY)): sha256_file(path)
        for path in (
            REPOSITORY / "model.py",
            REPOSITORY / "task_conditioned_mixture_routing_protocol.py",
            REPOSITORY / "scripts" / "verify_task_conditioned_mixture_routing.py",
        )
    }
    if provenance.get("source_sha256") != expected_sources:
        raise RuntimeError("TCMR invariant report source hashes are stale")
    return report


def matrix_paths(commands):
    run_root = REPOSITORY / "cache" / "task_conditioned_mixture_routing"
    matrix_directory = (
        REPOSITORY / "cache" / "manifests" /
        "task_conditioned_mixture_routing"
    )
    state_path = matrix_directory / "matrix_state.json"
    trial_paths = []
    for command in commands:
        name = command[command.index("--run-name") + 1]
        trial_paths.extend((
            run_root / name,
            REPOSITORY / "logs" / f"{name}.log",
        ))
    output_paths = (
        REPOSITORY / "result_task_conditioned_mixture_routing.csv",
        run_root / "gate_decision.json",
        run_root / "aggregate_result_manifest.json",
    )
    return matrix_directory, state_path, trial_paths + list(output_paths)


def prepare_matrix_state_path(commands):
    matrix_directory, state_path, _ = matrix_paths(commands)
    if state_path.exists():
        raise FileExistsError(f"matrix state already exists: {state_path}")
    (REPOSITORY / "logs").mkdir(parents=True, exist_ok=True)
    matrix_directory.mkdir(parents=True, exist_ok=True)
    return state_path


def clean_previous_aborted_matrix(commands):
    """Remove a previous matrix state that did not finish.

    A matrix that ended in ``blocked`` / ``preflight`` / ``running`` left only
    partial artifacts (aborted trial directories, stale logs, an incomplete
    state file). These must be removed before a re-execution can pass the
    immutable-path preflight. A ``done`` matrix is never touched; re-running it
    is refused so completed evidence is preserved.
    """
    _, state_path, immutable_paths = matrix_paths(commands)
    if not state_path.exists():
        return
    try:
        with open(state_path) as stream:
            previous = json.load(stream)
    except (json.JSONDecodeError, OSError):
        previous = {}
    if previous.get("status") == "done":
        raise FileExistsError(
            f"refusing to overwrite a completed matrix state: {state_path}"
        )
    state_path.unlink()
    for path in immutable_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def preflight_immutable_paths(commands):
    _, _, immutable_paths = matrix_paths(commands)
    conflicts = [str(path) for path in immutable_paths if path.exists()]
    if conflicts:
        raise FileExistsError(
            f"refusing to start matrix with existing immutable paths: {conflicts}"
        )


def write_state_atomic(path, state):
    temporary_path = path.with_suffix(".json.tmp")
    with open(temporary_path, "w") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def execute_matrix(commands, driver):
    authorization = verify_authorization(driver)
    clean_previous_aborted_matrix(commands)
    state_path = prepare_matrix_state_path(commands)
    devices = sorted({command[command.index("--device") + 1]
                      for command in commands})
    per_device = defaultdict(list)
    for command in commands:
        device = command[command.index("--device") + 1]
        per_device[device].append(command)
    state = {
        "status": "preflight",
        "driver": str(driver.relative_to(REPOSITORY)),
        "driver_sha256": sha256_file(driver),
        "authorization": authorization,
        "execution": f"{len(devices)}-device seed-group parallel",
        "devices": devices,
        "seed_device_map": {str(seed): devices[i % len(devices)]
                            for i, seed in enumerate(SEEDS)},
        "trials": [command[command.index("--run-name") + 1]
                   for command in commands],
        "completed_trials": [],
    }
    write_state_atomic(state_path, state)

    current_trial = None
    try:
        preflight_immutable_paths(commands)
        invariant_report = verify_or_create_invariant_report(driver)
        frozen_hashes = source_hashes(driver)
        state.update({
            "status": "running",
            "invariant_report_sha256": sha256_file(INVARIANT_REPORT_PATH),
            "invariant_report": invariant_report,
            "frozen_source_sha256": frozen_hashes,
        })
        write_state_atomic(state_path, state)

        next_index = {device: 0 for device in devices}
        running = {}  # subprocess.Popen -> (device, name, command)
        completed = []
        while (any(next_index[device] < len(per_device[device])
                    for device in devices) or running):
            busy_devices = {info[0] for info in running.values()}
            for device in devices:
                if device in busy_devices:
                    continue
                if next_index[device] >= len(per_device[device]):
                    continue
                command = per_device[device][next_index[device]]
                next_index[device] += 1
                name = command[command.index("--run-name") + 1]
                current_trial = name
                if source_hashes(driver) != frozen_hashes:
                    raise RuntimeError(
                        "frozen source hashes changed during matrix execution"
                    )
                log_path = REPOSITORY / "logs" / f"{name}.log"
                trial_environment = os.environ.copy()
                trial_environment[
                    "TASK_CONDITIONED_MIXTURE_ROUTING_MATRIX_EXECUTION"
                ] = "authorized"
                process = subprocess.Popen(
                    command, cwd=REPOSITORY, stdout=open(log_path, "x"),
                    stderr=subprocess.STDOUT, env=trial_environment,
                )
                running[process] = (device, name, command)
            for process in list(running):
                if process.poll() is not None:
                    device, name, command = running.pop(process)
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"trial {name} exited with code {process.returncode}"
                        )
                    completed.append(name)
                    state["completed_trials"] = completed
                    write_state_atomic(state_path, state)
            if not running and all(
                    next_index[device] >= len(per_device[device])
                    for device in devices):
                break
            time.sleep(MATRIX_POLL_INTERVAL_SECONDS)
        if source_hashes(driver) != frozen_hashes:
            raise RuntimeError("frozen source hashes changed before matrix completion")
        state["status"] = "done"
        write_state_atomic(state_path, state)
        return 0
    except Exception as error:
        state["status"] = "blocked"
        state["blocked_trial"] = current_trial
        state["error"] = f"{type(error).__name__}: {error}"
        write_state_atomic(state_path, state)
        raise


def main(argv=None):
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    driver = verify_driver_exists()
    commands = trial_commands(arguments.device)
    print(f"devices: {arguments.device}")
    if arguments.action == "plan":
        print(f"driver: {driver}")
        print("matrix: five routing modes x three paired seeds = fifteen trials")
        for command in commands:
            print(" ".join(command))
        return 0
    if not arguments.acknowledge_document_driver:
        raise SystemExit(
            "execute requires --acknowledge-document-driver and authorized JSON"
        )
    return execute_matrix(commands, driver)


if __name__ == "__main__":
    sys.exit(main())
