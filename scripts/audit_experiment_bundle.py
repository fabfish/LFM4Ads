#!/usr/bin/env python
"""Build a derived, non-destructive audit bundle for a historical experiment."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parent.parent
HISTORICAL_MANIFEST_DIRECTORY = (
    REPOSITORY / "cache" / "manifests" / "sample_weighting"
)
REPORT_SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recorded_path(recorded_path, artifact_label, run_identifier):
    expected = {
        "checkpoint": REPOSITORY / "cache" / f"{run_identifier}.pt",
        "summary": (
            REPOSITORY / "cache" /
            f"moe_pretrain_summary_{run_identifier}.json"
        ),
        "log": REPOSITORY / "logs" / f"sample_weighting_{run_identifier}.log",
        "csv": REPOSITORY / f"result_{run_identifier}.csv",
    }.get(artifact_label)
    if expected is None:
        return {"status": "unsupported-artifact-label", "resolved": None}
    candidate = Path(recorded_path) if recorded_path else None
    if candidate and candidate.exists() and candidate.resolve() == expected.resolve():
        return {"status": "recorded-path", "resolved": candidate.resolve()}
    if expected.exists():
        return {"status": "resolved-by-expected-run-structure", "resolved": expected.resolve()}
    return {"status": "unresolved", "resolved": None}


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def dataset_scope():
    dataset_path = REPOSITORY / "dataset.feather"
    data_frame = pd.read_feather(dataset_path, columns=["date", "tab"])
    train = data_frame.query("date < 20220503")
    validation = data_frame.query("20220503 <= date < 20220506")
    test = data_frame.query("20220506 <= date")

    def counts(frame):
        return {
            str(int(task)): int(count)
            for task, count in frame["tab"].value_counts().sort_index().items()
        }

    return {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "split_task_sample_counts": {
            "train": counts(train),
            "validation": counts(validation),
            "test": counts(test),
        },
        "training_task_count": int(train["tab"].nunique()),
        "formal_report_scenarios": REPORT_SCENARIOS,
        "formal_report_scenario_count": len(REPORT_SCENARIOS),
        "scope_warning": (
            "training uses all task identifiers; macro reporting covers only the "
            "eight declared report scenarios"
        ),
    }


def normalize_router_semantics(manifest):
    config = manifest.get("config", {})
    model = config.get("model")
    router = config.get("router")
    frozen = config.get("freeze_router") or router == "frozen"
    if model == "lowrank-full-dim" and frozen:
        return (
            "deterministic uniform gate over low-rank full-dimensional experts; "
            "zero noise; not vanilla-equivalent"
        )
    if frozen:
        return "deterministic uniform gate with zero routing noise"
    if router == "soft":
        return "learnable data-only soft routing"
    if router == "none":
        return "dense model without routing"
    return config.get("router_semantics") or "historical semantics not recorded"


def git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
        capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip()


def build_bundle(historical_run_identifier):
    manifest_path = (
        HISTORICAL_MANIFEST_DIRECTORY / f"{historical_run_identifier}.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"historical manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    recorded_paths = manifest.get("paths", {})
    paths = {}
    hashes = {}
    for label, recorded in recorded_paths.items():
        resolution = resolve_recorded_path(
            recorded, label, historical_run_identifier,
        )
        resolved = resolution["resolved"]
        paths[label] = {
            "recorded": recorded,
            "resolution_status": resolution["status"],
            "resolved_current_workspace": str(resolved) if resolved else None,
        }
        if resolved and resolved.is_file():
            hashes[label] = sha256_file(resolved)

    summary = None
    summary_resolved = paths.get("summary", {}).get("resolved_current_workspace")
    summary_identity_valid = False
    if summary_resolved:
        summary = read_json(summary_resolved)
        summary_config = summary.get("config", {})
        summary_identity_valid = summary_config.get("run_code") in (
            None, historical_run_identifier,
        )

    invariant_path = HISTORICAL_MANIFEST_DIRECTORY / "equ_swg_status.json"
    invariant = read_json(invariant_path) if invariant_path.exists() else None
    return {
        "audit_schema": "historical-experiment-derived-audit-bundle-v1",
        "historical_run_identifier": historical_run_identifier,
        "historical_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "status": manifest.get("status"),
        },
        "conclusion_eligible": (
            manifest.get("status") == "succeeded"
            and summary is not None
            and summary_identity_valid
        ),
        "summary_identity_valid": summary_identity_valid,
        "config": manifest.get("config", {}),
        "normalized_router_semantics": normalize_router_semantics(manifest),
        "paths": paths,
        "artifact_sha256": hashes,
        "results": summary,
        "sample_weighting_invariant": invariant,
        "dataset_scope": dataset_scope(),
        "provenance": {
            "historical_git_commit": manifest.get("git_commit"),
            "audit_code_git_commit": git_commit(),
            "historical_source_hashes": manifest.get(
                "provenance", {},
            ).get("source_hashes", {}),
        },
        "immutability": (
            "derived bundle only; historical manifest, summary, checkpoint, CSV, "
            "and log are not modified"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("historical_run_identifier")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    output_path = Path(arguments.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite audit bundle: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(arguments.historical_run_identifier)
    with open(output_path, "x") as stream:
        json.dump(bundle, stream, indent=2, ensure_ascii=False)
    print(f"[audit-bundle] {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
