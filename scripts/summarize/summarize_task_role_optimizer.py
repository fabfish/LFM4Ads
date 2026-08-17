#!/usr/bin/env python3
"""汇总开发筛选或四随机种子正式配对结果。"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache/task_role_optimizer_27k_siteB"
FORMAL_SEEDS = (42, 123, 456, 789)


def atomic_json_dump(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)


def load_results(pattern: str) -> list[tuple[Path, dict[str, object]]]:
    matches = sorted(Path(path) for path in glob.glob(str(OUT / pattern)))
    if not matches:
        raise SystemExit(f"没有匹配结果：{OUT / pattern}")
    loaded = []
    for path in matches:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "完成":
            raise ValueError(f"结果未完成：{path}")
        loaded.append((path, payload))
    return loaded


def config_from(payload: dict[str, object]) -> dict[str, object]:
    provenance = payload["provenance"]
    return {
        "optimizer_mode": provenance["optimizer_mode"],
        "expert_lr": provenance["expert_learning_rate"],
        "router_lr_ratio": provenance["router_learning_rate_ratio"],
        "shared_lr_ratio": provenance["shared_learning_rate_ratio"],
    }


def summarize_screen(pattern: str, output: Path, top_n: int) -> None:
    rows = []
    for path, payload in load_results(pattern):
        provenance = payload["provenance"]
        if not payload.get("development_only"):
            raise ValueError(f"开发筛选混入正式结果：{path}")
        if provenance.get("test_set_used"):
            raise ValueError(f"开发筛选泄漏测试集：{path}")
        if provenance.get("max_batches") or provenance.get("max_eval_batches"):
            raise ValueError(f"短步探针不得用于筛选学习率：{path}")
        routing = payload["selected_routing"]
        rows.append({
            "evidence": str(path.relative_to(ROOT)),
            "best_valid_macro": payload["best_valid_macro"],
            "best_epoch": payload["best_epoch"],
            "config": config_from(payload),
            "expert_coverage": routing["expert_coverage"],
            "all_expert_load_max_min_ratio": routing[
                "all_expert_load_max_min_ratio"],
            "selected_expert_reassignment_rate": routing[
                "selected_expert_reassignment_rate"],
        })
    rows.sort(key=lambda row: row["best_valid_macro"], reverse=True)
    selected = [row["config"] for row in rows[:top_n]]
    atomic_json_dump({
        "status_zh": "开发筛选完成",
        "selection_endpoint": "验证集八场景分别计算曲线下面积后等权平均",
        "test_set_used": False,
        "ranking": rows,
        "selected_configs": selected,
        "warning_zh": "开发随机种子只用于冻结候选，不构成最终效果结论。",
    }, output)
    print(f"开发筛选汇总写入：{output}")
    for index, row in enumerate(rows, 1):
        print(
            f"{index}. 验证集八场景等权指标={row['best_valid_macro']:.10f}；"
            f"配置={row['config']}；证据={row['evidence']}")


def config_key(payload: dict[str, object]) -> tuple[object, ...]:
    config = config_from(payload)
    return (
        config["optimizer_mode"], config["expert_lr"],
        config["router_lr_ratio"], config["shared_lr_ratio"],
    )


def verdict(deltas: list[float], noise_floor: float) -> str:
    mean_delta = statistics.mean(deltas)
    if all(value > 0 for value in deltas) and mean_delta > noise_floor:
        return "通过"
    if all(value < 0 for value in deltas) and abs(mean_delta) > noise_floor:
        return "失败"
    return "证据不足"


def summarize_formal(pattern: str, output: Path, baseline_index: int) -> None:
    groups: dict[tuple[object, ...], dict[int, tuple[Path, dict[str, object]]]] = {}
    for path, payload in load_results(pattern):
        provenance = payload["provenance"]
        if payload.get("development_only") or not provenance.get("test_set_used"):
            raise ValueError(f"正式汇总混入开发结果：{path}")
        if provenance.get("max_batches") or provenance.get("max_eval_batches"):
            raise ValueError(f"正式汇总混入截断结果：{path}")
        seed = int(provenance["seed"])
        groups.setdefault(config_key(payload), {})[seed] = (path, payload)
    ordered = sorted(groups)
    if len(ordered) != 2:
        raise ValueError(f"正式配对必须恰好两个配置，实际为 {len(ordered)}")
    if baseline_index not in (0, 1):
        raise ValueError("baseline_index must be 0 or 1")
    for key in ordered:
        missing = set(FORMAL_SEEDS) - set(groups[key])
        extra = set(groups[key]) - set(FORMAL_SEEDS)
        if missing or extra:
            raise ValueError(f"正式随机种子不完整：{key}, missing={missing}, extra={extra}")

    baseline_key = ordered[baseline_index]
    candidate_key = ordered[1 - baseline_index]
    baseline_values = [
        float(groups[baseline_key][seed][1]["test"]["macro"])
        for seed in FORMAL_SEEDS
    ]
    noise_floor = 2.0 * statistics.stdev(baseline_values)
    paired = []
    deltas = []
    for seed in FORMAL_SEEDS:
        baseline_path, baseline = groups[baseline_key][seed]
        candidate_path, candidate = groups[candidate_key][seed]
        baseline_device = baseline["provenance"]["device"]
        candidate_device = candidate["provenance"]["device"]
        if baseline_device != candidate_device:
            raise ValueError(f"随机种子 {seed} 跨显卡配对")
        baseline_value = float(baseline["test"]["macro"])
        candidate_value = float(candidate["test"]["macro"])
        delta = candidate_value - baseline_value
        deltas.append(delta)
        paired.append({
            "seed": seed,
            "device": baseline_device,
            "baseline_macro": baseline_value,
            "candidate_macro": candidate_value,
            "paired_delta": delta,
            "baseline_evidence": str(baseline_path.relative_to(ROOT)),
            "candidate_evidence": str(candidate_path.relative_to(ROOT)),
        })
    mean_delta = statistics.mean(deltas)
    floor_multiple = (
        abs(mean_delta) / noise_floor if noise_floor > 0
        else math.inf if mean_delta else 0.0)
    payload = {
        "status_zh": "正式配对汇总完成",
        "endpoint": "测试集八场景分别计算曲线下面积后等权平均",
        "baseline_config": {
            "optimizer_mode": baseline_key[0], "expert_lr": baseline_key[1],
            "router_lr_ratio": baseline_key[2],
            "shared_lr_ratio": baseline_key[3],
        },
        "candidate_config": {
            "optimizer_mode": candidate_key[0], "expert_lr": candidate_key[1],
            "router_lr_ratio": candidate_key[2],
            "shared_lr_ratio": candidate_key[3],
        },
        "paired_results": paired,
        "mean_paired_delta": mean_delta,
        "same_positive_direction_count": sum(value > 0 for value in deltas),
        "same_negative_direction_count": sum(value < 0 for value in deltas),
        "internal_noise_floor": noise_floor,
        "floor_multiple": floor_multiple,
        "verdict_zh": verdict(deltas, noise_floor),
        "cross_site_pairing": False,
    }
    atomic_json_dump(payload, output)
    print(f"正式配对汇总写入：{output}")
    print(
        f"平均配对差={mean_delta:+.10f}；内部随机波动范围={noise_floor:.10f}；"
        f"倍数={floor_multiple:.3f}；判定={payload['verdict_zh']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("screen", "formal"))
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--baseline-index", type=int, default=0)
    args = parser.parse_args()
    destination = args.output
    if not destination.is_absolute():
        destination = ROOT / destination
    if args.mode == "screen":
        summarize_screen(args.pattern, destination, args.top_n)
    else:
        summarize_formal(args.pattern, destination, args.baseline_index)


if __name__ == "__main__":
    main()
