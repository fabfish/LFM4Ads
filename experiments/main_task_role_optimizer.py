#!/usr/bin/env python3
"""按参数角色隔离子任务优化器的 27K 训练入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import GpuBatches, Split  # noqa: E402
from experiments.main_macro_auc import (  # noqa: E402
    EXPECTED_COUNTS,
    MACRO_SCENARIOS,
    build,
    evaluate_all,
)
from task_role_optimizer import (  # noqa: E402
    PARAMETER_ROLES,
    ParameterRoleRegistry,
    RoleHyperParameters,
    SharedAdamWBatchOptimizer,
    TaskRoleOptimizer,
    atomic_torch_save,
    capture_random_state,
    collect_task_gradients,
    default_role_hyperparameters,
    restore_random_state,
)
from task_role_optimizer_protocol import (  # noqa: E402
    EXPECTED_MODEL_PARAMS,
    FrozenTrainingConfig,
)

OUT_DIR = Path(os.environ.get(
    "LFM_TASK_ROLE_OUT", "cache/task_role_optimizer_27k_siteB"))


class LimitedSource:
    def __init__(self, source: Iterable[dict[str, torch.Tensor]], limit: int):
        self.source = source
        self.limit = int(limit)

    def __iter__(self):
        for index, batch in enumerate(self.source, 1):
            yield batch
            if index >= self.limit:
                break


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def atomic_json_dump(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)


def role_hyperparameters(args: argparse.Namespace) -> dict[str, RoleHyperParameters]:
    if args.optimizer_mode == "role_isolated":
        return default_role_hyperparameters(
            expert_learning_rate=args.expert_lr,
            router_learning_rate_ratio=args.router_lr_ratio,
            shared_learning_rate_ratio=args.shared_lr_ratio,
            weight_decay=args.weight_decay,
        )
    if args.optimizer_mode == "task_state_uniform":
        uniform = RoleHyperParameters(
            learning_rate=args.expert_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            use_first_moment=True,
        )
        return {role: uniform for role in PARAMETER_ROLES}
    raise ValueError(f"no role hyperparameters for {args.optimizer_mode}")


def build_optimizer(
    args: argparse.Namespace,
    registry: ParameterRoleRegistry,
):
    if args.optimizer_mode == "shared_adamw":
        return SharedAdamWBatchOptimizer(
            registry,
            learning_rate=args.expert_lr,
            weight_decay=args.weight_decay,
        )
    return TaskRoleOptimizer(registry, role_hyperparameters(args))


def router_statistics(model: torch.nn.Module, previous=None) -> dict[str, object]:
    assignments = []
    router_weights = []
    for layer in model.cross_layers:
        raw = layer.router.embed.weight.detach().float().cpu()
        gate = raw.softmax(dim=-1) * raw.shape[1]
        top = gate.topk(model.top_k, dim=-1).indices
        assignment = torch.zeros_like(gate, dtype=torch.bool)
        assignment.scatter_(1, top, True)
        assignments.append(assignment)
        router_weights.append(raw.tolist())
    stacked = torch.stack(assignments)
    loads = stacked.sum(dim=(0, 1)).to(torch.float64)
    positive = loads[loads > 0]
    coverage = int((loads > 0).sum())
    used_load_ratio = (
        float(positive.max() / positive.min()) if positive.numel() else None)
    all_expert_load_ratio = (
        float("inf") if coverage < loads.numel()
        else float(loads.max() / loads.min()))

    joint = stacked.to(torch.float64).sum(dim=0)
    joint /= joint.sum()
    scenario_probability = joint.sum(dim=1, keepdim=True)
    expert_probability = joint.sum(dim=0, keepdim=True)
    independent = scenario_probability * expert_probability
    nonzero = joint > 0
    mutual_information = float(
        (joint[nonzero] * (joint[nonzero] / independent[nonzero]).log()).sum())

    reassignment_rate = None
    if previous is not None:
        changed = torch.logical_xor(stacked, previous).sum()
        denominator = stacked.shape[0] * stacked.shape[1] * model.top_k * 2
        reassignment_rate = float(changed / denominator)
    return {
        "assignments": stacked,
        "report": {
            "router_weights": router_weights,
            "top_experts": [
                [[int(value) for value in row] for row in layer.tolist()]
                for layer in torch.stack([
                    item.to(torch.int64).topk(model.top_k, dim=-1).indices
                    for item in assignments
                ])
            ],
            "expert_coverage": coverage,
            "expert_loads": [int(value) for value in loads.tolist()],
            "all_expert_load_max_min_ratio": all_expert_load_ratio,
            "used_expert_load_max_min_ratio": used_load_ratio,
            "scenario_expert_mutual_information": mutual_information,
            "selected_expert_reassignment_rate": reassignment_rate,
        },
    }


def train_one_epoch(
    model: torch.nn.Module,
    source,
    registry: ParameterRoleRegistry,
    optimizer,
    max_batches: int,
) -> dict[str, object]:
    model.train()
    loss_sum = 0.0
    task_updates = 0
    batch_count = 0
    role_update_sum = {role: 0.0 for role in PARAMETER_ROLES}
    started = time.time()
    for batch_index, batch in enumerate(source, 1):
        gradients = collect_task_gradients(model, batch, registry)
        optimizer.step(gradients)
        batch_count += 1
        task_updates += len(gradients)
        loss_sum += sum(item.loss or 0.0 for item in gradients) / len(gradients)
        metrics = getattr(optimizer, "last_step_metrics", {})
        for role, value in metrics.get("role_update_norm", {}).items():
            role_update_sum[role] += float(value)
        if max_batches and batch_index >= max_batches:
            break
    if not batch_count:
        raise RuntimeError("training source yielded no batches")
    return {
        "mean_balanced_loss": loss_sum / batch_count,
        "mixed_batches": batch_count,
        "task_updates": task_updates,
        "mean_role_update_norm": {
            role: value / batch_count for role, value in role_update_sum.items()
        },
        "wall_clock_sec": time.time() - started,
    }


def save_checkpoint(
    destination: Path,
    model: torch.nn.Module,
    optimizer,
    train_source: GpuBatches,
    epoch: int,
    best: dict[str, object],
    history: list[dict[str, object]],
    since_improve: int,
    args: argparse.Namespace,
) -> None:
    atomic_torch_save({
        "format_version": 1,
        "model": cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "data_generator_state": train_source._gen.get_state().cpu(),
        "random_state": capture_random_state(),
        "epoch": epoch,
        "best": best,
        "history": history,
        "since_improve": since_improve,
        "config": vars(args),
    }, destination)


def load_checkpoint(
    source: Path,
    model: torch.nn.Module,
    optimizer,
    train_source: GpuBatches,
    args: argparse.Namespace,
):
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported training checkpoint format")
    if payload.get("config") != vars(args):
        raise ValueError("checkpoint configuration differs from current command")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    train_source._gen.set_state(payload["data_generator_state"])
    restore_random_state(payload["random_state"])
    return (
        int(payload["epoch"]) + 1,
        payload["best"],
        payload["history"],
        int(payload["since_improve"]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if os.environ.get("LFM_SITE", "B") != "B":
        raise SystemExit("本入口只允许在第二台机器命名空间运行")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set, valid_set, test_set = Split("all")
    counts = {
        "train": len(train_set),
        "valid": len(valid_set),
        "test": len(test_set),
    }
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"数据切分行数失败：{counts} != {EXPECTED_COUNTS}")
    train_source = GpuBatches(
        train_set, args.batch_size, args.device, shuffle=True, seed=args.seed)
    valid_source = GpuBatches(
        valid_set, args.batch_size, args.device, shuffle=False)
    test_source = None
    if not args.development:
        test_source = GpuBatches(
            test_set, args.batch_size, args.device, shuffle=False)
    del train_set, valid_set, test_set

    model, model_info = build(
        "moe", 5, True, args.device, top_k=2)
    if model_info["total_params"] != EXPECTED_MODEL_PARAMS:
        raise SystemExit(
            f"参数量失败：{model_info['total_params']} != {EXPECTED_MODEL_PARAMS}")
    registry = ParameterRoleRegistry.from_model(model)
    optimizer = build_optimizer(args, registry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT_DIR / f"checkpoint_{args.tag}.pt"
    start_epoch = 1
    best = {"score": -math.inf, "epoch": None, "model": None}
    history: list[dict[str, object]] = []
    since_improve = 0
    if args.resume and checkpoint_path.exists():
        start_epoch, best, history, since_improve = load_checkpoint(
            checkpoint_path, model, optimizer, train_source, args)
        print(f"从第 {start_epoch} 轮恢复训练：{checkpoint_path}")

    previous_assignments = None
    if history:
        previous_assignments = router_statistics(model)["assignments"]
    eval_valid = LimitedSource(valid_source, args.max_eval_batches) \
        if args.max_eval_batches else valid_source
    for epoch in range(start_epoch, args.max_epochs + 1):
        train_record = train_one_epoch(
            model, train_source, registry, optimizer, args.max_batches)
        valid = evaluate_all(model, eval_valid)
        routing = router_statistics(model, previous_assignments)
        previous_assignments = routing["assignments"]
        record = {
            "epoch": epoch,
            "train": train_record,
            "valid_macro": valid["macro"],
            "valid_pooled": valid["pooled"],
            "valid_per_scenario": valid["per_scenario"],
            "routing": routing["report"],
        }
        history.append(record)
        print(
            f"第 {epoch} 轮：验证集八场景等权指标={valid['macro']:.6f}，"
            f"全体样本合并指标={valid['pooled']:.6f}，"
            f"专家覆盖={routing['report']['expert_coverage']}，"
            f"耗时={train_record['wall_clock_sec']:.1f} 秒")
        if valid["macro"] > best["score"]:
            best = {
                "score": valid["macro"],
                "epoch": epoch,
                "model": cpu_state_dict(model),
            }
            since_improve = 0
        else:
            since_improve += 1
        save_checkpoint(
            checkpoint_path, model, optimizer, train_source, epoch,
            best, history, since_improve, args)
        if since_improve >= args.patience:
            print(f"连续 {args.patience} 轮没有改善，停止训练")
            break

    if best["model"] is None:
        raise RuntimeError("no valid checkpoint was selected")
    model.load_state_dict(best["model"])
    selected_routing = router_statistics(model)["report"]
    output: dict[str, object] = {
        "status": "完成",
        "development_only": bool(args.development),
        "best_valid_macro": best["score"],
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "history": history,
        "selected_routing": selected_routing,
        "model": model_info,
        "role_parameter_counts": registry.role_counts(),
        "provenance": {
            "site": "B",
            "script": "experiments/main_task_role_optimizer.py",
            "device": args.device,
            "seed": args.seed,
            "optimizer_mode": args.optimizer_mode,
            "expert_learning_rate": args.expert_lr,
            "router_learning_rate_ratio": args.router_lr_ratio,
            "shared_learning_rate_ratio": args.shared_lr_ratio,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "max_batches": args.max_batches,
            "max_eval_batches": args.max_eval_batches,
            "sample_counts": counts,
            "macro_scenarios": list(MACRO_SCENARIOS),
            "primary_endpoint": "测试集八场景分别计算曲线下面积后等权平均",
            "model_selection": "验证集同一指标精确最高轮次",
            "test_set_used": not args.development,
            "parameter_write_semantics": "每个混合批次每个参数最多写入一次",
            "preregistration": (
                "docs/20260817-1821-按参数角色隔离子任务优化器预注册-siteB.md"),
        },
    }
    if test_source is not None:
        eval_test = LimitedSource(test_source, args.max_eval_batches) \
            if args.max_eval_batches else test_source
        output["test"] = evaluate_all(model, eval_test)
    output["checkpoint_path"] = str(checkpoint_path)
    return output


def parse_args() -> argparse.Namespace:
    frozen = FrozenTrainingConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("device", nargs="?", default="cuda:0")
    parser.add_argument(
        "--optimizer-mode", required=True,
        choices=("shared_adamw", "task_state_uniform", "role_isolated"))
    parser.add_argument("--expert-lr", type=float, default=frozen.expert_learning_rate)
    parser.add_argument(
        "--router-lr-ratio", type=float,
        default=frozen.router_learning_rate_ratio)
    parser.add_argument(
        "--shared-lr-ratio", type=float,
        default=frozen.shared_learning_rate_ratio)
    parser.add_argument("--weight-decay", type=float, default=frozen.weight_decay)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--batch-size", type=int, default=frozen.batch_size)
    parser.add_argument("--max-epochs", type=int, default=frozen.max_epochs)
    parser.add_argument("--patience", type=int, default=frozen.patience)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.development and (args.max_batches or args.max_eval_batches):
        raise SystemExit("正式训练禁止截断训练集、验证集或测试集")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUT_DIR / f"run_{args.tag}.json"
    if destination.exists():
        print(f"结果已存在，安全跳过：{destination}")
        return
    print(
        "开始训练："
        f"优化器配置={args.optimizer_mode}，随机种子={args.seed}，"
        f"显卡={args.device}，专家基础学习率={args.expert_lr}，"
        f"路由器相对学习率={args.router_lr_ratio}，"
        f"共享主干相对学习率={args.shared_lr_ratio}")
    output = run(args)
    checkpoint_path = Path(output.pop("checkpoint_path"))
    atomic_json_dump(output, destination)
    checkpoint_path.unlink(missing_ok=True)
    print(f"训练完成，证据写入：{destination}")


if __name__ == "__main__":
    main()
