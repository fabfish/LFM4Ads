#!/usr/bin/env python3
"""按冻结顺序调度优化器状态和学习率筛选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "experiments/main_task_role_optimizer.py"
OUT = ROOT / "cache/task_role_optimizer_27k_siteB"
LOG = ROOT / "logs/task_role_optimizer_27k_siteB"
AUDIT = OUT / "invariant_audit.json"


@dataclass(frozen=True)
class Task:
    tag: str
    device: str
    optimizer_mode: str
    seed: int
    expert_lr: float
    router_ratio: float
    shared_ratio: float
    development: bool


def development_tasks(stage: str, expert_lr: float) -> list[Task]:
    if stage == "state-screen":
        return [
            Task("状态筛选_共享状态_s202", "cuda:1", "shared_adamw", 202,
                 5e-4, 1.0, 1.0, True),
            Task("状态筛选_场景状态同学习率_s202", "cuda:1", "task_state_uniform", 202,
                 5e-4, 1.0, 1.0, True),
            Task("状态筛选_完整角色隔离_s202", "cuda:1", "role_isolated", 202,
                 5e-4, 0.05, 1.0, True),
        ]
    if stage == "expert-lr-screen":
        return [
            Task(f"专家学习率_{lr:g}_s202", "cuda:1", "role_isolated", 202,
                 lr, 0.05, 1.0, True)
            for lr in (2e-4, 5e-4, 1e-3)
        ]
    if stage == "router-lr-screen":
        return [
            Task(f"路由器比例_{ratio:g}_s202", "cuda:1", "role_isolated", 202,
                 expert_lr, ratio, 1.0, True)
            for ratio in (0.0, 0.02, 0.05, 0.1)
        ]
    if stage == "shared-lr-screen":
        raise ValueError("shared-lr-screen requires --router-ratio and is built in main")
    raise ValueError(stage)


def formal_tasks(config_path: Path) -> list[Task]:
    with config_path.open(encoding="utf-8") as handle:
        configs = json.load(handle)
    if not isinstance(configs, list) or len(configs) != 2:
        raise ValueError("正式确认配置必须恰好包含两个候选")
    devices = {42: "cuda:0", 123: "cuda:1", 456: "cuda:2", 789: "cuda:0"}
    tasks = []
    for candidate_index, config in enumerate(configs, 1):
        for seed in (42, 123, 456, 789):
            tasks.append(Task(
                tag=f"正式候选{candidate_index}_s{seed}",
                device=devices[seed],
                optimizer_mode=config["optimizer_mode"],
                seed=seed,
                expert_lr=float(config["expert_lr"]),
                router_ratio=float(config["router_lr_ratio"]),
                shared_ratio=float(config["shared_lr_ratio"]),
                development=False,
            ))
    return tasks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_audit_authorizes_current_code() -> None:
    if not AUDIT.exists():
        raise SystemExit(f"缺少不变量审计授权：{AUDIT}")
    with AUDIT.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if not audit.get("long_run_authorized"):
        raise SystemExit("十三项不变量未全部通过，禁止启动训练")
    expected = audit.get("implementation", {}).get("sha256", {})
    current = {
        "task_role_optimizer.py": file_sha256(ROOT / "task_role_optimizer.py"),
        "task_role_optimizer_protocol.py": file_sha256(
            ROOT / "task_role_optimizer_protocol.py"),
        "scripts/verify/verify_task_role_optimizer.py": file_sha256(
            ROOT / "scripts/verify/verify_task_role_optimizer.py"),
        "experiments/main_task_role_optimizer.py": file_sha256(
            ROOT / "experiments/main_task_role_optimizer.py"),
        "scripts/matrix/run_task_role_optimizer_matrix.py": file_sha256(
            ROOT / "scripts/matrix/run_task_role_optimizer_matrix.py"),
        "scripts/summarize/summarize_task_role_optimizer.py": file_sha256(
            ROOT / "scripts/summarize/summarize_task_role_optimizer.py"),
    }
    if expected != current:
        raise SystemExit("优化器或验证代码已变化，旧不变量授权失效，请重新审计")


def task_fingerprint(task: Task, args: argparse.Namespace) -> str:
    payload = {
        "device": task.device,
        "optimizer_mode": task.optimizer_mode,
        "seed": task.seed,
        "expert_lr": task.expert_lr,
        "router_ratio": task.router_ratio,
        "shared_ratio": task.shared_ratio,
        "development": task.development,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "max_batches": args.max_batches,
        "max_eval_batches": args.max_eval_batches,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def resolved_tag(task: Task, args: argparse.Namespace) -> str:
    return f"{task.tag}_{task_fingerprint(task, args)}"


def command(task: Task, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(ENTRY), task.device,
        "--optimizer-mode", task.optimizer_mode,
        "--seed", str(task.seed),
        "--expert-lr", str(task.expert_lr),
        "--router-lr-ratio", str(task.router_ratio),
        "--shared-lr-ratio", str(task.shared_ratio),
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--tag", resolved_tag(task, args),
    ]
    if task.development:
        cmd.append("--development")
    if args.max_batches:
        cmd.extend(["--max-batches", str(args.max_batches)])
    if args.max_eval_batches:
        cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])
    return cmd


def run_task(task: Task, args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    tag = resolved_tag(task, args)
    result = OUT / f"run_{tag}.json"
    if result.exists():
        with result.open(encoding="utf-8") as handle:
            provenance = json.load(handle).get("provenance", {})
        expected = {
            "device": task.device,
            "seed": task.seed,
            "optimizer_mode": task.optimizer_mode,
            "expert_learning_rate": task.expert_lr,
            "router_learning_rate_ratio": task.router_ratio,
            "shared_learning_rate_ratio": task.shared_ratio,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "max_batches": args.max_batches,
            "max_eval_batches": args.max_eval_batches,
            "test_set_used": not task.development,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"同名结果来源不一致，拒绝复用：{result}")
        print(f"已完成且来源一致，跳过：{tag}")
        return
    environment = os.environ.copy()
    environment.update({
        "LFM_SITE": "B",
        "LFM_DATASET": str(ROOT / "dataset_27k.feather"),
        "LFM_VOCAB_JSON": str(ROOT / "cache/fields_27k.json"),
        "LFM_SAMPLE_COUNTS_JSON": str(ROOT / "cache/sample_counts_27k.json"),
        "LFM_TASK_ROLE_OUT": str(OUT),
    })
    log_path = LOG / f"{tag}.log"
    print(
        f"启动：{tag}；随机种子={task.seed}；显卡={task.device}；"
        f"专家学习率={task.expert_lr}；路由器比例={task.router_ratio}；"
        f"共享主干比例={task.shared_ratio}")
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command(task, args), cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"训练失败：{tag}，退出码={completed.returncode}，日志={log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(
        "state-screen", "expert-lr-screen", "router-lr-screen",
        "shared-lr-screen", "formal"))
    parser.add_argument("--expert-lr", type=float, default=5e-4)
    parser.add_argument("--router-ratio", type=float, default=0.05)
    parser.add_argument("--formal-config", type=Path)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stage == "formal" and (args.max_batches or args.max_eval_batches):
        parser.error("正式训练禁止截断训练集、验证集或测试集")
    if not args.dry_run:
        assert_audit_authorizes_current_code()

    if args.stage == "formal":
        if args.formal_config is None:
            parser.error("formal requires --formal-config")
        tasks = formal_tasks(args.formal_config)
    elif args.stage == "shared-lr-screen":
        tasks = [
            Task(f"共享主干比例_{ratio:g}_s202", "cuda:1", "role_isolated", 202,
                 args.expert_lr, args.router_ratio, ratio, True)
            for ratio in (0.5, 1.0)
        ]
    else:
        tasks = development_tasks(args.stage, args.expert_lr)

    print(f"本阶段共 {len(tasks)} 次训练：")
    for task in tasks:
        print("  " + " ".join(command(task, args)))
    if args.dry_run:
        return
    for task in tasks:
        run_task(task, args)


if __name__ == "__main__":
    main()
