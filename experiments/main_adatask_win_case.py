"""E17 入口：s0(one-vs-rest) 三臂 S/T/A 全数据训练（预注册 20260818-1815）。

用法（每个 run 一个臂）：
  LFM_DATASET=dataset_27k.feather LFM_VOCAB_JSON=cache/fields_27k.json \
  LFM_SAMPLE_COUNTS_JSON=cache/sample_counts_27k.json \
  python experiments/main_adatask_win_case.py --device cuda:0 --arm S --seed 101

产物：{out_dir}/run_e17_{arm}_s{seed}.json（成功后删除 checkpoint）。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from adatask_win_case import (  # noqa: E402
    ADATASK_ALPHA,
    ADATASK_BETA,
    ADATASK_CLIP,
    NUM_GROUPS,
    TwoGroupOptimizer,
    build_shared_optimizer,
    collect_group_gradients,
)
from dataset import GpuBatches, Split  # noqa: E402
from main_macro_auc import (  # noqa: E402
    EXPECTED_COUNTS,
    MACRO_SCENARIOS,
    build,
    evaluate_all,
)
from main_task_role_optimizer import router_statistics  # noqa: E402
from task_role_optimizer import ParameterRoleRegistry  # noqa: E402
from task_role_optimizer_protocol import EXPECTED_MODEL_PARAMS  # noqa: E402

PREREG_DOC = "docs/20260818-1815-结论复审与E17单场景AdaTask-win-case预注册.md"


def atomic_json_dump(payload, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    tmp.replace(path)


def run(args) -> None:
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f"run_e17_{args.arm}_s{args.seed}.json"
    ckpt_path = out_dir / f"ckpt_e17_{args.arm}_s{args.seed}.pt"

    train_set, valid_set, test_set = Split("all")
    counts = {"train": len(train_set), "valid": len(valid_set),
              "test": len(test_set)}
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"数据切分行数失败：{counts} != {EXPECTED_COUNTS}")

    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    model, model_info = build("moe", 5, True, device, top_k=2)
    if model_info["total_params"] != EXPECTED_MODEL_PARAMS:
        raise SystemExit(
            f"参数量失败：{model_info['total_params']} != {EXPECTED_MODEL_PARAMS}")
    if int(model.top_k) != 2:
        raise SystemExit(f"top_k 必须为 2，实际 {model.top_k}")
    registry = ParameterRoleRegistry.from_model(model)

    if args.arm == "S":
        optimizer = build_shared_optimizer(
            model, args.lr, (0.9, 0.999), 1e-8, args.weight_decay)
        optimizer_state = None
    else:
        optimizer = TwoGroupOptimizer(
            model, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=args.weight_decay, mode=args.arm,
            alpha=args.adatask_alpha, au_beta=args.adatask_beta,
            f_clip=tuple(args.adatask_clip))

    train_source = GpuBatches(
        train_set, args.batch_size, device, shuffle=True, seed=args.seed)
    valid_source = GpuBatches(
        valid_set, args.batch_size, device, shuffle=False)
    test_source = GpuBatches(
        test_set, args.batch_size, device, shuffle=False)

    start_epoch = 0
    best = {"score": -1.0, "epoch": None, "state": None}
    history: list[dict] = []
    batches_per_epoch = len(train_source)
    skip_batches_total = 0
    no_improve = 0
    rng_states = None
    if ckpt_path.exists() and args.resume:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["opt"])
        gen_state = payload["generator"]
        if gen_state.device.type != "cpu":
            gen_state = gen_state.cpu()
        train_source._gen.set_state(gen_state)
        torch.set_rng_state(payload["rng"].cpu())
        if device.startswith("cuda") and "cuda_rng" in payload:
            torch.cuda.set_rng_state_all(
                [s.cpu() for s in payload["cuda_rng"]])
        start_epoch = payload["epoch"] + 1
        best = payload["best"]
        history = payload["history"]
        skip_batches_total = payload["skip_batches_total"]
        no_improve = payload["no_improve"]
        print(f"[resume] 从 epoch {start_epoch} 恢复")
    elif ckpt_path.exists():
        ckpt_path.unlink()

    batches_this_run = 0
    run_start = time.time()
    done = False
    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        epoch_skips = 0
        routing_prev = None
        for batch in train_source:
            batches_this_run += 1
            groups = collect_group_gradients(
                model, batch, registry, target_tab=args.target_scenario)
            if len(groups) < NUM_GROUPS:
                epoch_skips += 1
                continue
            if args.arm == "S":
                optimizer.step(groups)
                loss = sum(g.loss for g in groups) / len(groups)
            else:
                stats = optimizer.step(groups)
                loss = stats["loss"]
            epoch_loss_sum += loss
            epoch_loss_n += 1
            if args.max_batches and batches_this_run >= args.max_batches:
                break
        skip_batches_total += epoch_skips
        if isinstance(optimizer, TwoGroupOptimizer):
            adatask_stats = optimizer.epoch_adatask_stats()
            optimizer.reset_epoch_stats()
        else:
            adatask_stats = None

        valid_metrics = evaluate_all(model, valid_source)
        routing_prev = router_statistics(model, routing_prev)
        routing_report = routing_prev["report"]
        score = float(valid_metrics["macro"])
        entry = {
            "epoch": epoch,
            "train_loss": (epoch_loss_sum / max(epoch_loss_n, 1)),
            "skipped_batches": epoch_skips,
            "valid": valid_metrics,
            "routing": routing_report,
            "seconds": round(time.time() - epoch_start, 1),
        }
        if adatask_stats is not None:
            entry["adatask"] = adatask_stats
        history.append(entry)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": copy.deepcopy(
                        {k: v.detach().cpu()
                         for k, v in model.state_dict().items()})}
            no_improve = 0
        else:
            no_improve += 1

        print(f"[{args.arm} s{args.seed}] epoch {epoch} "
              f"valid_macro={score:.6f} loss={entry['train_loss']:.6f} "
              f"skip={epoch_skips} "
              f"({entry['seconds']}s, best={best['score']:.6f}@{best['epoch']})",
              flush=True)

        if args.max_batches:  # smoke 模式不做 checkpoint/early-stop 语义
            done = True
            break
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": (optimizer.optimizer.state_dict()
                    if args.arm == "S" else optimizer.state_dict()),
            "generator": train_source._gen.get_state(),
            "rng": torch.get_rng_state(),
            "best": best,
            "history": history,
            "skip_batches_total": skip_batches_total,
            "no_improve": no_improve,
        }
        if device.startswith("cuda"):
            ckpt["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(ckpt, ckpt_path)
        if no_improve >= args.patience:
            done = True
            break
    else:
        done = True

    if best["state"] is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best["state"].items()})
    test_metrics = evaluate_all(model, test_source)
    final_routing = router_statistics(model)["report"]

    clip_rates = [h.get("adatask", {}).get("clip_rate", 0.0)
                  for h in history if h.get("adatask")]
    payload = {
        "status": "completed" if done else "interrupted",
        "arm": args.arm,
        "prereg": PREREG_DOC,
        "best_valid_macro": best["score"],
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "history": history,
        "test": test_metrics,
        "test_s0": (test_metrics["per_scenario"].get(str(args.target_scenario))
                    if test_metrics.get("per_scenario") else None),
        "valid_s0_at_best": (
            history[best["epoch"]]["valid"]["per_scenario"][
                str(args.target_scenario)]
            if best["epoch"] is not None and history else None),
        "clip_dominated": bool(clip_rates and max(clip_rates) >= 0.3),
        "data": {
            "counts": counts,
            "counts_check": "pass",
            "skip_batches_total": skip_batches_total,
            "batches_per_epoch": batches_per_epoch,
        },
        "model": model_info,
        "macro_scenarios": list(MACRO_SCENARIOS),
        "target_scenario": args.target_scenario,
        "optimizer": {
            "mode": args.arm,
            "lr": args.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "groups": "one_vs_rest",
            "adatask_alpha": args.adatask_alpha,
            "adatask_beta": args.adatask_beta,
            "adatask_clip": list(args.adatask_clip),
        },
        "selection": {
            "endpoint": "valid macro argmax",
            "early_stop": f"patience={args.patience}",
            "max_epochs": args.max_epochs,
        },
        "routing_final": final_routing,
        "provenance": {
            "site": os.environ.get("LFM_SITE", "A"),
            "seed": args.seed,
            "device": device,
            "seconds_total": round(time.time() - run_start, 1),
            "resumed": start_epoch > 0,
        },
    }
    atomic_json_dump(payload, run_path)
    if not args.max_batches and ckpt_path.exists():
        ckpt_path.unlink()
    print(f"[{args.arm} s{args.seed}] 完成 → {run_path} "
          f"best_valid_macro={best['score']:.6f}@{best['epoch']} "
          f"test_s0={payload['test_s0']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", required=True, choices=["S", "T", "A"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-scenario", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--adatask-alpha", type=float, default=ADATASK_ALPHA)
    parser.add_argument("--adatask-beta", type=float, default=ADATASK_BETA)
    parser.add_argument("--adatask-clip", type=float,
                        nargs=2, default=list(ADATASK_CLIP))
    parser.add_argument("--max-batches", type=int, default=0,
                        help=">0 时为 smoke 模式：只跑指定 batch 数，产物写审计目录")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir",
                        default=os.environ.get(
                            "LFM_E17_OUT",
                            "cache/adatask_win_s0_27k_siteA"))
    args = parser.parse_args()
    if args.max_batches:
        args.out_dir = "cache/audit/adatask_win_s0_27k_siteA"
    print(f"E17 {args.arm} 臂 seed={args.seed} device={args.device} "
          f"target=s{args.target_scenario} out={args.out_dir}", flush=True)
    run(args)


if __name__ == "__main__":
    main()
