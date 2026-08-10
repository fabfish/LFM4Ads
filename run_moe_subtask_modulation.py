"""子任务 路由网络/专家 促进-抑制 调制实验（核心 10 配置广扫）。

加载对应的「从零预训练 backbone」，在按场景切分的子批量上做梯度平方累计
（AU）驱动的逐目标（路由网络 / 专家）梯度调制，观察促进 / 抑制对上游
点击率预估任务的影响。

统一旋钮：
  --target  router | expert       调制目标（路由网络 / 专家）
  --direction 0 | 1 | -1          不调制(=基线) / 促进 / 抑制

核心 10 种配置 = 2 模型 ×（2 调制目标 × {促进, 抑制}）+ 2 基线（direction=0）。

产物：
  result_moe_subtask_modulation.csv   每行一个配置 × 8 场景 AUC + mean
  cache/subtask_modulation_{model}_{target}_{direction}_seed{seed}.json

用法：
  python run_moe_subtask_modulation.py --model fully-routed --target expert --direction 1 --seed 42 --device cuda:0
  python run_moe_subtask_modulation.py --model partial-shared --target router --direction -1 --seed 42 --device cuda:1
"""

import argparse
import csv
import json
import os
import sys

import pandas as pd
import torch
from torcheval.metrics import BinaryAUROC

from dataset import Dataset, Split
from model import AdaTaskOptimizer, DCNv2MoE, DCNv2MoE_V2

SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
CACHE_DIR = "cache"
RESULT_DIR = "."


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="子任务 路由网络/专家 促进-抑制 调制")
    ap.add_argument("--model", required=True,
                    choices=["fully-routed", "partial-shared"],
                    help="fully-routed=DCNv2MoE; partial-shared=DCNv2MoE_V2")
    ap.add_argument("--target", required=True, choices=["router", "expert"])
    ap.add_argument("--direction", required=True, type=int, choices=[0, 1, -1],
                    help="0=不调制(基线) / 1=促进 / -1=抑制")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--routing", default="data", choices=["data", "scenario"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.99)
    ap.add_argument("--epochs", type=int, default=1,
                    help="在 (训练+验证) 数据上做多少轮按场景调制训练")
    ap.add_argument("--batch-size", type=int, default=16384)
    return ap.parse_args(argv)


def model_spec(args):
    if args.model == "fully-routed":
        return (DCNv2MoE, f"{CACHE_DIR}/moe_fully_routed_seed{args.seed}.pt",
                "moe_fully_routed")
    return (DCNv2MoE_V2, f"{CACHE_DIR}/moe_partial_shared_seed{args.seed}.pt",
            "moe_partial_shared")


def evaluate_per_scenario_auc(model, device):
    model.eval()
    per = {}
    for s in SCENARIOS:
        _, _, test = Split(s)
        met = BinaryAUROC().to(device)
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(
                Dataset(test), batch_size=32768, shuffle=False,
                num_workers=4, pin_memory=True
            ):
                batch = {k: v.to(device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                model(batch)
                met.update(batch["logit"], batch["is_click"].float())
        per[s] = float(met.compute().item())
    return per


def evaluate_pooled_auc(model, device):
    model.eval()
    met = BinaryAUROC().to(device)
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            Dataset(Split("all")[2]), batch_size=32768, shuffle=False,
            num_workers=4, pin_memory=True
        ):
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            model(batch)
            met.update(batch["logit"], batch["is_click"].float())
    return float(met.compute().item())


def main():
    args = _parse_args(sys.argv[1:])
    torch.manual_seed(args.seed)
    cls, ckpt, tag = model_spec(args)

    print(f"[config] model={args.model} target={args.target} "
          f"direction={args.direction} device={args.device} seed={args.seed} "
          f"K={args.K} routing={args.routing} lr={args.lr} alpha={args.alpha} "
          f"epochs={args.epochs} batch={args.batch_size}")

    model = cls(dim=360, K=args.K, routing=args.routing).to(args.device)
    if args.model == "partial-shared":
        model.top_k = args.K  # 全软路由
    sd = torch.load(ckpt, map_location=args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt}  missing={len(missing)} unexpected={len(unexpected)}")

    opt = AdaTaskOptimizer(model, lr=args.lr, target=args.target,
                           direction=args.direction, alpha=args.alpha,
                           beta=args.beta, weight_decay=0.01)
    opt.register_hooks()

    train_set, valid_set = Split("all")[:2]
    all_data = pd.concat([train_set, valid_set])

    model.train()
    total_batches = 0
    for epoch in range(args.epochs):
        dl = torch.utils.data.DataLoader(
            Dataset(all_data), batch_size=args.batch_size, shuffle=True,
            num_workers=8, pin_memory=True)
        for batch in dl:
            tab = batch["tab"].to(args.device)
            for s_idx in tab.unique().tolist():
                mask = tab == s_idx
                if mask.sum() < 4:
                    continue
                mask_cpu = mask.cpu()
                sub = {k: v[mask_cpu].to(args.device)
                       for k, v in batch.items()
                       if isinstance(v, torch.Tensor)}

                opt.set_scenario(int(s_idx))
                model(sub)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    sub["logit"], sub["is_click"].float())
                loss.backward()
                opt.modulate_and_zero_grad()

            total_batches += 1
            if total_batches % 50 == 0:
                print(f"  [epoch {epoch}] batch {total_batches}, "
                      f"AU entries={len(opt.AU)}, loss={loss.item():.4f}")

    # 评估
    per = evaluate_per_scenario_auc(model, args.device)
    pooled = evaluate_pooled_auc(model, args.device)
    mean_sc = sum(per.values()) / len(per)
    for s in SCENARIOS:
        print(f"    scenario {s}: {per[s]:.4f}")
    print(f"  mean per-scenario AUC: {mean_sc:.4f}")
    print(f"  pooled test AUC: {pooled:.4f}")

    opt.remove_hooks()

    # 写 CSV
    csv_path = f"{RESULT_DIR}/result_moe_subtask_modulation.csv"
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if os.path.getsize(csv_path) == 0:
            w.writerow(["model", "target", "direction", "seed",
                        "mean_per_scenario", "test_auc_all"] +
                       [f"s{s}_auc" for s in SCENARIOS])
        w.writerow([args.model, args.target, args.direction, args.seed,
                    f"{mean_sc:.4f}", f"{pooled:.4f}"] +
                   [f"{per[s]:.4f}" for s in SCENARIOS])

    # 写 JSON
    json_path = (f"{CACHE_DIR}/subtask_modulation_{args.model}_{args.target}_"
                 f"{args.direction}_seed{args.seed}.json")
    summary = {
        "config": {
            "model": args.model, "target": args.target,
            "direction": args.direction, "seed": args.seed, "K": args.K,
            "routing": args.routing, "lr": args.lr, "alpha": args.alpha,
            "epochs": args.epochs, "batch_size": args.batch_size,
        },
        "test_auc_all": pooled,
        "mean_per_scenario_auc": mean_sc,
        "per_scenario_auc": {str(s): per[s] for s in SCENARIOS},
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {json_path}")


if __name__ == "__main__":
    main()
