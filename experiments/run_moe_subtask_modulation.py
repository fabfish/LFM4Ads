import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""子任务 路由网络/路由专家/共享专家 多维交叉促进-抑制 调制实验。

加载对应的「从零预训练 backbone」，在按场景切分的子批量上做梯度平方累计
（AU）驱动的逐目标（路由网络 / 路由专家 / 共享专家）梯度调制，观察
促进 / 抑制对上游点击率预估任务的影响。

三个调制目标是**相乘的独立维度**，各自可取 none / encourage / suppress：
  - 全路由混合专家（DCNv2MoE，无共享专家）：
        router(3) × expert(3) = 9 配置
  - 部分路由加共享专家（DCNv2MoE_V2，含共享专家）：
        router(3) × expert(3) × shared(3) = 27 配置

产物的命名以三目标模式标识，便于做交叉分析。

用法：
  python run_moe_subtask_modulation.py --model fully-routed \
      --router-mode suppress --expert-mode none --shared-mode none \
      --seed 42 --device cuda:0
  python run_moe_subtask_modulation.py --model partial-shared \
      --router-mode suppress --expert-mode encourage --shared-mode suppress \
      --seed 42 --device cuda:1
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
    ap = argparse.ArgumentParser(
        description="子任务 路由网络/路由专家/共享专家 交叉促进-抑制 调制")
    ap.add_argument("--model", required=True,
                    choices=["fully-routed", "partial-shared"],
                    help="fully-routed=DCNv2MoE; partial-shared=DCNv2MoE_V2")
    ap.add_argument("--router-mode", default="none",
                    choices=["none", "encourage", "suppress"])
    ap.add_argument("--expert-mode", default="none",
                    choices=["none", "encourage", "suppress"])
    ap.add_argument("--shared-mode", default="none",
                    choices=["none", "encourage", "suppress"])
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
    ap.add_argument("--save-backbone", action="store_true",
                    help="落盘调制后的 backbone 至 "
                         "cache/subtask_backbone_*.pt，供表征用法阶段加载")
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

    # 全路由模型无共享专家：强制 shared-mode=none 并提示
    if args.model == "fully-routed" and args.shared_mode != "none":
        print("[warn] fully-routed 无共享专家，忽略 --shared-mode，置为 none")
        args.shared_mode = "none"

    print(f"[config] model={args.model} router={args.router_mode} "
          f"expert={args.expert_mode} shared={args.shared_mode} "
          f"device={args.device} seed={args.seed} K={args.K} "
          f"routing={args.routing} lr={args.lr} alpha={args.alpha} "
          f"epochs={args.epochs} batch={args.batch_size}")

    model = cls(dim=360, K=args.K, routing=args.routing).to(args.device)
    if args.model == "partial-shared":
        model.top_k = args.K  # 全软路由
    sd = torch.load(ckpt, map_location=args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt}  missing={len(missing)} unexpected={len(unexpected)}")

    opt = AdaTaskOptimizer(
        model, lr=args.lr,
        router_mode=args.router_mode, expert_mode=args.expert_mode,
        shared_mode=args.shared_mode, alpha=args.alpha, beta=args.beta,
        weight_decay=0.01)
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
            w.writerow(["model", "router_mode", "expert_mode", "shared_mode",
                        "seed", "alpha", "mean_per_scenario", "test_auc_all"] +
                       [f"s{s}_auc" for s in SCENARIOS])
        w.writerow([args.model, args.router_mode, args.expert_mode,
                    args.shared_mode, args.seed, args.alpha, f"{mean_sc:.4f}",
                    f"{pooled:.4f}"] +
                   [f"{per[s]:.4f}" for s in SCENARIOS])

    # 写 JSON
    # alpha=0.5 为默认口径，文件名不带 alpha 后缀以保持既有产物命名兼容；
    # 非默认 alpha 追加 _a{alpha} 后缀，避免敏感性扫描覆盖默认口径结果。
    alpha_sfx = "" if abs(args.alpha - 0.5) < 1e-9 else f"_a{args.alpha:g}"
    json_path = (f"{CACHE_DIR}/subtask_modulation_{args.model}_"
                 f"r{args.router_mode}_e{args.expert_mode}_"
                 f"s{args.shared_mode}_seed{args.seed}{alpha_sfx}.json")
    summary = {
        "config": {
            "model": args.model,
            "router_mode": args.router_mode,
            "expert_mode": args.expert_mode,
            "shared_mode": args.shared_mode,
            "seed": args.seed, "K": args.K, "routing": args.routing,
            "lr": args.lr, "alpha": args.alpha,
            "epochs": args.epochs, "batch_size": args.batch_size,
        },
        "test_auc_all": pooled,
        "mean_per_scenario_auc": mean_sc,
        "per_scenario_auc": {str(s): per[s] for s in SCENARIOS},
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {json_path}")

    # 可选落盘调制后的 backbone，供表征用法（特征级/模块级/模型级）阶段加载。
    # 默认不存，避免 36×3 网格产生上百个 320MB 权重文件；
    # 仅对通过多种子验证的少数配置按需开启。
    if args.save_backbone:
        pt_path = json_path.replace(".json", ".pt").replace(
            "subtask_modulation_", "subtask_backbone_")
        torch.save(model.state_dict(), pt_path)
        print(f"  backbone → {pt_path}")


if __name__ == "__main__":
    main()
