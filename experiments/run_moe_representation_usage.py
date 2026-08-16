import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python
"""特征级表征用法：把（调制后的）混合专家 backbone 当作 LFM4Ads 做下游特征增强。

背景与设计依据
--------------
`FeatureUsage`（model.py）依赖 backbone 的两种表征来源：
  - `LFM4Ads.CRs[user_id, k]`（形状 1000×6×360）——用于 "concat CR_k" / "gate CR_k"
  - `LFM4Ads.sparse(batch)[:, :270]`      ——用于 "SUM"
`DCNv2MoE` / `DCNv2MoE_V2` 的 forward 已在 `hasattr(self, "CRs")` 时聚合
6 层表征（x0 + 3 个交叉层 + 2 个 DNN 层），与稠密 `DCNv2` 的层数和维度一致，
因此**特征级用法可直接复用，无需新增模型类**。

（对比：模块级 `ModuleUsage` 依赖 `LFM4Ads.layers[i]` 是 `nn.Linear(360,360)`，
而混合专家的交叉层是 K 个 `Linear(dim, dim//K)` 的专家拆分，结构不兼容，
需另建 `ModuleUsageMoE`，不在本脚本范围内。）

用法
----
  # 用调制后的 backbone（需先以 --save-backbone 跑调制）
  python run_moe_representation_usage.py --model fully-routed \
      --router-mode suppress --expert-mode encourage --shared-mode none \
      --seed 42 --device cuda:0

  # 用未调制的原始 backbone 作对照
  python run_moe_representation_usage.py --model fully-routed \
      --backbone pretrain --seed 42 --device cuda:0

产物
----
  result_moe_representation_usage.csv  每 (配置, 场景, 融合方法) 一行 AUC
  cache/representation_usage_<tag>.json 汇总（按方法的场景均值）
"""
import argparse
import csv
import json
import os
import sys

import pandas as pd
import torch

from dataset import Split
from model import DCNv2MoE, DCNv2MoE_V2, FeatureUsage
from train import infer, train

CACHE_DIR = "cache"
RESULT_DIR = "."
SCENARIOS = [1, 0, 4, 2, 6, 3, 8, 5]
# 融合方法：gate=门控加权（360 维），concat=拼接（370 维）。
# CR_1/CR_3 分别代表「首个交叉层」与「末个交叉层」输出，是最能体现
# 专家分工是否被调制改变的两层；SUM 作为不依赖 CRs 的对照。
DEFAULT_METHODS = ["SUM", "gate CR_1", "gate CR_3", "concat CR_1", "concat CR_3"]


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="特征级表征用法（混合专家 backbone）")
    ap.add_argument("--model", required=True,
                    choices=["fully-routed", "partial-shared"])
    ap.add_argument("--backbone", default="modulated",
                    choices=["modulated", "pretrain"],
                    help="modulated=用调制后的 backbone；pretrain=用未调制原始 backbone")
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
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--scenarios", nargs="*", type=int, default=SCENARIOS)
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="下游训练是否洗牌（默认 True，与上游口径一致）")
    return ap.parse_args(argv)


def resolve_backbone(args):
    """返回 (模型类, 权重路径, 标签)。"""
    cls = DCNv2MoE if args.model == "fully-routed" else DCNv2MoE_V2
    if args.backbone == "pretrain":
        stem = ("moe_fully_routed" if args.model == "fully-routed"
                else "moe_partial_shared")
        return cls, f"{CACHE_DIR}/checkpoints/{stem}_seed{args.seed}.pt", \
            f"{args.model}_pretrain_seed{args.seed}"
    tag = (f"{args.model}_r{args.router_mode}_e{args.expert_mode}_"
           f"s{args.shared_mode}_seed{args.seed}")
    return cls, f"{CACHE_DIR}/checkpoints/subtask_backbone_{tag}.pt", tag


def build_backbone(args):
    cls, ckpt, tag = resolve_backbone(args)
    if not os.path.exists(ckpt):
        raise SystemExit(
            f"[fatal] backbone 不存在: {ckpt}\n"
            f"  若 backbone=modulated，请先带 --save-backbone 跑一次调制：\n"
            f"  python run_moe_subtask_modulation.py --model {args.model} "
            f"--router-mode {args.router_mode} --expert-mode {args.expert_mode} "
            f"--shared-mode {args.shared_mode} --seed {args.seed} "
            f"--device {args.device} --epochs 1 --save-backbone")
    kwargs = dict(dim=360, K=args.K, routing=args.routing)
    if cls is DCNv2MoE_V2:
        kwargs["top_k"] = args.K
    model = cls(**kwargs).to(args.device)
    sd = torch.load(ckpt, map_location=args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
    model.requires_grad_(False)
    return model, tag


def aggregate_crs(backbone, device):
    """在 训练+验证 集上聚合逐用户表征，写入 backbone.CRs（1000×6×360）。

    与 main.py 的做法一致：forward 内以 0.99 衰减累加，最后做 layer_norm。
    """
    backbone.CRs = torch.zeros(1000, 6, 360).to(device)
    train_valid = pd.concat(Split("all")[:2])
    for _ in infer(backbone, train_valid):
        pass
    backbone.CRs = torch.nn.functional.layer_norm(backbone.CRs, [360])
    print(f"[CRs] aggregated {tuple(backbone.CRs.shape)}")


def main():
    args = _parse_args(sys.argv[1:])
    torch.manual_seed(args.seed)
    backbone, tag = build_backbone(args)
    print(f"[config] {tag} device={args.device} methods={args.methods}")
    aggregate_crs(backbone, args.device)

    csv_path = f"{RESULT_DIR}/result_moe_representation_usage.csv"
    new_csv = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    results = {m: {} for m in args.methods}
    for scenario in args.scenarios:
        for method in args.methods:
            model = FeatureUsage(backbone, method).to(args.device)
            auc = train(model, scenario, shuffle=args.shuffle)
            results[method][scenario] = float(auc)
            print(f"  [{tag}] scenario={scenario} method={method!r} "
                  f"AUC={auc:.4f}")
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new_csv:
                    w.writerow(["tag", "model", "backbone", "router_mode",
                                "expert_mode", "shared_mode", "seed",
                                "scenario", "method", "auc"])
                    new_csv = False
                w.writerow([tag, args.model, args.backbone, args.router_mode,
                            args.expert_mode, args.shared_mode, args.seed,
                            scenario, method, f"{auc:.4f}"])

    summary = {
        "config": {
            "model": args.model, "backbone": args.backbone,
            "router_mode": args.router_mode, "expert_mode": args.expert_mode,
            "shared_mode": args.shared_mode, "seed": args.seed,
            "K": args.K, "routing": args.routing, "shuffle": args.shuffle,
        },
        "per_method_per_scenario_auc": {
            m: {str(s): v for s, v in d.items()} for m, d in results.items()},
        "per_method_mean_auc": {
            m: (sum(d.values()) / len(d) if d else None)
            for m, d in results.items()},
    }
    json_path = f"{CACHE_DIR}/archives/moe_exploration/representation_usage_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("[mean AUC by method] " + "  ".join(
        f"{m}={v:.4f}" for m, v in summary["per_method_mean_auc"].items()
        if v is not None))
    print(f"  summary → {json_path}")


if __name__ == "__main__":
    main()
