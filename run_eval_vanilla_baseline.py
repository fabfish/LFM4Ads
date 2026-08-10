"""复评现有 vanilla DCNv2 基准基线（不重新训练）。

用于「混合专家是否改进上游」的对照。直接加载仓库已有的
cache/dcnv2_vanilla.pt（若存在），以与从零预训练 MoE 完全相同的方式
（按场景 + pooled 测试集）评估，得到可比的 AUC。

若给定 checkpoint 与当前 DCNv2 架构不兼容（键不匹配），脚本会报错退出，
此时需向用户确认是否改用 cache/vanilla_pretrain.pt 或重新评估。

产物：
  result_vanilla_baseline_eval.csv
  cache/vanilla_baseline_eval_summary.json
"""

import argparse
import csv
import json
import os
import sys

import torch

from dataset import Split
from model import DCNv2
from train import evaluate

SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
CACHE_DIR = "cache"
RESULT_DIR = "."


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="复评现有 vanilla DCNv2 基线")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=f"{CACHE_DIR}/dcnv2_vanilla.pt",
                    help="现有 vanilla 权重路径")
    return ap.parse_args(argv)


def main():
    args = _parse_args(sys.argv[1:])
    model = DCNv2().to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {args.ckpt}")
    print(f"  missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
    if missing:
        print("  MISSING:", missing[:10])
    if unexpected:
        print("  UNEXPECTED:", unexpected[:10])

    per = {}
    for s in SCENARIOS:
        _, _, test_set = Split(s)
        per[s] = float(evaluate(model, test_set))
    pooled = float(evaluate(model, Split("all")[2]))
    mean_sc = sum(per.values()) / len(per)

    print("  per-scenario AUC:")
    for s in SCENARIOS:
        print(f"    scenario {s}: {per[s]:.4f}")
    print(f"  mean per-scenario AUC: {mean_sc:.4f}")
    print(f"  pooled test AUC: {pooled:.4f}")

    csv_path = f"{RESULT_DIR}/result_vanilla_baseline_eval.csv"
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if os.path.getsize(csv_path) == 0:
            w.writerow(["model", "ckpt", "test_auc_all", "mean_per_scenario"] +
                       [f"s{s}_auc" for s in SCENARIOS])
        w.writerow(["vanilla-dcnv2", args.ckpt, f"{pooled:.4f}",
                    f"{mean_sc:.4f}"] + [f"{per[s]:.4f}" for s in SCENARIOS])

    summary = {
        "model": "vanilla-dcnv2",
        "ckpt": args.ckpt,
        "test_auc_all": pooled,
        "mean_per_scenario_auc": mean_sc,
        "per_scenario_auc": {str(s): per[s] for s in SCENARIOS},
    }
    json_path = f"{CACHE_DIR}/vanilla_baseline_eval_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {json_path}")


if __name__ == "__main__":
    main()
