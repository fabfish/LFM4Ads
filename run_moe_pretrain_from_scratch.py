"""从零预训练混合专家 backbone 与基准稠密模型（不加载任何预训练权重）。

三类 backbone（各随机初始化）：
  - 全路由混合专家模型   (model.py 中 DCNv2MoE 类, K=4, 数据级路由, top_k=K 全软)
  - 部分路由加共享专家模型 (model.py 中 DCNv2MoE_V2 类, K=4, 1 共享+4 路由, top_k=K 全软)
  - 基准稠密模型         (model.py 中 DCNv2 类)

全部从零训练，用于「混合专家结构是否改进上游点击率预估」的公平对照。

产物：
  cache/moe_fully_routed_seed{seed}.pt
  cache/moe_partial_shared_seed{seed}.pt
  cache/vanilla_from_scratch_seed{seed}.pt
  result_moe_fully_routed.csv / result_moe_partial_shared.csv / result_moe_vanilla.csv
  cache/moe_pretrain_summary_{model}_seed{seed}.json

用法：
  python run_moe_pretrain_from_scratch.py --model fully-routed --device cuda:0 --seed 42
  python run_moe_pretrain_from_scratch.py --model partial-shared --device cuda:1 --seed 42
  python run_moe_pretrain_from_scratch.py --model vanilla --device cuda:0 --seed 42
"""

import argparse
import csv
import json
import os
import sys
from copy import deepcopy

import torch
from tqdm import tqdm

import fields
from dataset import Dataset, Split
from model import DCNv2, DCNv2MoE, DCNv2MoE_V2
from train import evaluate, infer, train, train_moe

SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
CACHE_DIR = "cache"
RESULT_DIR = "."


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="从零预训练混合专家 backbone 与基准模型")
    ap.add_argument("--model", required=True,
                    choices=["fully-routed", "partial-shared", "vanilla"],
                    help="fully-routed=DCNv2MoE 全路由; "
                         "partial-shared=DCNv2MoE_V2 部分路由加共享专家; "
                         "vanilla=DCNv2 基准稠密")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=4, help="路由专家数量")
    ap.add_argument("--routing", default="data", choices=["data", "scenario"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=300,
                    help="安全上限；实际由验证 AUC 早停（δ=0.001）控制")
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="默认开启数据洗牌，使 --seed 生效")
    ap.add_argument("--freeze-router", action="store_true",
                    help="把路由器冻结在零初值，门控恒为均匀（判别性对照："
                         "检验路由机制本身是否为负担）")
    return ap.parse_args(argv)


def model_tag(args):
    base = {
        "fully-routed": "moe_fully_routed",
        "partial-shared": "moe_partial_shared",
        "vanilla": "vanilla_from_scratch",
    }[args.model]
    # 冻结路由器是独立变体，产物需与正常训练区分，避免覆盖。
    if getattr(args, "freeze_router", False):
        base += "_frozenrouter"
    # K=4 为默认口径，不加后缀以兼容既有产物名；非默认 K 追加 _K{K}，
    # 否则专家数扫描会覆盖 K=4 的结果。vanilla 无专家概念，不加。
    if args.model != "vanilla" and args.K != 4:
        base += f"_K{args.K}"
    return base


def freeze_routers(model):
    """把路由器参数冻结在零初值 → 门控恒为均匀（softmax(0)=1/K）。

    判别性实验：若「强制均匀门控」的上游 AUC 不低于「抑制路由网络」的最优值，
    则说明抑制路由的收益来自「减少按场景路由」而非「更好的分工」，
    亦即在此数据尺度上路由机制本身是负担。

    两类模型的路由器：V1 为 `layer.router`（DataRouter/ScenarioRouter），
    V2 为 `layer.w_gate` / `layer.w_noise`；两者均为零初始化。
    """
    n = 0
    for layer in model.cross_layers:
        mods = []
        if hasattr(layer, "router"):
            mods.append(layer.router)
        if hasattr(layer, "w_gate"):
            mods.append(layer.w_gate)
        if hasattr(layer, "w_noise"):
            mods.append(layer.w_noise)
        for mod in mods:
            for p in mod.parameters():
                torch.nn.init.zeros_(p)
                p.requires_grad_(False)
                n += p.numel()
    print(f"[freeze-router] 冻结 {n} 个路由参数于零初值（门控恒均匀）")
    return model


def build_model(args):
    if args.model == "fully-routed":
        model = DCNv2MoE(dim=360, K=args.K, routing=args.routing).to(args.device)
    elif args.model == "partial-shared":
        # top_k=K → 全软路由（共享专家常驻 + 路由专家加权，负载均衡损失恒为 0）
        model = DCNv2MoE_V2(dim=360, K=args.K, top_k=args.K,
                            routing=args.routing).to(args.device)
    else:
        return DCNv2().to(args.device)
    if getattr(args, "freeze_router", False):
        freeze_routers(model)
    return model


def evaluate_all_scenarios(model, device):
    """返回 {scenario: auc} 与 pooled test AUC（按场景独立评估）。"""
    per = {}
    for s in SCENARIOS:
        _, _, test_set = Split(s)
        per[s] = float(evaluate(model, test_set))
    # pooled = 在全部场景的测试集上整体评估
    pooled = float(evaluate(model, Split("all")[2]))
    return per, pooled


def train_partial_shared_from_scratch(args, model):
    """DCNv2MoE_V2 从零训练：全软路由，不做 warmup / 稀疏 / 负载均衡切换。

    复用 main_moe_v2 的按场景前向-反向骨架，但不调用 load_pretrained，
    且 top_k 全程 = K（full-soft，lb_loss 恒为 0）。
    """
    device = torch.device(args.device)
    train_set, valid_set, test_set = Split("all")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, args.beta2))
    auc_best = 0.0
    best_state = None
    epoch = 0
    history = []

    while True:
        epoch += 1
        loader = torch.utils.data.DataLoader(
            Dataset(train_set), batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=args.shuffle)
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            model.train()
            for field_name in fields.all:
                batch[field_name] = batch[field_name].to(device).int()
            tab_batch = batch["tab"]
            for s in tab_batch.unique():
                mask = tab_batch == s
                sub = {k: v[mask] for k, v in batch.items()}
                model(sub)
                loss = criterion(sub["logit"], sub["is_click"].float())
                loss = loss + sub.get("_load_balance_loss",
                                      torch.tensor(0.0, device=device))
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        auc = float(evaluate(model, valid_set))
        history.append({"epoch": epoch, "valid_auc": auc})
        print(f"  Epoch {epoch} valid AUC: {auc:.4f}")
        if auc_best < auc - 0.001:
            auc_best = auc
            best_state = deepcopy(model.state_dict())
        else:
            break
        if epoch >= args.max_epochs:
            print(f"  [max-epochs] stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best state (valid AUC={auc_best:.4f})")
    return history


def main():
    args = _parse_args(sys.argv[1:])
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tag = model_tag(args)
    ckpt_path = f"{CACHE_DIR}/{tag}_seed{args.seed}.pt"
    csv_path = f"{RESULT_DIR}/result_{tag}.csv"
    json_path = f"{CACHE_DIR}/moe_pretrain_summary_{tag}_seed{args.seed}.json"

    print(f"[config] model={args.model} device={args.device} seed={args.seed} "
          f"K={args.K} routing={args.routing} lr={args.lr} "
          f"beta2={args.beta2} shuffle={args.shuffle} max_epochs={args.max_epochs}")

    model = build_model(args)

    if args.model == "fully-routed":
        test_auc = train_moe(model, "all", lr=args.lr, beta2=args.beta2,
                             shuffle=args.shuffle)
    elif args.model == "partial-shared":
        history = train_partial_shared_from_scratch(args, model)
        test_auc = float(evaluate(model, Split("all")[2]))
    else:  # vanilla
        test_auc = train(model, "all", shuffle=args.shuffle)

    print(f"  [{args.model}] pretrain test AUC (all): {test_auc:.4f}")

    # 保存权重
    torch.save(model.state_dict(), ckpt_path)
    print(f"  checkpoint → {ckpt_path}")

    # 按场景评估（用于上游改进判定）
    per_scenario, pooled = evaluate_all_scenarios(model, args.device)
    print("  per-scenario AUC:")
    for s in SCENARIOS:
        print(f"    scenario {s}: {per_scenario[s]:.4f}")
    mean_sc = sum(per_scenario.values()) / len(per_scenario)
    print(f"  mean per-scenario AUC: {mean_sc:.4f}")
    print(f"  pooled test AUC: {pooled:.4f}")

    # 写 CSV（带测量口径说明）
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        # 首次写入表头
        if os.path.getsize(csv_path) == 0:
            w.writerow(["model", "seed", "routing", "K",
                        "test_auc_all", "mean_per_scenario"] +
                       [f"s{s}_auc" for s in SCENARIOS])
        w.writerow([args.model, args.seed, args.routing, args.K,
                    f"{pooled:.4f}", f"{mean_sc:.4f}"] +
                   [f"{per_scenario[s]:.4f}" for s in SCENARIOS])

    # 写 JSON 汇总
    summary = {
        "config": {
            "model": args.model, "device": args.device, "seed": args.seed,
            "routing": args.routing, "K": args.K, "lr": args.lr,
            "beta2": args.beta2, "shuffle": args.shuffle,
            "max_epochs": args.max_epochs,
            "freeze_router": bool(getattr(args, "freeze_router", False)),
        },
        "test_auc_all": pooled,
        "mean_per_scenario_auc": mean_sc,
        "per_scenario_auc": {str(s): per_scenario[s] for s in SCENARIOS},
        "checkpoint": ckpt_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {json_path}")


if __name__ == "__main__":
    main()
