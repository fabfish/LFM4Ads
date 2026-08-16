import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from model import DCNv2, DCNv2MoE, DCNv2MoE_V2, DCNv2MoE_LowRank
from train import evaluate, infer, train, train_moe, scenario_loss

SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
CACHE_DIR = "cache"
RESULT_DIR = "."


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="从零预训练混合专家 backbone 与基准模型")
    ap.add_argument("--model", required=True,
                    choices=["fully-routed", "partial-shared", "vanilla",
                             "lowrank-full-dim", "same-flops-dense"],
                    help="fully-routed=DCNv2MoE 全路由; "
                         "partial-shared=DCNv2MoE_V2 部分路由加共享专家; "
                         "vanilla=DCNv2 基准稠密; "
                         "lowrank-full-dim=StageB 低秩全维专家 MoE; "
                         "same-flops-dense=StageB 普通 DCNv2(同 FLOPs 对照)")
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
    ap.add_argument("--router", default=None, choices=["frozen", "soft", "none"],
                    help="Stage B 路由模式: frozen=冻结路由器(均匀门控); "
                         "soft=可学习 DataRouter; none=稠密(无路由)")
    ap.add_argument("--rank", type=int, default=None,
                    help="Stage B lowrank-full-dim 的低秩专家秩 r; "
                         "默认 dim//(2K) 以与 dense 同 FLOPs")
    ap.add_argument("--vanilla-per-scenario", action="store_true",
                    help="仅对 --model vanilla 生效：改用与混合专家相同的"
                         "按场景子批量训练过程，消除训练过程混淆")
    ap.add_argument("--scenario-loss-weighting", default="equal",
                    choices=["equal", "sample"],
                    help="equal=每场景等权(历史过程); "
                         "sample=按样本占比加权(理论等价整批, 见驱动文档§二/§六)")
    ap.add_argument("--run-code", default=None,
                    help="实验代号(见驱动文档§三), 用于产物命名与追溯; "
                         "不填则回退到 model_tag 兼容命名以避免覆盖既有产物")
    return ap.parse_args(argv)


def model_tag(args):
    base = {
        "fully-routed": "moe_fully_routed",
        "partial-shared": "moe_partial_shared",
        "vanilla": "vanilla_from_scratch",
        "lowrank-full-dim": "moe_lowrank_fulldim",
        "same-flops-dense": "moe_sameflops_dense",
    }[args.model]
    # 冻结路由器是独立变体，产物需与正常训练区分，避免覆盖。
    if getattr(args, "freeze_router", False) or getattr(args, "router", None) == "frozen":
        base += "_frozenrouter"
    # 按场景训练的稠密基准是独立变体，同样需区分。
    if args.model == "vanilla" and getattr(args, "vanilla_per_scenario", False):
        base += "_perscenario"
    # 低秩专家秩 r（Stage B）作为独立变体后缀，避免覆盖不同秩的产物。
    if args.model == "lowrank-full-dim":
        r = args.rank if args.rank is not None else 360 // (2 * args.K)
        base += f"_r{r}"
    # K=4 为默认口径，不加后缀以兼容既有产物名；非默认 K 追加 _K{K}，
    # 否则专家数扫描会覆盖 K=4 的结果。vanilla 无专家概念，不加。
    if args.model not in ("vanilla", "same-flops-dense") and args.K != 4:
        base += f"_K{args.K}"
    return base


def freeze_routers(model):
    """把路由器参数冻结在零初值 → 门控恒为均匀（softmax(0)=1/K）、零随机噪声。

    判别性实验：若「强制均匀门控」的上游 AUC 不低于「抑制路由网络」的最优值，
    则说明抑制路由的收益来自「减少按场景路由」而非「更好的分工」，
    亦即在此数据尺度上路由机制本身是负担。

    两类模型的路由器：
      - V1 为 `layer.router`（DataRouter/ScenarioRouter，确定性 softmax，本无噪声），
      - V2 为 `layer.w_gate` / `layer.w_noise` + 噪声门控（CrossExpertLayerV2）。
    两者均为零初始化；V2 额外关闭 `_router_noise_enabled`，保证冻结即严格
    「零噪声、恒均匀门控」（此前漏关，冻结后仍注入 F.softplus(0)+1e-2 量级的噪声）。
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
        # 关闭 V2 的随机门控噪声，保证冻结即严格「零噪声、恒均匀」
        if hasattr(layer, "_router_noise_enabled"):
            layer._router_noise_enabled = False
    print(f"[freeze-router] 冻结 {n} 个路由参数于零初值（门控恒均匀、零噪声）")
    return model


def build_model(args):
    if args.model == "fully-routed":
        model = DCNv2MoE(dim=360, K=args.K, routing=args.routing).to(args.device)
    elif args.model == "partial-shared":
        # top_k=K → 全软路由（共享专家常驻 + 路由专家加权，负载均衡损失恒为 0）
        model = DCNv2MoE_V2(dim=360, K=args.K, top_k=args.K,
                            routing=args.routing).to(args.device)
    elif args.model == "lowrank-full-dim":
        # Stage B：低秩全维专家 MoE。r 默认 dim//(2K) 以与 dense 同 FLOPs。
        rank = args.rank if args.rank is not None else 360 // (2 * args.K)
        model = DCNv2MoE_LowRank(dim=360, K=args.K, r=rank,
                                 routing="data").to(args.device)
    elif args.model == "same-flops-dense":
        # Stage B：同 FLOPs 稠密对照 = 普通 DCNv2（无路由）。
        return DCNv2().to(args.device)
    else:
        return DCNv2().to(args.device)
    # --router frozen（或旧 --freeze-router）都在此统一冻结路由器。
    if getattr(args, "freeze_router", False) or getattr(args, "router", None) == "frozen":
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


def collect_gate_stats(model, device):
    """汇聚各交叉层门控的均值/熵，刻画路由结构（Stage B 指标）。

    - frozen 低秩模型：门控恒为均匀 → 熵 = ln K（最大熵，无结构）。
    - soft 低秩模型：门控随样本变化 → 熵 < ln K 且均值偏离均匀，体现结构。
    - 稠密（无路由）模型：无 set_top_k → 返回 None。
    仅用于事后分析，不影响训练/评估指标。
    """
    if not hasattr(model, "set_top_k"):
        return None
    model.eval()
    K = getattr(model, "K", 4)
    per_layer = [[] for _ in range(len(model.cross_layers))]
    loader = torch.utils.data.DataLoader(
        Dataset(Split("all")[2]), batch_size=10000,
        num_workers=args_num_workers(), pin_memory=True)
    import math
    with torch.inference_mode():
        for batch in loader:
            for f in fields.all:
                batch[f] = batch[f].to(device).int()
            model(batch)
            for li, g in enumerate(model.last_gates):
                per_layer[li].append(g.detach().cpu())
    stats = {}
    for li in range(len(per_layer)):
        g = torch.cat(per_layer[li], 0)  # [N, K]
        mean = g.mean(0).tolist()
        ent = float(-(g * (g.clamp_min(1e-8)).log()).sum(-1).mean().item())
        stats[f"layer{li}"] = {
            "mean_gate": [round(float(x), 4) for x in mean],
            "entropy": round(ent, 4),
            "max_entropy": round(math.log(K), 4),
        }
    return stats


def args_num_workers():
    # collect_gate_stats 在 main 内调用，复用全局 NUM_WORKERS 默认值。
    return int(os.environ.get("LFM_NUM_WORKERS", "10"))


def train_partial_shared_from_scratch(args, model, loss_weighting="equal"):
    """DCNv2MoE_V2 从零训练：全软路由，不做 warmup / 稀疏 / 负载均衡切换。

    复用 main_moe_v2 的按场景前向-反向骨架，但不调用 load_pretrained，
    且 top_k 全程 = K（full-soft，lb_loss 恒为 0）。
    `loss_weighting` 仅缩放基础 BCE（与 train_moe 一致的语义），
    负载均衡损失是独立项，不随样本占比缩放。
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
                loss = scenario_loss(
                    criterion, sub["logit"], sub["is_click"].float(),
                    mask, tab_batch, loss_weighting,
                )
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

    # 产物命名：提供 --run-code（驱动文档§三）时以其为唯一前缀，避免覆盖既有产物；
    # 否则回退到 model_tag 兼容命名（历史脚本/手动运行不传 --run-code 时使用）。
    if args.run_code:
        tag = args.run_code
        ckpt_path = f"{CACHE_DIR}/{tag}.pt"
        csv_path = f"{RESULT_DIR}/result_{tag}.csv"
        json_path = f"{CACHE_DIR}/moe_pretrain_summary_{tag}.json"
    else:
        tag = model_tag(args)
        ckpt_path = f"{CACHE_DIR}/{tag}_seed{args.seed}.pt"
        csv_path = f"{RESULT_DIR}/result_{tag}.csv"
        json_path = f"{CACHE_DIR}/moe_pretrain_summary_{tag}_seed{args.seed}.json"

    print(f"[config] model={args.model} device={args.device} seed={args.seed} "
          f"K={args.K} routing={args.routing} lr={args.lr} "
          f"beta2={args.beta2} shuffle={args.shuffle} "
          f"scenario_loss_weighting={args.scenario_loss_weighting} "
          f"run_code={args.run_code} max_epochs={args.max_epochs}")

    model = build_model(args)

    # Stage B 新变体（lowrank-full-dim / same-flops-dense）与 fully-routed /
    # vanilla_per_scenario 共用 train_moe 的「按场景子批量前向-反向」骨架，
    # 以 sample 加权保证与稠密基准可比（pooled=full-batch 等价）。
    # train_moe 不依赖任何混合专家专属属性，故可直接用于稠密/低秩模型。
    if (args.model in ("fully-routed", "lowrank-full-dim", "same-flops-dense")
            or args.vanilla_per_scenario):
        test_auc = train_moe(model, "all", lr=args.lr, beta2=args.beta2,
                             shuffle=args.shuffle,
                             loss_weighting=args.scenario_loss_weighting)
    elif args.model == "partial-shared":
        history = train_partial_shared_from_scratch(
            args, model, loss_weighting=args.scenario_loss_weighting)
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

    # 路由结构（Stage B 指标）：soft 低秩应呈现熵<lnK 的非均匀门控。
    gate_stats = collect_gate_stats(model, args.device)
    if gate_stats:
        print(f"  gate structure: {gate_stats}")

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
    router = getattr(args, "router", None)
    router_semantics = (
        "frozen: router weights zeroed (uniform, zero-noise)"
        if (getattr(args, "freeze_router", False) or router == "frozen")
        else ("soft: learnable DataRouter (data routing)"
              if router == "soft"
              else "none: dense, no router")
    )
    rank = args.rank if args.rank is not None else (
        360 // (2 * args.K) if args.model == "lowrank-full-dim" else None)
    summary = {
        "config": {
            "model": args.model, "device": args.device, "seed": args.seed,
            "routing": args.routing, "K": args.K, "lr": args.lr,
            "beta2": args.beta2, "shuffle": args.shuffle,
            "max_epochs": args.max_epochs,
            "freeze_router": bool(getattr(args, "freeze_router", False)),
            "router": router,
            "router_semantics": router_semantics,
            "rank": rank,
            "vanilla_per_scenario": bool(
                getattr(args, "vanilla_per_scenario", False)),
            "scenario_loss_weighting": args.scenario_loss_weighting,
            "run_code": args.run_code,
        },
        "test_auc_all": pooled,
        "mean_per_scenario_auc": mean_sc,
        "per_scenario_auc": {str(s): per_scenario[s] for s in SCENARIOS},
        "gate_structure": gate_stats,
        "checkpoint": ckpt_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {json_path}")


if __name__ == "__main__":
    main()
