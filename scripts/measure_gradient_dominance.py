#!/usr/bin/env python
"""Measure per-group RMS gradient dominance for DCNv2 / MoE V1 / MoE V2.

回答两个问题（对应 D7 §1.3 修订口径）:

  Q1 「某组参数是否更新不足?」
     ——用 **每参数 RMS 梯度** 而非组范数占比:
            RMS_g = ||∇_θ L||_2^(g) / sqrt(N_g)
        组范数占比 (sqrt(v_g / Σv)) 会被参数量主导 (Sparse 占 99.2% 参数),
        无法回答"更新强度"。RMS 做了参数量归一化，可横向比较。

  Q2 「MoE 是 shared 均衡 还是 routed 特殊化?」
     ——把梯度按 (layer, expert, scenario) 分解:
        对每层、每个 scenario 做列归一化 (去掉 scenario 样本量/损失尺度的干扰),
        得到 share[e|s]；若接近 1/K 则为均衡化，若尖锐则为特殊化。

只做 forward + backward，**不** optimizer.step()，因此不会改动 checkpoint。

Usage:
    # 训练后的 MoE V2 (稀疏模式 top_k=2)
    python scripts/measure_gradient_dominance.py --device cuda:0 --arch v2 \
        --ckpt cache/dcnv2_moe_v2_k4.pt --top-k 2 --tag v2

    # MoE V1
    python scripts/measure_gradient_dominance.py --device cuda:0 --arch v1 \
        --ckpt cache/dcnv2_moe_k4.pt --tag v1

    # 初始化时刻 (从 vanilla split 而来, 尚未训练)
    python scripts/measure_gradient_dominance.py --device cuda:0 --arch v2 \
        --from-vanilla cache/dcnv2_vanilla.pt --top-k 2 --tag v2_init

Output: cache/grad_dominance_<tag>.json
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fields  # noqa: E402
from dataset import Dataset, Split  # noqa: E402
from model import (  # noqa: E402
    DCNv2,
    DCNv2MoE,
    DCNv2MoE_V2,
    param_group_of,
)

SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]


# ------------------------------------------------------------------
#  Parameter name → (layer index, unit) for the MoE cross layers
# ------------------------------------------------------------------
def parse_cross_unit(name: str):
    """'cross_layers.0.experts.2.weight' -> (0, 'E2');
       'cross_layers.1.shared.bias'      -> (1, 'shared');
       'cross_layers.2.w_gate.weight'    -> (2, 'router');
       'layers.1.weight' (vanilla)       -> (1, 'cross');
       otherwise None."""
    parts = name.split(".")
    if parts[0] == "cross_layers" and len(parts) >= 3 and parts[1].isdigit():
        li = int(parts[1])
        if parts[2] == "experts" and parts[3].isdigit():
            return li, f"E{int(parts[3])}"
        if parts[2] == "shared":
            return li, "shared"
        if parts[2] in ("w_gate", "w_noise", "router"):
            return li, "router"
        return li, parts[2]
    if parts[0] == "layers" and parts[1].isdigit() and int(parts[1]) < 3:
        return int(parts[1]), "cross"
    return None


# ------------------------------------------------------------------
#  Model construction
# ------------------------------------------------------------------
def build_model(args):
    if args.arch == "vanilla":
        model = DCNv2()
    elif args.arch == "v1":
        model = DCNv2MoE(dim=args.dim, K=args.K, routing=args.routing or "scenario")
    elif args.arch == "v2":
        model = DCNv2MoE_V2(
            dim=args.dim, K=args.K,
            top_k=args.top_k if args.top_k is not None else args.K,
            routing=args.routing or "data",
            noise_scale=args.noise_scale, lb_alpha=args.lb_alpha,
        )
    else:
        raise ValueError(args.arch)

    model = model.to(args.device)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state)
        print(f"  loaded checkpoint: {args.ckpt}")
    elif args.from_vanilla:
        vanilla = DCNv2().to(args.device)
        vanilla.load_state_dict(torch.load(args.from_vanilla, map_location=args.device))
        if args.arch == "vanilla":
            model = vanilla
        else:
            model.load_pretrained(vanilla)
            del vanilla
        print(f"  initialized from vanilla: {args.from_vanilla}")
    else:
        raise SystemExit("Need --ckpt or --from-vanilla")

    if args.arch == "v2" and args.top_k is not None:
        model.set_top_k(args.top_k)
        print(f"  top_k set to {args.top_k}")
    return model


# ------------------------------------------------------------------
#  Core measurement
# ------------------------------------------------------------------
def measure_rms_dominance(model, train_set, args):
    """Return dict with group RMS + per-(layer, expert, scenario) decomposition."""
    device = torch.device(args.device)
    criterion = torch.nn.BCEWithLogitsLoss()

    named = list(model.named_parameters())
    group_of = {n: param_group_of(n) for n, _ in named}
    unit_of = {n: parse_cross_unit(n) for n, _ in named}

    # --- static parameter counts ---
    group_n = defaultdict(int)
    unit_n = defaultdict(int)
    for n, p in named:
        group_n[group_of[n]] += p.numel()
        if unit_of[n] is not None:
            unit_n[unit_of[n]] += p.numel()

    # --- accumulators (sum of squared grad norms over sub-batches) ---
    group_sq = defaultdict(float)          # group -> Σ ||g||²
    unit_sq = defaultdict(float)           # (li, unit) -> Σ ||g||²
    unit_scn_sq = defaultdict(float)       # (li, unit, s) -> Σ ||g||²
    scn_subcnt = defaultdict(int)          # s -> number of sub-batches
    sparse_active_sq, sparse_active_n = 0.0, 0
    n_subbatches = 0
    scenario_rows = defaultdict(int)

    # 数据按时间排序，前若干 batch 的 scenario 覆盖不全；shuffle 保证
    # per-(expert, scenario) 分解拿到全部 8 个 scenario。torch.manual_seed
    # 已固定，因此依旧可复现。
    loader = torch.utils.data.DataLoader(
        Dataset(train_set), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=args.shuffle,
    )

    model.train(args.train_mode)
    pbar = tqdm(loader, total=args.n_batches, desc="grad-dominance")
    for bi, batch in enumerate(pbar):
        if bi >= args.n_batches:
            break
        for field in fields.all:
            batch[field] = batch[field].to(device).int()
        tab_batch = batch["tab"]

        for s in tab_batch.unique():
            si = int(s.item())
            mask = tab_batch == s
            if int(mask.sum()) < args.min_rows:
                continue
            sub = {k: v[mask] for k, v in batch.items()}
            scenario_rows[si] += int(mask.sum())

            model.zero_grad(set_to_none=True)
            model(sub)
            loss = criterion(sub["logit"], sub["is_click"].float())
            lb = sub.get("_load_balance_loss")
            if lb is not None and args.include_lb_loss:
                loss = loss + lb
            loss.backward()

            for n, p in named:
                if p.grad is None:
                    continue
                sq = float(p.grad.detach().pow(2).sum())
                group_sq[group_of[n]] += sq
                u = unit_of[n]
                if u is not None:
                    unit_sq[u] += sq
                    unit_scn_sq[(u[0], u[1], si)] += sq
                if args.sparse_active and group_of[n] == "Sparse" and p.grad.dim() == 2:
                    rows = int((p.grad.detach().abs().sum(1) > 0).sum())
                    sparse_active_sq += sq
                    sparse_active_n += rows * p.grad.shape[1]

            scn_subcnt[si] += 1
            n_subbatches += 1

        model.zero_grad(set_to_none=True)

    if n_subbatches == 0:
        raise RuntimeError("No sub-batch was measured; check data / min_rows.")

    # --- group-level statistics ---
    total_sq = sum(group_sq.values())
    nonsparse_sq = total_sq - group_sq.get("Sparse", 0.0)
    groups = {}
    for g, n_par in group_n.items():
        mean_sq = group_sq.get(g, 0.0) / n_subbatches
        groups[g] = {
            "n_params": n_par,
            "param_share_all": n_par / sum(group_n.values()),
            "param_share_nonsparse": (
                n_par / (sum(group_n.values()) - group_n.get("Sparse", 0))
                if g != "Sparse" else None
            ),
            "mean_grad_sq": mean_sq,
            "rms": math.sqrt(mean_sq / n_par) if n_par else 0.0,
            # 辅助口径（不作判据）: 组梯度范数占比
            "norm_share_all": math.sqrt(group_sq.get(g, 0.0) / total_sq) if total_sq else 0.0,
            "norm_share_nonsparse": (
                math.sqrt(group_sq.get(g, 0.0) / nonsparse_sq)
                if (g != "Sparse" and nonsparse_sq > 0) else None
            ),
        }
    if args.sparse_active and sparse_active_n > 0 and "Sparse" in groups:
        groups["Sparse"]["rms_active_rows"] = math.sqrt(
            (sparse_active_sq / n_subbatches) / (sparse_active_n / n_subbatches)
        )
        groups["Sparse"]["mean_active_elems"] = sparse_active_n / n_subbatches

    # --- per (layer, unit) ---
    per_unit = {}
    for (li, u), n_par in sorted(unit_n.items()):
        mean_sq = unit_sq.get((li, u), 0.0) / n_subbatches
        per_unit[f"L{li}/{u}"] = {
            "n_params": n_par,
            "rms": math.sqrt(mean_sq / n_par) if n_par else 0.0,
        }

    # --- per (layer, unit, scenario) ---
    per_unit_scn = {}
    for (li, u, si), sq in sorted(unit_scn_sq.items()):
        n_par = unit_n[(li, u)]
        n_sub = max(scn_subcnt[si], 1)
        mean_sq = sq / n_sub
        per_unit_scn[f"L{li}/{u}/S{si}"] = {
            "rms": math.sqrt(mean_sq / n_par) if n_par else 0.0,
            "n_subbatches": n_sub,
        }

    # --- specialization: column-normalized share over routed experts ---
    specialization = {}
    layers = sorted({li for (li, u) in unit_n if u.startswith("E")})
    expert_ids = sorted({u for (li, u) in unit_n if u.startswith("E")})
    for li in layers:
        mat, scns = {}, []
        for si in SCENARIOS:
            col = []
            for e in expert_ids:
                key = f"L{li}/{e}/S{si}"
                col.append(per_unit_scn[key]["rms"] if key in per_unit_scn else 0.0)
            tot = sum(col)
            if tot <= 0:
                continue
            scns.append(si)
            mat[f"S{si}"] = [c / tot for c in col]
        if not scns:
            continue
        K = len(expert_ids)
        uniform = 1.0 / K
        max_shares = [max(v) for v in mat.values()]
        entropies = []
        for v in mat.values():
            h = -sum(p * math.log(p + 1e-12) for p in v) / math.log(K)
            entropies.append(h)
        specialization[f"layer_{li}"] = {
            "experts": expert_ids,
            "scenario_normalized_share": mat,     # share[e | s], 行=scenario
            "uniform_baseline": uniform,
            "max_share_mean": sum(max_shares) / len(max_shares),
            "normalized_entropy_mean": sum(entropies) / len(entropies),
            # 0 = 完全均衡, 1 = 完全特殊化
            "specialization_index": (
                (sum(max_shares) / len(max_shares) - uniform) / (1 - uniform)
            ),
        }

    # --- verdicts (D7 §1.3.2 判据) ---
    def rms(g):
        return groups.get(g, {}).get("rms", 0.0)

    expert_rms = rms("CrossExpert") or rms("CrossVanilla")
    verdict = {
        "expert_vs_dnn": (expert_rms / rms("DNN")) if rms("DNN") else None,
        "router_vs_expert": (rms("Router") / expert_rms) if expert_rms else None,
        "shared_vs_routed": (
            rms("CrossShared") / rms("CrossExpert") if rms("CrossExpert") else None
        ),
        "specialization_index_mean": (
            sum(v["specialization_index"] for v in specialization.values())
            / len(specialization) if specialization else None
        ),
    }
    verdict["readings"] = []
    if verdict["expert_vs_dnn"] is not None:
        r = verdict["expert_vs_dnn"]
        verdict["readings"].append(
            f"RMS(Expert)/RMS(DNN)={r:.3f} → "
            + ("专家更新不足 (<1/3)，建议提高 expert LR 或降 beta2"
               if r < 1 / 3 else
               "专家更新强度与 DNN 相当或更强，健康")
        )
    if verdict["router_vs_expert"] is not None:
        r = verdict["router_vs_expert"]
        verdict["readings"].append(
            f"RMS(Router)/RMS(Expert)={r:.3f} → "
            + ("路由几乎不学，MoE 退化为均匀路由" if r < 0.1
               else "路由在有效学习")
        )
    if verdict["shared_vs_routed"] is not None:
        r = verdict["shared_vs_routed"]
        verdict["readings"].append(
            f"RMS(Shared)/RMS(Routed)={r:.3f} → "
            + ("共性知识由 shared 承担，routed 未分化" if r > 2
               else "shared 与 routed 更新强度相当")
        )
    if verdict["specialization_index_mean"] is not None:
        r = verdict["specialization_index_mean"]
        verdict["readings"].append(
            f"specialization_index={r:.3f} (0=均衡,1=特殊化) → "
            + ("routed 专家特殊化" if r > 0.3
               else "routed 专家均衡化 (无明显分工)")
        )

    return {
        "groups": groups,
        "per_unit": per_unit,
        "per_unit_scenario": per_unit_scn,
        "specialization": specialization,
        "verdict": verdict,
        "n_subbatches": n_subbatches,
        "scenario_rows": dict(scenario_rows),
    }


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arch", choices=["vanilla", "v1", "v2"], required=True)
    ap.add_argument("--ckpt", default=None, help="state_dict of the target arch")
    ap.add_argument("--from-vanilla", default=None,
                    help="build MoE by splitting this vanilla DCNv2 checkpoint")
    ap.add_argument("--tag", required=True, help="output file suffix")
    ap.add_argument("--dim", type=int, default=360)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=None, help="V2 active routed experts")
    ap.add_argument("--routing", default=None, choices=["scenario", "data"])
    ap.add_argument("--noise-scale", type=float, default=0.1)
    ap.add_argument("--lb-alpha", type=float, default=0.01)
    ap.add_argument("--include-lb-loss", action="store_true", default=True)
    ap.add_argument("--no-lb-loss", dest="include_lb_loss", action="store_false")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--min-rows", type=int, default=32,
                    help="skip scenario sub-batches smaller than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--train-mode", action="store_true", default=True,
                    help="model.train(True): gate noise on, matches training")
    ap.add_argument("--eval-mode", dest="train_mode", action="store_false")
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="shuffle batches so all 8 scenarios are covered")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.add_argument("--sparse-active", action="store_true", default=True,
                    help="also report Sparse RMS over touched embedding rows")
    ap.add_argument("--out-dir", default="cache")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 66)
    print(f"Gradient dominance (RMS-per-param)  arch={args.arch}  tag={args.tag}")
    print("=" * 66)

    model = build_model(args)
    split_idx = {"train": 0, "valid": 1, "test": 2}[args.split]
    data = Split("all")[split_idx]
    print(f"  split={args.split}  rows={len(data):,}  "
          f"n_batches={args.n_batches}  batch_size={args.batch_size}")

    result = measure_rms_dominance(model, data, args)
    result["meta"] = {
        "arch": args.arch, "tag": args.tag, "ckpt": args.ckpt,
        "from_vanilla": args.from_vanilla, "K": args.K, "top_k": args.top_k,
        "routing": args.routing, "split": args.split, "seed": args.seed,
        "n_batches": args.n_batches, "batch_size": args.batch_size,
        "train_mode": args.train_mode, "include_lb_loss": args.include_lb_loss,
        "device": args.device,
    }

    # ---- report ----
    print()
    print(f"{'group':<14}{'n_params':>12}{'RMS grad':>14}"
          f"{'norm share':>12}{'param share':>12}")
    print("-" * 66)
    for g, v in sorted(result["groups"].items(), key=lambda kv: -kv[1]["rms"]):
        print(f"{g:<14}{v['n_params']:>12,}{v['rms']:>14.3e}"
              f"{v['norm_share_all'] * 100:>11.2f}%{v['param_share_all'] * 100:>11.3f}%")
    if "rms_active_rows" in result["groups"].get("Sparse", {}):
        print(f"{'Sparse(active)':<14}{'':>12}"
              f"{result['groups']['Sparse']['rms_active_rows']:>14.3e}")

    print()
    print("Verdict:")
    for line in result["verdict"]["readings"]:
        print(f"  - {line}")

    if result["specialization"]:
        print()
        print("Scenario→expert share (column-normalized, uniform="
              f"{1 / args.K:.3f}):")
        for lname, sp in result["specialization"].items():
            print(f"  {lname}  spec_index={sp['specialization_index']:+.3f}  "
                  f"H={sp['normalized_entropy_mean']:.3f}")
            for s, row in sp["scenario_normalized_share"].items():
                print(f"    {s:<4}" + "".join(f"{p:8.3f}" for p in row))

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"grad_dominance_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
