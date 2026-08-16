import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python
"""路由门控诊断：区分「分工更稳定」与「路由被冻结在近均匀」两种竞争解释。

动机
----
「抑制路由网络」是全部实验中唯一跨两类模型稳健成立的正向旋钮，
此前的机制解读是「路由权重更新放缓 → 专家分工不被逐场景噪声反复重塑」。
但该解读有一个**未排除的竞争解释**：

  解释 A（分工更稳定）：路由仍在学习，只是更新更平滑，最终形成清晰且稳定的分工。
  解释 B（路由被冻结）：抑制把梯度压得过小，门控几乎停留在初始化的近均匀状态，
                       收益其实来自「等效关闭了路由」，即混合专家结构本身是负担。

两者对下游含义完全相反：A 支持「调制改良了混合专家」，
B 则支持「混合专家在此尺度无用，最好的路由就是不路由」。

注意：不能用 `AdaTaskOptimizer.dominance_matrix` 来判定——它基于梯度平方累计（AU），
而调制正是在缩放梯度，用它验证调制效果属于循环论证。
本脚本改用**前向门控分布**（`batch["_gate"]`），与梯度无关。

判据
----
对每个场景 s，取该场景全部样本的平均门控向量 g_s ∈ R^K（和为 1）。
  - 均匀度：H(g_s) / log(K) ∈ [0,1]，越接近 1 越接近均匀（越"没在分工"）。
  - 分工度：各场景门控向量之间的平均 L1 距离，越大表示场景间分化越明显。
若「抑制」相对「不调制」的均匀度更高且分工度更低 → 支持解释 B。
若均匀度相近但分工度不低 → 支持解释 A。

用法
----
  python run_gate_diagnostics.py --model fully-routed --seed 42 --device cuda:0
（会自动诊断 cache/ 下所有已落盘的 subtask_backbone_<model>_*_seed<seed>.pt，
  以及未调制的原始 backbone 作为参照。）
"""
import argparse
import glob
import json
import os
import sys

import torch

from dataset import Dataset, Split
from model import DCNv2MoE, DCNv2MoE_V2

CACHE_DIR = "cache"
SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="路由门控分布诊断")
    ap.add_argument("--model", required=True,
                    choices=["fully-routed", "partial-shared"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--routing", default="data", choices=["data", "scenario"])
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--max-batches", type=int, default=8,
                    help="每场景最多取多少批（门控分布收敛很快，无需全量）")
    return ap.parse_args(argv)


def build(args, ckpt):
    cls = DCNv2MoE if args.model == "fully-routed" else DCNv2MoE_V2
    kwargs = dict(dim=360, K=args.K, routing=args.routing)
    if cls is DCNv2MoE_V2:
        kwargs["top_k"] = args.K
    model = cls(**kwargs).to(args.device)
    sd = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def mean_gate_per_scenario(model, args):
    """返回 {场景: 平均门控向量(list[K])}，取所有交叉层的平均。"""
    out = {}
    for s in SCENARIOS:
        test = Split(s)[2]
        acc = None
        n = 0
        loader = torch.utils.data.DataLoader(
            Dataset(test), batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= args.max_batches:
                    break
                batch = {k: v.to(args.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}
                model(batch)
                gates = batch.get("_gate")
                if not gates:
                    return {}
                # 每层 gate 形状 (B, K) 或 (B, 1, K)；跨层与样本取均值
                g = torch.stack([gg.reshape(gg.shape[0], -1).mean(0)
                                 for gg in gates]).mean(0)
                acc = g if acc is None else acc + g
                n += 1
        if n:
            out[s] = (acc / n).float().cpu().tolist()
    return out


def metrics(gate_by_scenario, K):
    """计算三个指标。

    - uniformity（均匀度）：各场景门控的归一化熵均值，越接近 1 越均匀。
    - specialization（按场景分工度）：场景间门控向量的平均 L1 距离，
      越大表示「不同场景用不同专家」越明显。
    - imbalance（专家间不均衡度）：场景平均门控向量的 max/min 比，
      越大表示「某些专家总体被偏好」越明显。

    必须三者同看：按场景分工度下降 + 专家间不均衡度上升，代表分工方式
    从「按场景切换」转为「固定偏好某专家」，而非分工消失。
    """
    import math
    ent = []
    for g in gate_by_scenario.values():
        tot = sum(g) + 1e-30
        p = [max(x, 0.0) / tot for x in g]
        h = -sum(x * math.log(x + 1e-30) for x in p)
        ent.append(h / math.log(K))
    ss = list(gate_by_scenario.values())
    dists = []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            dists.append(sum(abs(a - b) for a, b in zip(ss[i], ss[j])))
    avg = [sum(x[i] for x in ss) / len(ss) for i in range(K)] if ss else []
    imbalance = (max(avg) / max(min(avg), 1e-30)) if avg else None
    return {
        "uniformity": sum(ent) / len(ent) if ent else None,
        "specialization": sum(dists) / len(dists) if dists else None,
        "imbalance": imbalance,
        "mean_gate": avg,
    }


def main():
    args = _parse_args(sys.argv[1:])
    torch.manual_seed(args.seed)
    stem = ("moe_fully_routed" if args.model == "fully-routed"
            else "moe_partial_shared")
    targets = [("pretrain(未调制)", f"{CACHE_DIR}/{stem}_seed{args.seed}.pt")]
    for p in sorted(glob.glob(
            f"{CACHE_DIR}/subtask_backbone_{args.model}_*_seed{args.seed}.pt")):
        label = os.path.basename(p).replace(
            f"subtask_backbone_{args.model}_", "").replace(
            f"_seed{args.seed}.pt", "")
        targets.append((label, p))

    report = {}
    for label, ckpt in targets:
        if not os.path.exists(ckpt):
            print(f"[skip] {label}: {ckpt} 不存在")
            continue
        model = build(args, ckpt)
        gbs = mean_gate_per_scenario(model, args)
        if not gbs:
            print(f"[warn] {label}: 未取到门控（模型未暴露 _gate）")
            continue
        m = metrics(gbs, args.K)
        report[label] = {"metrics": m,
                         "gate_by_scenario": {str(k): v for k, v in gbs.items()}}
        print(f"[{label}] 均匀度={m['uniformity']:.4f} "
              f"按场景分工度={m['specialization']:.4f} "
              f"专家间不均衡度={m['imbalance']:.2f}")
        for s, g in gbs.items():
            print(f"    scenario {s}: " + " ".join(f"{x:.3f}" for x in g))
        del model
        torch.cuda.empty_cache()

    out = f"{CACHE_DIR}/gate_diagnostics_{args.model}_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  summary → {out}")
    if "pretrain(未调制)" in report and len(report) > 1:
        base = report["pretrain(未调制)"]["metrics"]
        print("\n=== 相对未调制的变化 ===")
        for label, r in report.items():
            if label == "pretrain(未调制)":
                continue
            du = r["metrics"]["uniformity"] - base["uniformity"]
            dsp = r["metrics"]["specialization"] - base["specialization"]
            dib = r["metrics"]["imbalance"] - base["imbalance"]
            # 三种情形（不是二分）：
            #   冻结      —— 趋近均匀且两种分化都减弱
            #   分工方式改变 —— 按场景分工减弱但专家间偏好增强
            #   分工保持   —— 均无明显变化
            if du > 0.005 and dsp < 0 and dib <= 0:
                verdict = "路由趋于均匀/冻结（等效关闭路由）"
            elif dsp < -0.01 and dib > 0.2:
                verdict = "分工方式改变：按场景切换 → 固定偏好某专家"
            elif abs(du) < 0.005 and abs(dsp) < 0.01:
                verdict = "分工基本保持（路由未被冻结）"
            else:
                verdict = "需人工判读"
            print(f"  {label}: Δ均匀度={du:+.4f} Δ按场景分工度={dsp:+.4f} "
                  f"Δ专家间不均衡度={dib:+.2f}\n      → {verdict}")


if __name__ == "__main__":
    main()
