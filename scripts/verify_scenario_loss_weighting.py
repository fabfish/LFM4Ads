#!/usr/bin/env python
"""数值等价校验（run-code: equ-swg）。

验证 train.py::scenario_loss 的 sample 加权与整批 BCE mean 严格等价（§六 不变量）：
  1. 损失等价：整批 BCE mean 与「按场景 BCE mean 的样本加权和」绝对误差 <= 1e-6
  2. 梯度等价：对权重的梯度最大绝对误差 <= 1e-6，相对误差 <= 1e-5
  3. equal 一致性：scenario_loss(equal) == 各场景 mean 之和（历史 train_moe 行为）
  4. 未知 weighting 必须抛 ValueError

不进入正式训练热路径；确定性（固定 seed + 单 batch + 无随机层模型）。
失败则写 equ_swg_status.json = {"status":"fail"}，供 summarize 判 DEBUG。
"""

import hashlib
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dataset import Dataset, Split  # noqa: E402
import fields  # noqa: E402
from model import (  # noqa: E402
    DCNv2, CrossExpertLayerV2, DCNv2MoE, DCNv2MoE_V2, ScenarioRouter, DataRouter,
)
from train import scenario_loss  # noqa: E402

MANIFEST_DIR = os.path.join(REPO, "cache", "manifests", "sample_weighting")


def assert_router_invariants():
    """HY3 审计 fix #4/#5 的路由器门控不变量（冻结即零噪声、恒均匀）。

    与原 equ-swg 损失等价校验并列，作为 Stage A 的前置门控：任一失败即
    DEBUG。返回 (ok, errors)。
    """
    errs = []
    torch.manual_seed(0)
    K = 4
    x = torch.randn(8, 360)

    # V2 冻结：忠实复现 freeze_routers（清零 w_gate/w_noise + 关噪声）→ 门控恒 uniform(=1,
    # warmup 模式 softmax*K，等价于 vanilla Cross)。
    v2 = DCNv2MoE_V2(dim=360, K=K, routing='data')
    for layer in v2.cross_layers:
        layer._router_noise_enabled = False
        torch.nn.init.zeros_(layer.w_gate.weight)
        torch.nn.init.zeros_(layer.w_gate.bias)
        torch.nn.init.zeros_(layer.w_noise.weight)
        torch.nn.init.zeros_(layer.w_noise.bias)
    v2.train()
    g1, _, _ = v2.cross_layers[0]._gate(x, training=True)
    g2, _, _ = v2.cross_layers[0]._gate(x, training=True)
    if not torch.allclose(g1, torch.ones_like(g1), atol=1e-6):
        errs.append("V2 frozen: gate not uniform(=1, vanilla-equiv)")
    if not torch.allclose(g1, g2, atol=0.0):
        errs.append("V2 frozen: gate not reproducible (noise still injected)")

    # V1 data router 默认 uniform=1/K；gate_scaling=K 时对齐 ScenarioRouter=1
    dr_def = DataRouter(360, K)
    dr_scaled = DataRouter(360, K, gate_scaling=K)
    if not torch.allclose(dr_def(x), torch.full_like(dr_def(x), 1.0 / K), atol=1e-6):
        errs.append("V1 DataRouter default gate not uniform(1/K)")
    if not torch.allclose(dr_scaled(x), torch.ones_like(dr_scaled(x)), atol=1e-6):
        errs.append("V1 DataRouter gate_scaling=K not uniform(1)")
    return (len(errs) == 0), errs
LOSS_TOL = 1e-6
GRAD_ABS_TOL = 1e-6
GRAD_REL_TOL = 1e-5


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # 前置门控：路由器不变量（fix #4/#5）。失败即 DEBUG，不进入正式校验。
    router_ok, router_errs = assert_router_invariants()
    if not router_ok:
        marker = {"status": "fail", "errors": [f"router-invariant: {e}" for e in router_errs]}
        with open(os.path.join(MANIFEST_DIR, "equ_swg_status.json"), "w") as f:
            json.dump(marker, f, indent=2)
        print("FAIL router-invariant")
        for e in router_errs:
            print("  -", e)
        return 1

    model = DCNv2().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 一个真实小批量（关闭洗牌，确定性）
    loader = torch.utils.data.DataLoader(
        Dataset(Split("all")[0]), batch_size=10000,
        num_workers=4, pin_memory=False, shuffle=False,
    )
    batch = next(iter(loader))
    for f in fields.all:
        batch[f] = batch[f].to(device).int()
    tab_batch = batch["tab"]

    def grads_of_model():
        return {id(p): p.grad.detach().clone()
                for p in model.parameters() if p.requires_grad and p.grad is not None}

    # 1) 整批 loss + grad
    # 注意：DCNv2.forward 原地写入 batch["logit"]，返回 None。
    model.train()
    model.zero_grad()
    model(batch)
    loss_full = criterion(batch["logit"], batch["is_click"].float())
    loss_full.backward()
    grad_full = grads_of_model()

    # 2) sample 加权 逐场景 loss + grad
    model.zero_grad()
    loss_sp = torch.tensor(0.0, device=device)
    for s in tab_batch.unique():
        mask = tab_batch == s
        sub = {k: v[mask] for k, v in batch.items()}
        model(sub)
        loss_sp = loss_sp + scenario_loss(
            criterion, sub["logit"], sub["is_click"].float(),
            mask, tab_batch, "sample",
        )
    loss_sp.backward()
    grad_sp = grads_of_model()

    # 3) equal 逐场景 loss（应等于各场景 mean 之和）
    model.zero_grad()
    loss_eq = torch.tensor(0.0, device=device)
    for s in tab_batch.unique():
        mask = tab_batch == s
        sub = {k: v[mask] for k, v in batch.items()}
        model(sub)
        loss_eq = loss_eq + scenario_loss(
            criterion, sub["logit"], sub["is_click"].float(),
            mask, tab_batch, "equal",
        )

    # ---- 断言 ----
    errors = []

    loss_abs = (loss_sp - loss_full).abs().item()
    if loss_abs > LOSS_TOL:
        errors.append(f"loss equivalence: |L_sp - L_full|={loss_abs:.2e} > {LOSS_TOL:.0e}")

    # equal 应等于各场景 mean 之和（手工重算一份做对照）
    manual_eq = torch.tensor(0.0, device=device)
    for s in tab_batch.unique():
        mask = tab_batch == s
        sub = {k: v[mask] for k, v in batch.items()}
        model(sub)
        manual_eq = manual_eq + criterion(sub["logit"], sub["is_click"].float())
    if (loss_eq - manual_eq).abs().item() > LOSS_TOL:
        errors.append("equal path != sum of per-scenario means")

    # 梯度逐项比较
    for pid in grad_full:
        g_full = grad_full[pid]
        g_sp = grad_sp.get(pid)
        if g_sp is None:
            errors.append("grad missing for a param in sample path")
            continue
        abs_err = (g_full - g_sp).abs().max().item()
        denom = g_full.abs().max().item() + 1e-12
        rel_err = abs_err / denom
        if abs_err > GRAD_ABS_TOL:
            errors.append(f"grad abs err={abs_err:.2e} > {GRAD_ABS_TOL:.0e}")
        if rel_err > GRAD_REL_TOL:
            errors.append(f"grad rel err={rel_err:.2e} > {GRAD_REL_TOL:.0e}")

    # 4) 未知 weighting 必须抛错
    try:
        mask0 = tab_batch == tab_batch.unique()[0]
        sub0 = {k: v[mask0] for k, v in batch.items()}
        model(sub0)
        scenario_loss(criterion, sub0["logit"], sub0["is_click"].float(),
                      mask0, tab_batch, "bogus")
        errors.append("unknown weighting did not raise ValueError")
    except ValueError:
        pass

    status = "pass" if not errors else "fail"
    marker = {
        "status": status,
        "loss_abs_err": loss_abs,
        "grad_abs_tol": GRAD_ABS_TOL,
        "grad_rel_tol": GRAD_REL_TOL,
        "errors": errors,
    }
    with open(os.path.join(MANIFEST_DIR, "equ_swg_status.json"), "w") as f:
        json.dump(marker, f, indent=2)

    if errors:
        print("FAIL equ-swg")
        for e in errors:
            print("  -", e)
        return 1
    print(f"PASS equ-swg | loss_abs_err={loss_abs:.2e} (<= {LOSS_TOL:.0e}) | grads ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
