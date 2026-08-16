#!/usr/bin/env python
"""HY3 审计 fix #4/#5 的数值验证：路由器门控不变量。

不训练、不读 cache，纯张量断言，跑得快（CPU 即可）：
  1. V2 冻结（_router_noise_enabled=False）门控恒均匀、零噪声、可复现。
  2. V1 DataRouter.gate_scaling=K 时，uniform init 输出全 1，与 ScenarioRouter 一致
     （修复「统一恒等初始化」语义：此前 data router uniform=1/K, scenario=1）。
  3. V1/V2 冻结门控继承自 vanilla（gate 取 softmax(0) 后乘 scaling）：
     - V1 scenario: ScenarioRouter(softmax*K) → 1
     - V1 data:     DataRouter(softmax*1)      → 1/K
     - V2:          CrossExpertLayerV2 冻结    → 1/K（top_k=K warmup，softmax(K)）

用法：
  python scripts/verify/verify_router_invariants.py
退出码：0=全部通过，1=存在失败。
"""
import os
import sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from model import (
    ScenarioRouter, DataRouter,
    CrossExpertLayerV2, DCNv2MoE_V2, DCNv2MoE,
)

torch.manual_seed(0)
K = 4
B = 8
x = torch.randn(B, 360)
tab = torch.randint(0, 15, (B,))


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    return cond


ok = True

# --- 1. V2 冻结：恒均匀 + 零噪声 + 可复现 ---
# 忠实复现 run_moe_pretrain_from_scratch.freeze_routers：冻结 w_gate/w_noise 于零初值
# 且关闭噪声门控（V2 的 w_gate/w_noise 默认非 0 初始化，必须显式清零才得到恒均匀）。
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
# 恒均匀：warmup 模式 gates = softmax(logits)*K，冻结 logits=0 → 全 1（每个专家权重=1，
# 等价于 vanilla Cross layer 拼接）。零噪声/可复现：两次前向完全一致。
uniform_ok = torch.allclose(g1, torch.ones_like(g1), atol=1e-6)
repro_ok = torch.allclose(g1, g2, atol=0.0)
ok &= check("V2 frozen: gate uniform (=1, vanilla-equiv)", uniform_ok, f"value={g1.flatten()[0]:.4f}")
ok &= check("V2 frozen: zero-noise reproducible", repro_ok)

# 对照：未冻结 V2 在 training 下应含噪声（两次不一致）
v2b = DCNv2MoE_V2(dim=360, K=K, routing='data')
v2b.train()
h1, _, _ = v2b.cross_layers[0]._gate(x, training=True)
h2, _, _ = v2b.cross_layers[0]._gate(x, training=True)
noisy_ok = not torch.allclose(h1, h2, atol=1e-6)
ok &= check("V2 trainable: gating noise present (sanity)", noisy_ok)

# --- 2. V1 DataRouter.gate_scaling 兼容性 ---
sr = ScenarioRouter(K)
dr_def = DataRouter(360, K)              # 默认 scaling=1 → uniform=1/K
dr_scaled = DataRouter(360, K, gate_scaling=K)  # scaling=K → uniform=1

sr_gate = sr(tab)
dr_def_gate = dr_def(x)
dr_scaled_gate = dr_scaled(x)

scenario_uniform = torch.allclose(sr_gate, torch.ones_like(sr_gate), atol=1e-6)
data_def_uniform = torch.allclose(dr_def_gate, torch.full_like(dr_def_gate, 1.0 / K), atol=1e-6)
data_scaled_uniform = torch.allclose(dr_scaled_gate, torch.ones_like(dr_scaled_gate), atol=1e-6)

ok &= check("V1 ScenarioRouter uniform = 1", scenario_uniform)
ok &= check("V1 DataRouter default uniform = 1/K", data_def_uniform)
ok &= check("V1 DataRouter gate_scaling=K uniform = 1 (== ScenarioRouter)",
            data_scaled_uniform and scenario_uniform and data_def_uniform)

# --- 3. V1/V2 冻结门控继承自 vanilla 的语义对齐 ---
# 冻结 V1 scenario 层 gate 应 = 1（与 ScenarioRouter 对齐）；V1 data 层 = 1/K
from model import DCNv2MoE
moe = DCNv2MoE(dim=360, K=K, routing='data')
gate_data = moe.cross_layers[0](x, x, None)[1]
data_v1_uniform = torch.allclose(gate_data, torch.full_like(gate_data, 1.0 / K), atol=1e-6)
ok &= check("V1 data router (init) gate uniform = 1/K", data_v1_uniform,
            "注：与 ScenarioRouter(=1) 的 1/K 差异即原 docs 宣称「统一恒等」的破口；"
            "已通过 DataRouter.gate_scaling=K 提供数值对齐路径，但默认 1.0 不变以保既有训练。")

print("=" * 64)
print("router invariant checks:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
