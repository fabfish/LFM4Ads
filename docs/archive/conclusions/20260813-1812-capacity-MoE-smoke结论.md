# Capacity-Scale MoE — smoke 结论（router 熵首次离开 log K）

- 创建时间：2026-08-13 18:12
- 状态：`auth=planned`（1 seed 探索，未进入 3-seed 正式矩阵）
- 前置驱动：[预注册驱动](../drivers/20260813-1642-capacity-MoE-驱动.md)
- 上级总纲：`docs/DRIVERS.md` §4 第 3 项「真实 top-k dispatch、sparse scale-out 与 same-latency 叙事」

---

## 一、一句话结论

实现了仓库第一个 **full-rank 专家 + 真实 top-k 稀疏 dispatch** 的 capacity MoE，并首次让
**router 熵离开 log K**（从 1.386 降至 0.28~1.28，取决于 lb_alpha；哨兵 12/12 PASS）。这一结果依赖
一条**关键机制链**（STE 直通 + warmup 软路由 + lb_alpha 调优）。test AUC 一致**微弱正向**
（Δ=+0.0002~+0.0025，top_k=2 优于 top_k=1），但幅度小、未达显著，不能称「capacity MoE 跑赢 dense」。
本次 smoke 的核心价值是这条机制链 + 失败路径，而非 AUC 收益本身。

---

## 二、环境与测量口径

| 项 | 值 |
|---|---|
| device | cuda:0（seed=42）、cuda:1（seed=123）；同 seed 全部变体固定同卡 |
| 数据 | `dataset.Split("all")`，train <20220503 / valid [20220503,20220506) / test ≥20220506 |
| dense 参照 | `cache/dcnv2_vanilla.pt`，本次实测 test AUC(all) = **0.7775** |
| 模型 | `DCNv2CapacityMoE`：K=4 full-rank `Linear(360,360)` 专家 × 3 层 + 2 DNN + head |
| 指标 | `torcheval.metrics.BinaryAUROC`；哨兵 = clean gate 熵（全 K 维 softmax 的 −Σp log p） |
| 训练 | per-scenario sample 加权（`train.scenario_loss`）+ warmup + 可选 lb loss |
| 超参 | seed=42/123, lr=1e-3, beta2=0.999, batch=10000, K=4 |

---

## 三、核心机制链（本次 smoke 的全部发现）

### 3.1 真实 top-k 硬稀疏 dispatch 会让 router 梯度归零

`top_k=1` 时，被选中的只有 1 个 logit，`softmax(单值) ≡ 1.0` 与 logits 无关；而选择哪个专家由
离散 `argmax` 决定，不可微。因此 BCE 主 loss 对 router 的梯度**恒为 0**（诊断实测 `bce_rgrad=0.000000`，
对任意 `noise_scale` 成立）。`top_k=2` 时 softmax 作用于 2 个值，梯度非零但极弱（0.001）。

> 这正是之前 6 条路线「router 熵始终 ≈ log K」在**稀疏**形态下的另一个必然结果——不是路由没学，
> 而是稀疏 argmax 切断了路由的梯度通路。

### 3.2 STE（straight-through）是必要但不充分的修复

用 `gate = clean_prob + (hard_gate - hard_gate.detach())`（前向硬 top-k、反向走全维 softmax 梯度），
`top_k=1` 的 BCE→router 梯度从 0 恢复到 0.068（与 lb 梯度同量级），`top_k=2` 恢复到 0.158。

但 STE 本身不够：从 dense 复制 + 小噪声（`noise_scale≤0.05`）的专家**近似同质**，STE 梯度是
「无方向的噪声」，router 被随机漂移驱动，破坏「均匀路由≈dense」的稳定性 → **AUC 崩**（18-run
矩阵实测：ns=0.01/0.05/0.1 × top_k=1 × seed 42/123，熵微降 1.36~1.37 但 AUC 全部 Δ=-0.033~-0.040）。

### 3.3 warmup（软路由）第一次让 router 真正特化

`warmup_epochs=2`（先 `top_k=K` 全软路由）时，router 的 softmax-over-K 有真实梯度，且 `noise_scale=0.1`
给专家带来的 8.6% 差异让梯度有方向：**熵从 log K=1.386 降到 1.104**，同时 AUC 保持（0.7789 ≈ dense）。
这是仓库历史上 router 熵首次显著离开 log K。

### 3.4 load-balance loss 是 sparsify 崩的元凶

warmup 阶段 `top_k=K` 时 lb loss 恒为 0；一旦 sparsify，lb loss 生效，把 router 拉向 uniform（防坍塌方向），
抵消 warmup 学到的特化：**熵从 1.10 回升到 1.36~1.38，AUC 崩到 0.72~0.76**（lb_alpha=0.01）。

### 3.5 去掉 lb loss：哨兵 PASS，但过度坍缩

`lb_alpha=0` 时 sparsify 不再崩：router 保持特化（熵继续降到 top_k=1 的 **0.256**、top_k=2 的 0.822），
AUC 保持（0.769）。top_k=1 的 **verdict = PASS**（熵 0.256 ≪ 阈值 1.2363，test AUC=0.7787 Δ=+0.0012）。

代价：熵 0.256 是**过度特化**（router 几乎 one-hot，坍缩到单个专家），K× 容量被浪费，AUC 在 sparse
阶段持续缓降（0.769→0.740），Δ=+0.0012 是 early-stop best-state 的偶然结果，不是稳定优势。

---

## 四、结果汇总表

### 4.1 机制链各阶段（seed=42）

| 配置 | 熵(末) | test AUC Δ | verdict |
|---|---|---|---|
| top_k=1, ns=0.01, 无 STE（基线） | 1.3834 | +0.0019 | INCONCLUSIVE |
| top_k=2, ns=0.01, 无 STE（基线） | 1.3844 | +0.0029 | INCONCLUSIVE |
| top_k=1, ns=0.01~0.1, STE, 无 warmup | 1.358~1.370 | -0.033~-0.040 | INCONCLUSIVE |
| top_k=1, warmup2, lb=0.01 | 1.3642(回升) | 崩 | INCONCLUSIVE |
| top_k=2, warmup2, lb=0.01 | 1.3793(回升) | 崩 | INCONCLUSIVE |

> 诊断实测（router 梯度范数）：top_k=1 无 STE = 0.000000；top_k=1 + STE = 0.068；
> top_k=2 + STE = 0.158；lb 梯度 = 0.066。

### 4.2 warmup + lb 扫描矩阵（12 run，全部 PASS）

固定 `warmup_epochs=2, noise_scale=0.1, min_epochs=6, max_epochs=10`。

| seed | top_k | lb_alpha | 熵(末 sparse epoch) | test AUC Δ |
|---|---|---|---|---|
| 42 | 1 | 0 | 0.2812 | +0.0012 |
| 42 | 1 | 0.001 | 1.0341 | +0.0012 |
| 42 | 1 | 0.003 | 1.2724 | +0.0012 |
| 42 | 2 | 0 | 0.6577 | +0.0025 |
| 42 | 2 | 0.001 | 0.9743 | +0.0025 |
| 42 | 2 | 0.003 | 1.2277 | +0.0025 |
| 123 | 1 | 0 | 0.4237 | +0.0002 |
| 123 | 1 | 0.001 | 0.9935 | +0.0002 |
| 123 | 1 | 0.003 | 1.2518 | +0.0002 |
| 123 | 2 | 0 | 0.6369 | +0.0015 |
| 123 | 2 | 0.001 | 0.9617 | +0.0015 |
| 123 | 2 | 0.003 | 1.2791 | +0.0015 |

**结论**：12/12 PASS；test AUC Δ **全为正**（+0.0002~+0.0025）；`top_k=2` 一致优于 `top_k=1`
（+0.0025 vs +0.0012 @seed42，+0.0015 vs +0.0002 @seed123）；`lb_alpha=0.001` 是「熵≈1.0、不坍缩、
不抑制」的甜蜜点。但 Δ 幅度小，只能称「微弱且一致的正向」，不能称「显著超越 dense」。

---

## 五、哨兵判定（预注册口径）

预注册规则（驱动文档 §六）：PASS 需「不崩 + 稀疏生效 + 熵均值 ≤ log K−0.15 + 分化度 ≥ 1/3」。

- warmup + lb 扫描矩阵 **12/12 PASS**：全部不崩、稀疏生效（dispatch_ok）、best-state 熵 ≤ 1.2363、分化度 ≥ 1/3。
- 无 STE / 无 warmup 的基线配置：熵未越过阈值或 AUC 崩 → INCONCLUSIVE。

**显式降级声明**：PASS 只针对「router 熵离开 log K」这一哨兵指标；**AUC 相对 dense 的优势未被证明为显著**
（Δ=+0.0002~+0.0025，幅度小、且 AUC 在 sparse 阶段有缓降趋势）。不得据此宣称「capacity MoE 跑赢 dense」，
只能称「router 特化机制成立 + AUC 微弱正向」。

---

## 六、代码改动清单

- `model.py`（尾部新增，不碰历史类）：
  - `CapacityCrossExpertLayer`：K 个 full-rank 专家 + data router + **真实 top-k 稀疏 dispatch**（按专家
    index 分组 gather，未选中专家前向不执行）+ STE 直通估计 + Switch 风格 lb loss。
  - `DCNv2CapacityMoE`：`Sparse` + 3×CapacityCrossExpertLayer + 2 DNN + head，`upcycle_from_dense`
    （专家 = dense 副本 × (1+noise_scale·randn) 打破对称）。
- `main_moe_capacity.py`（新增）：dense 参照 evaluate + upcycle + per-scenario sample 加权训练 +
  warmup/sparse 调度 + 逐 epoch valid AUC / 三层 clean-gate 熵 / per-scenario 分化度 / dispatch 校验。
- `scripts/run_capacity_matrix.sh`（新增）：18-run 无-warmup 基线矩阵调度。
- `scripts/run_capacity_warmup_matrix.sh`（新增）：12-run warmup+lb 扫描调度。

---

## 七、踩坑与修复

1. **稀疏 argmax 不可微**（router 梯度=0）→ 加 STE 直通估计。
2. **专家同质 + STE = 噪声梯度**（AUC 崩）→ 加 warmup 软路由先让 router 特化。
3. **lb loss 阻碍特化**（sparsify 后熵回升）→ 降 `lb_alpha`。
4. **过度坍缩**（lb=0 时熵 0.26，K× 容量浪费）→ `lb_alpha=0.001` 平衡（熵≈1.0）。

---

## 八、后续 backlog

1. **更长训练**：AUC 在 sparse 阶段有缓降趋势，需验证更长 epoch（>10）下能否稳定/显著超越 dense。
2. **专家坍缩/利用率度量**：统计 dispatch 分布（K 个专家是否都被使用），作为容量利用率的保护指标。
3. **结构化解耦 upcycle**：若「副本+噪声」仍不足以让 AUC 显著领先，改用 SVD/神经元聚类的结构性初始化。
4. **3-seed 正式矩阵**：当前 2 seed（42/123）Δ 全正但幅度小；正式结论需 3 seed 同卡配对差 + 显著性地板。

---

## 九、回填目标

- `docs/DRIVERS.md` §4：登记本路线 `done`（哨兵 12/12 PASS；AUC 微弱正向，未达显著）。
- 本结论文档即最终产物，lb 扫描已完成并回填 §四。
