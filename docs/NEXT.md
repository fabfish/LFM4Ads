# 后续安排与授权边界

> **当前决策（2026-08-17 12:15 更新）：MoE 路线已拿到公平比较下的正面结果，进入"深挖路由维度"阶段。**
>
> - **已成立**：27K + macro 端点 + 硬路由 top_k=2，Δ=**+0.005012**（轻量，4/4 seed 正，地板 3.70×）；
>   full-ID 复核 Δ=**+0.004329**（4/4 正，地板 3.24×）→ **公平性质疑排除**。
>   当前最优 macro = **0.735388**（轻量 + MoE K=5 top_k=2）。
> - **归因**：换 27K（给够测量精度）+ 换 macro 端点（对准收益位置）+ 硬路由（放大 2.7×），三者缺一不可。
>   删 ID 表只是工程使能条件，**不是收益来源**。见[突破归因](./20260817-1208-突破归因-公平比较下的正面结果.md)。
> - **下一阶段**：三个开放实验 **E12（专家利用率诊断）→ E14（top_k 单调性）→ E13（router 粒度升级）**，
>   见[开放实验计划](./20260817-1215-开放实验计划-下一个关键MoE改动.md)。
> - **已关闭**：capacity-MoE（加容量）、embedding 加宽（ID 表在 27K 上被证**有害**，−0.0055 4/4 负）、
>   Stage B、TCMR、共享残差持续学习、下游迁移、E9 K 扫描（授权缩减）。
> - **已弃用**：E2/E3/E4「特征信息侧」（当时判断"瓶颈在信息侧"，但路由维度仍有可挖收益，优先级更高）。
>
> 状态见 [`DRIVERS.md`](DRIVERS.md)，已可靠结论见 [`ANALYSIS.md`](ANALYSIS.md)，
> 交接一页纸见 [`HANDOFF.md`](HANDOFF.md)，结果索引见 [`results/INDEX.md`](../results/INDEX.md)。

## 1. 已冻结的旧路线

共享残差最终状态：

- G1 Function-Preserving Upcycling：`done / PASS`；
- G2 Learning-Rate Semantics：`done / PASS`；
- G3 Specialist-Only Screen：`done / INCONCLUSIVE`；
- `winning_specialist_learning_rate=null`；
- `unlock_shared_path_necessity_gate=false`。

因此 shared-path necessity、shared/router LR、完整持续学习矩阵、alignment-aware 消融和 sparse scale-out 均继续 `blocked`。不再扫描 specialist LR，不用单 seed 或跨 seed 均值重解释旧结果。

## 2. 已关闭的下游迁移路线（协议归档）

下游留出域参数高效迁移路线已 `done / G1 FAIL`，完整预注册协议（P0 实现审计 / P1 seed-42 可行性门控 /
P2 正式优势 gate / 禁止项）归档于[下游迁移驱动](archive/drivers/20260812-2303-MoE下游留出域参数高效迁移驱动.md)，
结论见 [G1 结论](archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md)。不再执行，仅保留其通用实验纪律：
不更换判定阈值凑正结果、不按 arm 重采样、不给 MoE 额外参数而不给 dense 匹配对照、不以机制诊断替代 test AUC 主终点。

## 3. 当前可执行动作

### 3.1 已关闭（不再执行）

- 下游迁移路线：`done / G1 FAIL`，12/12 seed-42 trial 完成，G1 验证门控失败，按预注册协议停止、未扩 seeds 123/456；结论见 [G1 结论](archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md)。
- capacity-MoE（cross 层）路线：`done / 正式关闭`，容量收益≈0 + 稀疏代价≈−0.001，dense-widened 独立证伪"cross 层容量是瓶颈"。**不再调 K/lb/warmup/lr/top_k，不扩 3-seed。** 结论见 [根因定位与决定性容量实验](./20260814-2111-专家无收益根因定位与决定性容量实验.md)。
- 共享残差持续学习 / sparse scale-out / same-latency 叙事：继续 `blocked`（§1）。

### 3.2 当前阶段方向：深挖路由维度（原"特征信息侧"已弃用）

capacity 路线关闭 + embedding 伪瓶颈证伪后，曾判断"瓶颈是可用输入信息量"，提出 E2/E3/E4
特征信息侧路线。**但 E7/E10/E11 证明路由/隔离维度仍有可挖收益（Δ 高达 +0.005，4/4 seed 正），
优先级高于特征工程 → E2/E3/E4 弃用（未执行，如需重启须重新预注册）。**

**三个硬约束（继续有效）**：
1. **加容量前必须先证明存在容量缺口**（cross 层的教训）；
2. **判断瓶颈不能用参数占比**（embedding 的教训：97.9% 参数不仅不是瓶颈，27K 上加回还有害 −0.0055）；
3. **开跑前先算噪声地板**（1K 的教训：地板 0.0113 > 待测效应 0.005，两千块实验全 INCONCLUSIVE）。

#### 已完成（E1/E5/E6/E7/E8/E10/E11，均 `done`）

| 实验 | 判定 | 关键数字 |
|---|---|---|
| E1 ID 表死重（1K） | PASS | Δ_id 2/2 在地板内；参数 −99.31% |
| E5/E6 场景内 MoE + 隔离上界（1K） | INCONCLUSIVE | 地板 0.0113 > 效应 → 换 27K |
| **E7 软路由（27K）** | **PASS** | +0.001881（地板 1.39×），首个 work case |
| E8 pooled 组合 | INCONCLUSIVE | 5/6 正；moe+pooled 绝对成绩最高 |
| **E10 硬路由 top_k=2** | **PASS** | **+0.005012（地板 3.70×）← 当前最优** |
| **E11 full-ID 公平复核** | **PASS** | +0.004329（4/4 正）；ID 表**有害** −0.0055 |

**默认基线（授权）**：轻量模型（去三大 ID 表，dim=330，0.87M 参数，3.4min/epoch）+ 27K + macro 端点，
**4 seed 口径**（42/123/456/789）——成本已不再是限制。

#### 开放实验（E12/E13/E14，`auth=planned`）

预注册见[开放实验计划](./20260817-1215-开放实验计划-下一个关键MoE改动.md)。

1. **E12 专家利用率诊断**（**必做，50 min**）：硬路由下 15 个场景是否路由到不同专家，还是坍缩到同样
   2 个？`ScenarioRouter` 的 gate 完全由 225 个数决定，dump 出来即可完全还原 dispatch → 零推理成本。
   判据：专家覆盖数 = 5 且负载 max/min ≤ 2 → 已分化（做 E13）；覆盖 ≤ 3 → 坍缩（先做 E12b）。
2. **E14 top_k 单调性**（4h，独立可先跑）：top_k 5→2 收益 ×2.7，→**1** 会继续涨（"越特化越好"）
   还是回落（"存在最优协作度"）？top_k=1 也是真条件计算最易落地的形态。
3. **E13 router 粒度升级**（12h，**关键 MoE 改动**）：`ScenarioRouter` 只有 **15 种 gate 模式**
   （每场景一套固定权重、与样本内容无关）。升级到 `DataRouter=Linear(330,5)` 样本级路由 / Hybrid。
   注意：B 臂比 A 多 4,740 参数（0.55%），**不再严格守恒，须显式登记**。
4. **E12b load-balance loss**（`blocked`，条件解锁）：仅当 E12 判定坍缩时执行。

**决策树**：E12 → 坍缩则 E12b；已分化则 E13。E14 独立，可插空跑。总预算约 17h（不含 E12b）。

所有新实验必须：
- 先算噪声地板 + 先跑零成本诊断，再决定是否开 GPU；
- 同 seed 同卡配对差 + 4 seed；
- 预注册哨兵与判定阈值，事后不得更换。

### 3.3 交接给后续执行者

- 一份纸版初级结论与路线图见 [`HANDOFF.md`](HANDOFF.md)。
- 已可靠结论（"已知道什么"）见 [`ANALYSIS.md`](ANALYSIS.md) §2。
- 实时证据索引见 [`DRIVERS.md`](DRIVERS.md) §2/§3。
- 三处修复 + GpuBatches 高性能数据路径已固化到代码，直接复用。
