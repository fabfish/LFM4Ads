# 后续安排与授权边界

> 当前决策（2026-08-14 更新）：**capacity-MoE（cross 层）路线已正式关闭**——机制跑通但收益为负
> （容量收益≈0 + 稀疏代价≈−0.001），瓶颈在 embedding（97.9% 参数）而非 cross（0.46%）。
> 下一阶段唯一有证据支持的方向是 **embedding / 特征交互侧**，且必须先做"容量缺口证伪"对照（类比
> dense-widened）再决定是否加容量。旧路线（共享残差持续学习、下游迁移）维持冻结状态不变。
> 状态见 [`DRIVERS.md`](DRIVERS.md)，已可靠结论见 [`ANALYSIS.md`](ANALYSIS.md)，
> 交接一页纸见 [`HANDOFF.md`](HANDOFF.md)。

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

### 3.2 下一阶段唯一有证据支持的方向：embedding / 特征交互侧

capacity-MoE 关闭给出一个**硬约束 + 一个可行动假设**：

- 硬约束：**加容量前必须先证明存在容量缺口**（否则只会重演 cross 层的"容量收益=0"）。
- 可行动假设：瓶颈在 84M embedding 表（97.9% 参数 + 全部过拟合压力），不是 cross 层。

建议的下一批实验（按顺序，均需预注册）：

1. **embedding 瓶颈定位对照**（最高性价比，类比 dense-widened）：给 embedding/特征交互侧加容量
   （如更高的 embedding dim、或特征交叉扩展），做一个"widened vs dense"对照。若加宽也不涨 → embedding
   侧也不是容量瓶颈，需要转向数据/任务本身；若加宽涨 → 找到了真正的容量缺口，再考虑在其上引入条件化容量。
2. **特征交互侧的表达能力**：DCNv2 的 cross 层只做 `x0⊙(W·x)` 的低阶交互，未覆盖高阶特征交叉
   （如 FM/DeepFM 式二次项、attention 式交互）。可先测"加一层特征交叉是否涨"，定位交互侧是否缺表达。
3. **数据/任务侧**：当前 AUC 天花板约 0.78，dense 已贴近；若模型侧全部证伪，则问题在数据/任务，
   需更强的标签、负采样或更大的数据。

所有新实验必须：
- 默认口径 `--freeze sparse --full-batch-loss --gpu-resident-data`（已固化的修复）；
- 同 seed 同卡配对差 + 2–3 seed；
- 先做"容量缺口证伪"对照，再谈"加容量/加专家"。

### 3.3 交接给后续执行者

- 一份纸版初级结论与路线图见 [`HANDOFF.md`](HANDOFF.md)。
- 已可靠结论（"已知道什么"）见 [`ANALYSIS.md`](ANALYSIS.md) §2。
- 实时证据索引见 [`DRIVERS.md`](DRIVERS.md) §2/§3。
- 三处修复 + GpuBatches 高性能数据路径已固化到代码，直接复用。
