# 现有探索综合分析

> 本文回答“已经可靠地知道什么”。实时状态与证据入口见 [`DRIVERS.md`](DRIVERS.md)，允许执行的后续动作见 [`NEXT.md`](NEXT.md)。

## 1. 探索主线

现有工作经历了四次关键收敛：

1. **纠正比较基线**：旧预训练 checkpoint 与新协议不一致，改为各模型从零、同协议、3 seed 比较。
2. **识别训练目标混淆**：按场景内均值等权优化与 pooled AUC 的样本加权目标不一致；sample weighting 修复后，旧 `-0.0140` 代价基本消失。
3. **否定静态竞争性叙事**：同容量与 same-FLOPs 两套公平比较均未给出 MoE 跨 seed 稳定优势，且 Stage B 的 MoE wall-clock 更慢。
4. **把持续学习拆成受控关卡**：TCMR 静态路由未晋级；shared-residual 虽通过函数保持和 LR 语义检查，但 specialist-only 持续适配 screen 仍为 `INCONCLUSIVE`，未解锁 shared path。

因此，当前最稳健的结论不是“MoE 一定更差”，而是：**在现有 KuaiRand-1K、模型规模、预算和 3-seed 口径下，没有稳定证据支持 MoE 的静态 AUC、效率、任务条件路由或 specialist-only 持续适配收益。**

## 2. 高置信结论

### 2.1 Sample weighting 修复了目标错配

- 场景子批 loss 按样本数加权后，数学目标恢复为 full-batch per-sample mean。
- sample-weighted dense 三 seed pooled AUC 精确均值 `0.780663205`，full-batch dense 为 `0.780676857`。
- 增益恢复率 `R_gain≈0.999`；同 seed 配对残差恢复率 `R_resid≈0.992`。二者口径不同，不得混称。
- 结论：旧按场景等权训练的主要损失来自目标错配，不是“按场景梯度”本身的必然代价。

证据：[结论](archive/conclusions/20260811-2242-按场景训练代价消除结论.md)、[审计](archive/analysis/20260811-2300-按场景训练代价消除-HY3审计.md)。

### 2.2 Stage B 不支持 MoE 竞争性和效率主张

- low-rank full-dimensional MoE 相对 same-FLOPs dense：frozen 与 soft 的同 seed差均为 `[-,-,+]`。
- soft router 学到非均匀门控，但没有转化为稳定 pooled-AUC 增益。
- manifest wall-clock 显示 MoE 明显慢于 dense；same-FLOPs 不能替代 same-latency 实测。
- 结论：竞争性主张 `FAIL`，sparse scaling / same-latency 叙事关闭。

这不是“证明所有 MoE 都显著退化”；准确表述是：**该实现与预算下未稳定超过 dense，且没有效率优势。**

证据：[Stage B 结论](archive/conclusions/20260812-0103-StageB-MoE-九次训练结论.md)。

### 2.3 TCMR 没有形成稳定任务条件路由收益

- 15/15 trial 完成，无路由塌缩。
- DATR−FUR=`[+0.000619,+0.000559,-0.000090]`；DATR−DOR=`[+0.000180,-0.000077,+0.000561]`。
- 两个主比较均跨 seed 变号；平均差也未超过锚点三 seed 极差。
- 结论：`INCONCLUSIVE`、`stable_improvement_claim=NOT_SUPPORTED`，不可晋级。

任务标识产生了少量非零路由信号，但“有结构”不等于“有收益”。

证据：[TCMR 结论](archive/conclusions/20260812-1703-TCMR-结论.md)、[机器判定](../cache/task_conditioned_mixture_routing/gate_decision.json)。

### 2.4 Shared-residual 的实现正确性成立，但收益未成立

G1/G2 已确认：

- dense→shared-residual MoE 的函数保持误差满足预注册阈值；
- specialist zero-init、shared copy 与 frozen-uniform router 正确；
- pre-Adam 常数梯度缩放在稳定 Adam 下 update ratio≈`1.0`，不能称 task-specific LR；
- parameter-group 10× LR 的实际 update ratio≈`9.9982`；
- 非拥有 specialists、非活动 task head 严格不变。

G3 已确认：

- 2 orders × 3 seeds × 4 arms = 24/24 完成，0 失败；
- specialist LR `2e-4/5e-4/1e-3` 均未同时满足双 order 的 BWT 方向与实际显著性联合规则；
- `winning_specialist_learning_rate=null`；
- `unlock_shared_path_necessity_gate=false`。

因此 G1/G2 的 `PASS` 只能说明“可以可信地测”，不能转写为“已经测到持续学习收益”。G3 的准确结论是 `INCONCLUSIVE`：现有证据既不支持稳定收益，也不足以宣称普遍显著退化。

证据：[G1/G2 结论](archive/conclusions/20260812-1832-共享残差混合专家-G1G2不变量结论.md)、[G3 结论](archive/conclusions/20260812-1956-共享残差specialist-screen-INCONCLUSIVE结论.md)、[机器判定](../cache/shared_residual_continual/specialist_screen_gate_decision.json)。

## 2.5 下游证据口径重审

代码复核显示，旧 `main_moe.py` / `main_moe_v2.py` 没有完成其注释声称的三级下游：实际是 13 个 FeatureUsage 配置，加一个 `ModuleUsage(..., "Vanilla")`。后者不会加载任何预训练 sparse/cross 参数，因此只是随机初始化占位，不是 MoE Module transfer；`ModelUsage` 也未被调用。

因此，旧“MoE 三级下游 112 对”必须降格为“104 个 FeatureUsage 对照 + 8 个随机初始化占位”，不能用于否定或支持真正的模块级/模型级迁移。现有 FeatureUsage 还使用 train+validation 聚合的用户静态 `CRs`，不等于目标域标签留出的逐样本迁移。

合适的新问题不是继续扫描旧 CR 融合层，而是检验：目标场景完全留出、4096 总标注固定、下游可训练参数近似匹配时，预训练 MoE 专家能否通过 router-only 适配优于 dense adapter，并以双重差分确认增益来自适配而非仅来自冻结表示。该路线独立于已关闭的静态 pooled-AUC 与持续学习关卡，协议见[下游迁移驱动](archive/drivers/20260812-2303-MoE下游留出域参数高效迁移驱动.md)。在正式结果前状态为 `NOT_EVALUATED`，不能预设优势成立。

## 3. 已关闭或降格的主张

| 主张 | 当前处理 | 原因 |
|---|---|---|
| 旧三目标调制带来可靠收益 | 关闭 | frozen-uniform 可平替，旧结果还受训练目标错配影响 |
| MoE 与 dense “等价/匹配” | 降格 | 多组同 seed 配对差跨 seed 变号，只能称未稳定匹配 |
| same-FLOPs 意味 same-latency | 关闭 | 实测 wall-clock 更慢 |
| soft router 有结构即可支持扩展 | 关闭 | 非均匀门控没有转化为稳定 AUC 收益 |
| TCMR 已解锁持续学习 | 关闭 | `INCONCLUSIVE/no-unlock` |
| G1/G2 PASS 证明持续学习收益 | 关闭 | 它们只验证函数、优化器和冻结语义 |
| G3 可继续 shared/router LR | 关闭 | 机器字段 `unlock_shared_path_necessity_gate=false` |
| 单 seed 持续学习结果可作结论 | 降格为动机 | 已观察到高比例跨 seed 变号 |

## 4. 未决问题

1. G3 的方向不稳定来自真实异质性、seed 噪声，还是当前每 scenario 24 步预算过小？现有证据不能区分。
2. shared path 是否对持续学习有必要性？G3 未 PASS，因此尚未获得测试授权，也没有结果。
3. 更大数据、更多任务、更高容量或真正稀疏 dispatch 下是否不同？当前实验均不能外推。
4. 任务条件路由需要更强任务信号还是不同归纳偏置？TCMR 仅说明当前实现没有稳定收益。

## 5. 结论边界

- 数据集边界：结论限于 KuaiRand-1K 当前切分与场景定义。
- 架构边界：限于当前 DCNv2、K=4、rank=45 等实现；不是对所有 MoE 的普遍定理。
- 预算边界：G3 为每 scenario 24 个优化步、2 个任务顺序；不代表长预算结果。
- 统计边界：主要探索口径为 3 seed；跨 seed 变号时不得用均值覆盖不稳定性。
- 设备边界：允许跨卡并行，但同 seed 的全部配对模式必须在同一张卡。
- 指标边界：跨 scenario 只看相对排序；pooled AUC 不能独自代表小场景表现。
- 因果边界：路由熵、MI 或 churn 是机制诊断，不是收益证据。

## 6. 统一复核规则

1. 样本计数守恒，无重复、无漏样本。
2. 除被测项外 seed、batch size、LR、device、loss weighting 完全一致。
3. 跨变体使用同 seed 配对差，不用跨 seed 均值冒充匹配或改进。
4. frozen router 必须同时冻结门控参数并关闭噪声，语义为 uniform、零噪声。
5. `R_gain` 使用精确 full/equal 锚点；`R_resid` 单独报告。
6. 任一新结论先登记 claim、evidence、verdict、可降级边界，再启动运行。
7. 缺 trial、失败、hash 或公平性异常时判 `BLOCKED`，不得解释部分结果。
