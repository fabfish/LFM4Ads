# 现有探索综合分析

> 本文回答“已经可靠地知道什么”。交接速览见 [`HANDOFF.md`](HANDOFF.md)，实时状态与证据入口见 [`DRIVERS.md`](DRIVERS.md)，允许执行的后续动作见 [`NEXT.md`](NEXT.md)。

## 1. 探索主线

现有工作经历了五次关键收敛：

1. **纠正比较基线**：旧预训练 checkpoint 与新协议不一致，改为各模型从零、同协议、3 seed 比较。
2. **识别训练目标混淆**：按场景内均值等权优化与 pooled AUC 的样本加权目标不一致；sample weighting 修复后，旧 `-0.0140` 代价基本消失。
3. **否定静态竞争性叙事**：同容量与 same-FLOPs 两套公平比较均未给出 MoE 跨 seed 稳定优势，且 Stage B 的 MoE wall-clock 更慢。
4. **把持续学习拆成受控关卡**：TCMR 静态路由未晋级；shared-residual 虽通过函数保持和 LR 语义检查，但 specialist-only 持续适配 screen 仍为 `INCONCLUSIVE`，未解锁 shared path。
5. **把 capacity-MoE 完整走完并关闭**：full-rank 专家 + 真实 top-k 稀疏 dispatch 的机制（router 特化、稀疏生效）**完全跑通**，但收益为负——经三处修复 + 决定性容量对照，把收益精确拆成"4× 容量收益 ≈ 0 + 稀疏化代价 ≈ −0.001"，净效应必为负。
6. **把"瓶颈在 embedding"这一推断证伪（2026-08-14 22:12）**：用零训练成本诊断发现占总参数 97.9% 的三张 ID 表在推理期**全部清零只掉 0.000193 AUC**（噪声内），而 320 参数的 `upload_type` 单独清零掉 0.0053。**84M 参数是死重与过拟合负担，不是瓶颈**；"参数最多处即瓶颈"这一推理方式本身被否定。方向改为**特征信息侧**。

因此，当前最稳健的结论不是“MoE 一定更差”，而是：**在现有 KuaiRand-1K、模型规模、预算和 3-seed 口径下，没有稳定证据支持 MoE 的静态 AUC、效率、任务条件路由、specialist-only 持续适配，或 cross 层容量扩展收益；而 capacity-MoE 已把"容量收益≈0 + 稀疏代价<0"钉死。至于瓶颈位置，现有证据显示既不在 cross 层容量，也不在 embedding 容量（后者根本没被用上）——AUC 0.78 的天花板来自可用输入信息量。**

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

合适的新问题不是继续扫描旧 CR 融合层，而是检验：目标场景完全留出、4096 总标注固定、下游可训练参数近似匹配时，预训练 MoE 专家能否通过 router-only 适配优于 dense adapter，并以双重差分确认增益来自适配而非仅来自冻结表示。该路线独立于已关闭的静态 pooled-AUC 与持续学习关卡，协议见[下游迁移驱动](archive/drivers/20260812-2303-MoE下游留出域参数高效迁移驱动.md)。该路线 G1（seed 42）可行性门控已 **FAIL** 并停止：validation 上 moe-router 仅 1/3 target 胜 dense-adapter-r2，test 差分方向混合（Δ_primary = t2 −0.0072 / t5 +0.0128 / t6 +0.0053），未建立稳定的留出域迁移优势；详见[G1 结论](archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md)。

## 2.6 capacity-MoE 路线：机制成立、收益为负、已正式关闭

这是 2026-08-14 一整天的完整结论，也是"cross 层 MoE 有没有用"这一长期悬置问题的最终答案。三步到位：

1. **机制完全跑通**（12/12 哨兵 PASS）：full-rank 专家 + 真实 top-k 稀疏 dispatch + STE + warmup + lb 首次让 router 熵离开 log K（1.386→0.74），dispatch 校验 True。router 特化机制是真的。
2. **收益被实现缺陷污染后修正**：三个根因修复后（见下），Δ_necessity 从 −0.0019 收窄到 −0.0003（进入噪声地板 = 统计平局），Δ_smoke 从 +0.0018 翻倍到 +0.0037。
3. **决定性收益分解把失败归因钉死**（reinit-cross 8 run + dense-widened 4 run）：

| 分量 | 证据 | 数值 |
|---|---|---|
| 4× 容量收益 | `top_k=K` 零稀疏代价 / dense-widened 4× 加宽 | **≈ 0**（跨 seed 变号、噪声内） |
| 稀疏化代价 | `top_k=2` 真实稀疏 | **≈ −0.001**（4/4 一致为负） |

- **三处修复**（均已固化到代码）：① 占总参数 **97.9%** 的 84M embedding 表在 `lr=1e-3` 下全程解冻，是 dense 与 MoE 双双从 epoch1 崩到 0.67 的**共因**（原"稀疏化崩 AUC"结论被此污染）→ `--freeze sparse`；② `lb_loss` 未按 `|B_s|/|B|` 缩放、被 8 个 scenario 重复累加 ~8 倍 → 同乘缩放（lb 0.0807→0.0081）；③ `Dataset.__getitem__` 逐样本 `iloc.to_dict()` + forward 内每专家 3 次 host sync → host-bound → `GpuBatches` GPU 常驻数据表 + `--full-batch-loss`。
- **性能**：训练吞吐 11×、评估 14×、双卡 util 13–33%→100%，`GpuBatches` 与 DataLoader 路径 AUC 等价（|Δ|=0.00e+00）。
- **决定性收官**：`dense-widened 4×`（cross 层加宽 720 维 + ReLU，参数量精确 4.00×，零路由零稀疏）Δ_capacity = {+0.0005,−0.0003,+0.0001,−0.0005} 跨 seed/lr 变号、噪声内 → **独立证实"cross 层容量不是 AUC 瓶颈"**。
- **最终结论**：capacity-MoE 失败 = 容量收益 0（非 MoE 独有）+ 稀疏代价 −0.001（MoE 独有）→ 净效应必为负，**路线正式关闭，且是结构上无利可图、非实现问题**。~~瓶颈在 embedding（97.9% 参数 + 全部过拟合压力），不在 cross（0.46%）。~~ **末句已于 2026-08-14 22:12 证伪并撤回，见 §2.7。**

证据：[根因定位与决定性容量实验结论](./20260814-2111-专家无收益根因定位与决定性容量实验.md)（含 §六.5 dense-widened 收官）、[DRIVERS.md §3.6–3.10](DRIVERS.md)。

## 2.7 84M ID embedding 是死重，不是瓶颈（修订 §2.6 末句）

零训练成本诊断（两个脚本，合计 < 1 分钟，无需长跑）：

**参数集中度与监督稀薄度**（`scripts/diagnose/diagnose_embedding_capacity.py` → `cache/embedding_capacity/diagnosis.json`）：

- 三张 ID 表 `video_id`/`author_id`/`music_id` = **83,984,250 参数 = embedding 的 99.96%**；其余 33 个字段合计仅 **33,140** 参数。
- `video_id`：vocab 4,369,953，train 中出现 3,560,309，**平均曝光 2.6 次/ID**，**79.83% 的 ID 曝光≤2 次**，**test 中 74.19% 的样本其 video_id 在 train 从未出现**。
- **17.53%（14,725,500）的 embedding 参数从未收到任何梯度**，永久停留在随机初始化。
- 样本计数（后续哨兵基线）：train 9,281,007 / valid 1,230,368 / test 1,201,670。

**推理期字段消融**（`scripts/diagnose/diagnose_field_ablation.py` → `cache/embedding_capacity/field_ablation.json`，基线 ckpt `dcnv2_vanilla.pt`，test AUC 0.777490）：

| 消融对象 | 参数 | Δ test AUC |
|---|---|---|
| 三张 ID 表同时清零 | **83,984,250** | **−0.000193**（噪声内） |
| `tag` | 11,890 | −0.006555 |
| `upload_type` | **320** | −0.005251 |
| `user_id` | 10,000 | −0.004879 |
| `onehot_feat1` | **70** | −0.002973 |

- top-8 依赖字段合计 **29,500 参数（embedding 的 0.035%）**，每个的单字段 Δ 都比 84M 三大 ID **合计**的 Δ 大 8–34 倍；`upload_type`(320 参数) 的贡献是 `video_id`(43.7M 参数) 的 **292 倍**。
- 结论：**84M 参数对预测的净贡献在噪声内**。原 §2.6 用"参数占 97.9%"+"冻结后 AUC 更高"推出"瓶颈在 embedding"是**方向读反**——"冻结后更好"说明它是 **burden（负担）**，不是 **bottleneck（瓶颈）**。

**结论边界**：消融是**推理期依赖度**，不是**重训练必要性**，系统性**高估**字段重要性。因此可安全推出的只有一条：Δ≈0 的字段不可能是容量瓶颈 → **embedding 加宽路线关闭**。"84M 可直接删除"这一工程 claim 由 E1（§2.8）在重训练口径下确认。

证据：[伪瓶颈证伪与第一步实验预注册](./20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md)、[DRIVERS.md §3.11](DRIVERS.md)。

## 2.8 E1：84M ID embedding 死重的重训练确认（PASS）

三臂 from-scratch（`full` / `idzero`=三张 ID 表置零+冻结、与 full 架构完全同构 / `iddrop`=真移除 dim=330），
一进程一 seed 保证同卡配对，`lr=1e-3`、batch 10000、15 epoch、best=精确 argmax(valid)。
哨兵 **10/10 PASS**（样本计数守恒、置零与冻结前后 maxabs=0、通道恒零、可训练参数差精确 = 83,984,250、非 sparse 参数量相同）。

| seed | full | idzero | iddrop | **Δ_id** | Δ_drop |
|---|---|---|---|---|---|
| 42 | 0.780961 | 0.781649 | 0.781523 | **+0.000688** | +0.000561 |
| 123 | 0.781830 | 0.782051 | 0.781753 | **+0.000220** | −0.000077 |

- **判定 `PASS`（死重成立）**：2/2 seed |Δ_id| < 噪声地板 0.001 → 84M ID 参数**无净贡献**。
  peak valid Δ(idzero−full) = {+0.001134, +0.000522}，方向一致。
- 工程收益：参数 84,672,605 → **584,255（−99.31%）**、wall/epoch 10.6s → 7.1s（**1.48×**）。
  **`iddrop` 成为后续实验的默认轻量基线**，使 3 seed × 多配置矩阵变得廉价。
- 机器判定：`cache/embedding_capacity/e1_decision.json`，`verdict=PASS`、
  `unlock_feature_information_track=true`、`unlock_id_capacity_track=false`。

**结论边界**：①**"无可测差异"≠"删掉更好"**——Δ 虽 2/2 为正，但均在地板内，不得写成 AUC 提升；
②预算限于 lr=1e-3 / 15 epoch / 2 seed（两臂 epoch 2–4 即达峰，预算不是限制）；
③`iddrop` 的 dim=330 连带缩小 cross 层，只作工程对照不作主判定；
④结论限于**"裸 ID 查表"这一建模方式**（平均 2.6 次曝光、test 74% OOV），不等于"ID 信息本身无用"——后者是 E2 的问题；
⑤本轮 `full` 臂 test AUC（0.7810/0.7818）高于历史 `dcnv2_vanilla.pt`（0.777490），因 best-state 精确 argmax 修复 + shuffle 训练，属口径改进，两者不可混用作对照。

证据：[E1 结论](./20260814-2225-E1结论-ID-embedding死重确认.md)、[机器判定](../cache/embedding_capacity/e1_decision.json)、[DRIVERS.md §3.12](DRIVERS.md)。

## 2.9 【当前主线】MoE 在 27K + macro 端点 + 硬路由下有可复现收益（PASS）

这是本项目**第一个公平比较下的正面结果**，也是对 §2.6"capacity-MoE 关闭"的**维度补充而非推翻**：
§2.6 证伪的是"**加容量**"，本节证实的是"**改分配**"。

### 结果（27K，macro AUC 8 场景等权，同 seed 同卡配对）

| 实验 | 配置 | Δ_moe 均值 | 噪声地板 | 倍数 | 正向 seed | 判定 |
|---|---|---|---|---|---|---|
| E7 | 软路由 K=5 | +0.001881 | 0.001354 | 1.39× | 4/4 | **PASS（勉强）** |
| **E10** | **硬路由 top_k=2** | **+0.005012** | 0.001354 | **3.70×** | 4/4 | **PASS（稳健）** |
| E11 | 硬路由 + **加回 551M ID** | +0.004329 | 0.001335 | 3.24× | 4/4 | **PASS（公平性排除）** |

- **hard − soft 配对差 = +0.003131，4/4 seed 全正** → 硬路由的放大作用独立成立。
- **当前最优**：轻量 + MoE(K=5, top_k=2) = macro **0.735388**。

### 收益来源的三重证明（不是容量、不是实现差异）

1. **参数守恒**：moe − dense = **+225**（3 层 × 15 场景 × 5，纯 router 表），占 0.026%；
   full-ID 下同为 +225，占 0.00004%。
2. **frozen-router 哨兵**：把 router 冻结在 gate≡1（此时 K 个专家 concat 与单个 `Linear` 代数等价），
   实测 macro 差 −0.000261 / +0.000083 ≈ 0 → 差异确实由路由产生。
3. **优化器/数据/超参完全一致**，唯一变量是"是否按场景路由"。

### 突破归因（五组反事实，缺一不可）

| 数据集 | 端点 | 门控 | 判定 |
|---|---|---|---|
| **1K** | macro | 软 | INCONCLUSIVE（地板 0.0113 > 效应） |
| 27K | **pooled** | 软 | INCONCLUSIVE（3/4 正，收益被场景先验稀释） |
| 27K | macro | **软** | PASS 但勉强（1.39×） |
| 27K | macro | **硬 top_k=2** | **PASS 稳健（3.70×）** |
| 27K | macro | 硬 + full-ID | PASS（3.24×） |

→ **换 27K（给够测量精度，地板 ↓8.4×）与换 macro 端点（对准收益作用位置，macro 的 Δ 是 pooled 的
2–7 倍）是两个缺一不可的必要条件；硬路由把效应放大 2.7 倍使结论稳健；删 ID 表只是让实验跑得起来的
工程手段（E11 已证不是收益来源）。**

### 副产结论：ID 表在 27K 上**有害**（强于 1K 的"死重"）

加回 5.5 亿 ID 参数的净影响（同架构配对，4/4 seed 一致为负）：dense **−0.005499**、moe **−0.006182**。
§2.7/§2.8 在 1K 上说的是"删掉 AUC 不变（死重）"，27K 上升级为"加回 AUC 显著降（有害）"。

**结论边界**：①效应 +0.004~0.005 macro（约 0.5 个百分点，相对 +0.7%），方向可复现但**绝对值小**，
不得写成"大幅提升"；②**macro 与 pooled 是不同端点，不得与历史 pooled 数字混入同一张表**；
③top_k=2 是**激活稀疏**（5 个专家仍全算），wall 0.98×，**未省算力**；
④**训练后的 dispatch 分布尚未验证**——专家是否真分化、有无坍缩未知（见开放实验 E12）；
⑤`ScenarioRouter` 只有 15 种 gate 模式，样本级路由未测（E13）。

证据：[E7 结论](./20260816-1237-E7结论-27K场景内MoE首个work-case.md)、
[E8910 结论](./20260816-1950-E8910结论-硬路由topk2最终形态.md)、
[突破归因（含 E11）](./20260817-1208-突破归因-公平比较下的正面结果.md)、[DRIVERS.md §2](DRIVERS.md)。

## 3. 已关闭或降格的主张

| 主张 | 当前处理 | 原因 |
|---|---|---|
| **「MoE 在本任务上没有收益」** | **推翻（2026-08-17）** | 那是 1K + pooled 端点下的**测不出**（地板 0.0113 > 效应），不是没效应；27K + macro + 硬路由下 4/4 seed PASS（§2.9） |
| **「E2/E3/E4 特征信息侧是唯一方向」** | **弃用** | 路由维度仍有 +0.005 可挖，优先级更高；E2/E3/E4 未执行，重启须重新预注册 |
| 旧三目标调制带来可靠收益 | 关闭 | frozen-uniform 可平替，旧结果还受训练目标错配影响 |
| MoE 与 dense “等价/匹配” | 降格 | 多组同 seed 配对差跨 seed 变号，只能称未稳定匹配 |
| same-FLOPs 意味 same-latency | 关闭 | 实测 wall-clock 更慢 |
| soft router 有结构即可支持扩展 | 关闭 | 非均匀门控没有转化为稳定 AUC 收益（**限 pooled@1K**；27K macro 下软路由已 PASS，见 §2.9） |
| TCMR 已解锁持续学习 | 关闭 | `INCONCLUSIVE/no-unlock` |
| G1/G2 PASS 证明持续学习收益 | 关闭 | 它们只验证函数、优化器和冻结语义 |
| G3 可继续 shared/router LR | 关闭 | 机器字段 `unlock_shared_path_necessity_gate=false` |
| 单 seed 持续学习结果可作结论 | 降格为动机 | 已观察到高比例跨 seed 变号 |
| capacity-MoE 有微弱正向 AUC（+0.0025） | 降格 | 实为"微调红利"（dense-cont 同样 +0.0036）；对公平基线后转负 |
| "稀疏化崩 AUC（0.76→0.67）" | 关闭 | 共因是 84M embedding 在 lr=1e-3 下过拟合，dense 也崩，与稀疏化无关 |
| "router 越特化 AUC 越差" | 降格 | 熵与 AUC 同为训练进度函数，不可读作因果 |
| cross 层 MoE 可继续调 K/lb/warmup/lr 翻盘 | 关闭 | 容量收益≈0 + 稀疏代价≈−0.001 已钉死，dense-widened 独立证伪容量瓶颈 |
| **「瓶颈在 embedding 表（97.9% 参数）」** | **关闭（2026-08-14 22:12）** | 三张 ID 表清零 Δ=−0.000193（噪声内）；17.53% 参数无梯度；test 74% video_id OOV。参数占比不能推断瓶颈 |
| **「下一步做 embedding-widened 加宽对照」** | **撤销** | 现有容量本就未被用上，加宽的前提不存在（零成本挡掉一次 GPU 长跑） |
| **「模型侧全部证伪 ⇒ 数据/任务不行」** | **细化** | 不是笼统"数据不行"，而是**特征信息侧**：长尾 ID 缺可泛化表示，有效信息集中在 ~3 万参数的低基数字段 |

## 4. 未决问题

1. G3 的方向不稳定来自真实异质性、seed 噪声，还是当前每 scenario 24 步预算过小？现有证据不能区分。
2. shared path 是否对持续学习有必要性？G3 未 PASS，因此尚未获得测试授权，也没有结果。
3. ~~**瓶颈在 embedding 侧，具体是哪种？**~~ **已解决（否定式）**：embedding 容量根本没被用上（§2.7），
   84M ID 参数的净贡献 −0.000193 在噪声内。因此不存在"embedding 容量缺口"这一未决问题。
4. ~~**84M ID embedding 在重训练下是否也无贡献？**~~ **已解决**：E1 `PASS`（§2.8），
   Δ_id = {+0.000688, +0.000220} 均在地板内 → 无净贡献；模型可压到 0.58M 参数而 AUC 不变。
5. ~~**信息在哪、天花板由什么决定？**~~ **优先级下调**：曾判断"瓶颈是可用输入信息量"，据此提出
   E2/E3/E4 特征信息侧路线；但 §2.9 证明**路由维度仍有 +0.005 可挖**（4/4 seed PASS），
   优先级高于特征工程 → E2/E3/E4 弃用（未执行）。
6. **【当前主要未决】硬路由下专家是否真的分化？** §2.9 证明了收益，但训练后的 dispatch 分布
   从未记录。若 15 个场景全路由到同样 2 个专家（坍缩），则当前只用了 2/5 容量，还有大空间；
   若已分化，则粒度是新瓶颈。→ 开放实验 **E12**（50 min，零推理成本）。
7. **【当前主要未决】top_k 的单调性？** 5(软)→2 收益 ×2.7；→1 会继续涨（"越特化越好"）还是回落
   （"存在最优协作度"）？这决定机制解释，也决定能否做真条件计算省算力。→ **E14**。
8. **【当前主要未决】路由粒度是否是新瓶颈？** `ScenarioRouter = Embedding(15,K)` 只有 **15 种
   gate 模式**（每场景一套固定权重、与样本内容无关）。样本级路由（`DataRouter`）未在 macro 端点下
   测过。→ **E13（关键 MoE 改动）**。
9. 更大数据、更多任务下 cross 层是否仍无**容量**收益？dense-widened 给出强先验：加容量前先证缺口。
   注意这与 §2.9 的"改分配有收益"是两个维度，不矛盾。
10. 任务条件路由需要更强任务信号还是不同归纳偏置？TCMR 仅说明当前实现（pooled@1K）没有稳定收益；
    27K + macro 端点下未重测。

见[开放实验计划](./20260817-1215-开放实验计划-下一个关键MoE改动.md)（E12/E13/E14 预注册）。

## 5. 结论边界

- 数据集边界：**主线结论（§2.9）限于 KuaiRand-27K**；§2.1–§2.8 的历史结论限于 KuaiRand-1K。
  **两者绝对 AUC 不可比**，只信同 seed 同数据集内的配对差。
- 架构边界：限于当前 DCNv2、K=5、dim=330（轻量）/360（full-ID）等实现；不是对所有 MoE 的普遍定理。
- 预算边界：主线为 20 epoch 上限 + patience 10（两臂通常 12–16 epoch 早停）；G3 为每 scenario 24 步。
- 统计边界：主线口径为 **4 seed**（历史探索为 3 seed）；跨 seed 变号时不得用均值覆盖不稳定性。
- 设备边界：允许跨卡并行，但同 seed 的全部配对模式必须在同一张卡。
- **端点边界（关键）**：**macro AUC（8 场景等权）与 pooled AUC 是两个不同端点，不得混入同一张表**。
  macro 的 Δ 通常是 pooled 的 2–7 倍（收益作用在场景内排序）。跨 scenario 只看相对排序。
- **噪声地板边界**：地板必须**实测**且随端点/数据集变化：pooled@1K 0.001 / macro@1K **0.0113** /
  macro@27K **0.00135** / pooled@27K 0.00102 / macro@27K-fullID 0.001335。
  **开跑前先算地板**；地板 > 预期效应则实验设计本身无效。
- 因果边界：路由熵、MI 或 churn 是机制诊断，不是收益证据。
- **capacity-MoE 边界**：`容量收益≈0 + 稀疏代价≈−0.001` 严格限定在 **cross 层 + pooled 端点 + 
  KuaiRand-1K**（dim=360、K=4、top_k=2、freeze sparse）。它证伪的是"**加容量**"，
  **不适用于 §2.9 的"改分配"**——27K + macro 下硬稀疏（top_k=2）反而是正收益。
- **稀疏语义边界**：当前 top_k 是**激活稀疏**（专家全算、关闭者输出置零），wall 0.98×，
  **不得声称省算力**；真条件计算未实现。
- **字段消融边界**：§2.7 的 Δ 是**推理期依赖度**，系统性高估重要性。只能用于"Δ≈0 ⇒ 不可能是容量
  瓶颈"这一个方向；反向需重训练确认。子群 AUC 不做跨子群比较，仅作 provenance。
- **瓶颈定位方法论边界**：**参数占比不构成瓶颈证据**。须用（a）容量对照（widened）或
  （b）零成本诊断（曝光/OOV/消融）。27K 上进一步证明 97.9% 的参数加回来还**有害**（−0.0055）。
- **主线口径**：27K + 轻量模型 + macro 端点 + `--loss balanced --lr 1e-3 --batch-size 10000`。
  **历史口径**（`--freeze sparse --full-batch-loss --lr 2e-4`）仅适用于 capacity-MoE 时代的产物；
  当 embedding 本身是被测对象时（E1、E11）不得使用 `--freeze sparse`。

## 6. 统一复核规则

1. 样本计数守恒，无重复、无漏样本。
2. 除被测项外 seed、batch size、LR、device、loss weighting 完全一致。
3. 跨变体使用同 seed 配对差，不用跨 seed 均值冒充匹配或改进。
4. frozen router 必须同时冻结门控参数并关闭噪声，语义为 uniform、零噪声。
5. `R_gain` 使用精确 full/equal 锚点；`R_resid` 单独报告。
6. 任一新结论先登记 claim、evidence、verdict、可降级边界，再启动运行。
7. 缺 trial、失败、hash 或公平性异常时判 `BLOCKED`，不得解释部分结果。
