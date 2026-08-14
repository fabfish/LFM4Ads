# 实验驱动与证据索引

> 当前状态唯一入口。交接速览见 [`HANDOFF.md`](HANDOFF.md)，综合分析见 [`ANALYSIS.md`](ANALYSIS.md)，后续授权边界见 [`NEXT.md`](NEXT.md)。
> 历史驱动、结论、审计与运维记录均保存在 `archive/`；本表只登记当前有效状态，不以历史文档中的旧状态覆盖机器判定。

## 1. 状态约定

- 生命周期：`not-started | planned | running | blocked | done | superseded`。
- 关卡判定：`PASS | FAIL | INCONCLUSIVE | BLOCKED | NOT_EVALUATED`。
- `done` 仅表示计划矩阵完整结束，不等于 `PASS`。
- `INCONCLUSIVE` 表示证据不足以支持稳定方向，不表示已证明方法显著退化。
- 后续关卡只有在上游机器判定明确 `PASS` 且解锁字段为 `true` 时才可另立驱动。

## 2. 当前路线总表

| 路线 | 生命周期 | 判定 | 当前结论 | 后续状态 | 驱动与证据 |
|---|---|---|---|---|---|
| 从零预训练与子任务调制 | `superseded` | 旧网格关闭 | 冻结均匀路由可平替旧调制；单 seed 与旧训练目标结果不作强结论 | 不重跑旧 36 配置网格 | [阶段结论](archive/conclusions/20260811-1450-子任务调制交叉网格阶段结论.md)；[旧分析](archive/analysis/20260810-2300-混合专家从零预训练与子任务调制.md) |
| 按场景 sample weighting | `done` | **PASS** | 样本加权恢复 full-batch 目标；`R_gain≈0.999`、同 seed `R_resid≈0.992` | 已解锁并完成 Stage B | [驱动](archive/drivers/20260811-2010-按场景训练代价消除与混合专家可竞争性验证.md)；[结论](archive/conclusions/20260811-2242-按场景训练代价消除结论.md)；[审计](archive/analysis/20260811-2300-按场景训练代价消除-HY3审计.md) |
| Stage B same-FLOPs MoE | `done` | **FAIL**（竞争性主张） | frozen/soft 相对 same-FLOPs dense 的同 seed 差均为 `[-,-,+]`；MoE wall-clock 更慢 | sparse scaling / same-latency 叙事关闭 | [驱动](archive/drivers/20260811-2310-StageB-MoE-九次训练驱动.md)；[结论](archive/conclusions/20260812-0103-StageB-MoE-九次训练结论.md) |
| 下游留出域参数高效迁移 | `done` | **FAIL**（G1 可行性门控） | 12/12 trial 完成；validation 上 moe-router 仅 t2 胜 dense-adapter-r2（+0.0003），t5/t6 均败；G1 未过 → 按协议停止、未扩 seeds 123/456 | 停止，不扫描 LR/K/r/target | [驱动](archive/drivers/20260812-2303-MoE下游留出域参数高效迁移驱动.md)；[结论](archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md)；[G1 判定](../cache/downstream_transfer/g1_decision.json) |
| Capacity-MoE 真实稀疏（full-rank + top-k dispatch） | `done` | **PASS**（哨兵） | 首个 full-rank 专家 + 真实 top-k 稀疏 dispatch；STE+warmup+lb 调优首次让 router 熵离开 log K，12/12 哨兵 PASS；test AUC Δ 微弱正向（+0.0002~+0.0025，top_k=2 优于 1），未达显著 | AUC 优势未证明，暂不扩 3-seed | [驱动](./20260813-1642-capacity-MoE-驱动.md)；[结论](./20260813-1812-capacity-MoE-smoke结论.md) |
| Capacity-MoE 必要性验证（dense-continued 公平对照） | `done` | **INCONCLUSIVE**（方向为负） | 剥离"继续训练红利"后 Δ_necessity=[−0.0019,−0.0006] 2 seed 均为负；+0.0025 被证实为微调红利；router 特化 PASS 但稀疏化崩 AUC（特化与 AUC 反相关） | 不扩 3-seed；先诊断"稀疏化为何崩 AUC"再谈 scaling | [驱动](./20260814-1120-capacity-MoE-必要性验证驱动.md)；[结论](./20260814-1239-capacity-MoE-必要性结论与scaling迁移.md) |
| AdaTask 三模式 × capacity MoE | `done` | **INCONCLUSIVE**（调制无益） | 三模式熵均离开 log K，但方向反直觉（suppress 最特化）；真实稀疏下未激活专家 AU 冻结；AUC 均低，encourage/suppress 相对 none 负向 | 单 seed 方向证据；修"专家饿死循环"后再议 | [结论](./20260814-1522-AdaTask-capacity-MoE-结论.md) |
| 专家无收益根因定位 + 决定性容量实验 | `done` | **INCONCLUSIVE**（容量收益无效应）+ **FAIL**（稀疏净代价为负） | 修复 3 根因（84M embedding 解冻共因 / lb 8× 放大 bug / host-bound）后 Δ_necessity −0.0019→−0.0003；收益分解：4× 容量收益≈0（跨 seed 变号、噪声内），稀疏代价≈−0.001（4/4 同号）；瓶颈在 embedding(97.9% 参数)非 cross(0.46%) | cross 层 MoE 上限=打平 dense，不再调参；下一步做 dense-widened 4× 对照 | [结论](./20260814-2111-专家无收益根因定位与决定性容量实验.md) |
| dense-widened 4× 对照（容量瓶颈独立证伪） | `done` | **FAIL**（容量收益无可测效应） | 给 cross 层 4× 真实容量（加宽+非线性、零路由零稀疏）Δ_capacity={+0.0005,−0.0003,+0.0001,−0.0005} 跨 seed 变号、噪声内；独立证实容量非瓶颈 | capacity-MoE 路线正式关闭（结构上无利可图） | [结论](./20260814-2111-专家无收益根因定位与决定性容量实验.md) §六.5 |
| **E0/E0b embedding 伪瓶颈证伪（零成本诊断）** | `done` | **FAIL**（"瓶颈在 embedding"被证伪） | 三张 ID 表 = embedding 的 99.96%（83,984,250 参数）；推理期全部清零 Δ=**−0.000193**（噪声内）；`upload_type`(320 参数) 单独清零 Δ=−0.0053；`video_id` 平均 2.6 次曝光、test **74.19% OOV**；**17.53% 参数无梯度** | 撤销 embedding-widened 实验；方向改特征信息侧；E1 已启动 | [预注册+审计](./20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md)；[E0 证据](../cache/embedding_capacity/diagnosis.json)；[E0b 证据](../cache/embedding_capacity/field_ablation.json) |
| **E1 ID embedding 死重重训练确认** | `done` | **PASS**（死重成立） | 6 run（2 seed × 3 臂）0 失败、哨兵 10/10 PASS；Δ_id = idzero−full = **{+0.000688, +0.000220}** 2/2 seed 在噪声地板内 → 84M ID 表**无净贡献**；参数 **−99.31%**（84,672,605→584,255）、wall **1.48×** | 解锁特征信息侧（E2/E3/E4）；`iddrop` 成为后续默认轻量基线 | [E1 结论](./20260814-2225-E1结论-ID-embedding死重确认.md)；[机器判定](../cache/embedding_capacity/e1_decision.json)；[预注册 §5](./20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md) |
| TCMR 静态任务条件路由 | `done` | **INCONCLUSIVE** | 15/15 完成；DATR 对 FUR、DOR 的同 seed pooled-AUC 差均跨 seed 变号且未越过噪声地板 | AdaTask、持续学习、BWT、稀疏扩展仍 `blocked` | [驱动](archive/drivers/20260812-1139-Task-Conditioned-Mixture-Routing-驱动.md)；[结论](archive/conclusions/20260812-1703-TCMR-结论.md)；[机器判定](../cache/task_conditioned_mixture_routing/gate_decision.json) |
| 共享残差 G1：函数保持 upcycling | `done` | **PASS** | 3 seed 的 logits/loss/AUC 与 dense 在预注册阈值内一致 | 仅与 G2 一起授权既定 G3 | [正式驱动](archive/drivers/20260812-1807-共享残差混合专家-函数保持与持续学习-驱动.md)；[G1/G2 结论](archive/conclusions/20260812-1832-共享残差混合专家-G1G2不变量结论.md)；[不变量](../cache/audit/shared_residual_continual/shared_residual_experiment_invariants.json) |
| 共享残差 G2：LR/冻结语义 | `done` | **PASS** | pre-Adam 常数缩放 update ratio≈1；parameter-group 10× LR update ratio≈9.9982；冻结与更新隔离通过 | 仅授权既定 G3 | 同上 |
| 共享残差 G3：specialist-only screen | `done` | **INCONCLUSIVE** | 24/24 完成、0 失败；3 档 LR 均未同时满足双 order 的 BWT 方向与实际显著性规则；无 winner | `unlock_shared_path_necessity_gate=false`；后续均 `blocked` | [结论](archive/conclusions/20260812-1956-共享残差specialist-screen-INCONCLUSIVE结论.md)；[机器判定](../cache/shared_residual_continual/specialist_screen_gate_decision.json)；[矩阵状态](../cache/manifests/shared_residual_continual/matrix_state.json) |

## 3. 关键证据链

### 3.1 Sample-weighting Gate

- 数值不变量：`cache/manifests/sample_weighting/equ_swg_status.json`。
- 12 次运行与汇总：`cache/manifests/sample_weighting/`、`cache/moe_pretrain_summary_swg-*.json`。
- 结论边界：sample-weighted dense 可称与 full-batch **数值等价**；MoE 三组只能称“未稳定匹配、存在 seed 级反转”。
- 冻结路由语义：uniform、零噪声；不得沿用“冻结仍注入噪声”的旧错误表述。

### 3.2 Stage B

- 9/9 succeeded：`cache/manifests/sample_weighting/stgb-*.json`。
- 同 seed 配对差跨 seed 变号，不能用跨 seed 均值宣称超过、匹配或等价。
- same-FLOPs 是理论预算口径，不等于 same-latency；实测 wall-clock 不支持效率优势。

### 3.3 TCMR

- 矩阵：`cache/manifests/task_conditioned_mixture_routing/matrix_state.json`，15/15 完成。
- 机器判定：`cache/task_conditioned_mixture_routing/gate_decision.json`，`status=done`、`verdict=INCONCLUSIVE`。
- DATR−FUR=`[+0.000619,+0.000559,-0.000090]`；DATR−DOR=`[+0.000180,-0.000077,+0.000561]`。
- 没有路由塌缩或系统性场景退化证据，但也没有稳定收益证据。

### 3.4 共享残差 G1/G2/G3

- G1/G2：`cache/audit/shared_residual_continual/shared_residual_experiment_invariants.json`，`status=pass`、`gate_verdict=PASS`。
- G3 矩阵：`cache/manifests/shared_residual_continual/matrix_state.json`，`status=done`、24/24、`failed_trials=[]`。
- G3 判定：`cache/shared_residual_continual/specialist_screen_gate_decision.json`，`verdict=INCONCLUSIVE`、`winning_specialist_learning_rate=null`、`unlock_shared_path_necessity_gate=false`。
- 结论边界：G1/G2 只证明实现语义；G3 没有支持 specialist-only 持续适配的稳定增益，不得外推完整持续学习或稀疏扩展。

### 3.5 下游留出域参数高效迁移

- 代码审计：旧 MoE downstream 实际为 13 个 FeatureUsage + 1 个随机初始化 Module placebo，无 true ModuleUsage/ModelUsage。
- 新主张：source `[0,1,3,4,8]` 预训练、targets `[2,5,6]` 完全留出、每 target 4096 总标注（3072 fit + 1024 validation）；比较约 4.7k 下游可训练参数的 `moe-router` 与 `dense-adapter-r2`，不声称冻结 backbone 总容量匹配。
- 协议不变量：`cache/audit/downstream_transfer/protocol_invariants.json` 为 `pass`（含 forward smoke）；四 arm 可训练参数为 361/4681/361/4693，adapter zero-init 精确为 0，source/target 与 9 组 3072/1024 索引 hash 已冻结；`formal_training_authorized=true`（用户 2026-08-12 授权）。
- 当前证据：G1（seed 42）12/12 trial 完成；validation 上 moe-router 仅 t2 胜 dense-adapter-r2（0.7635 vs 0.7633），t5/t6 均败，G1 门控 **FAIL** → 按预注册协议停止、未扩 seeds 123/456。test 差分方向混合：Δ_primary = t2 −0.0072 / t5 +0.0128 / t6 +0.0053。
- 启动边界：本路线在 G1 关闭；不扫描 LR/K/r/target。若需更强证据，仅可另立独立"扩充 seed"关卡（协议其余不变、独立目录、不覆盖本 12 trial 与判定）。

### 3.6 Capacity-MoE 真实稀疏

- 代码：`model.py` 新增 `CapacityCrossExpertLayer`（K 个 full-rank `Linear(360,360)` 专家 + 真实 top-k
  稀疏 dispatch + STE 直通）+ `DCNv2CapacityMoE`；入口 `main_moe_capacity.py`。
- 机制链（本次核心发现）：① 硬 top-1 argmax 不可微 → router 梯度=0；② STE 恢复梯度但专家同质时是噪声；
  ③ warmup（软路由 top_k=K）首次让 router 熵离开 log K（1.386→1.10）；④ lb loss 阻碍特化（sparsify 后
  熵回升）；⑤ `lb_alpha≈0.001` 是「防坍缩 + 不阻碍特化」的平衡点（熵≈1.0）。
- 哨兵：warmup+lb 矩阵 **12/12 PASS**（熵 0.28~1.28，均离开 log K）；test AUC Δ 全正但微弱
  （+0.0002~+0.0025，top_k=2 > top_k=1）。
- 结论边界：PASS 只证明「router 特化机制成立」，**未证明 capacity MoE 显著跑赢 dense**；不得据此宣称
  AUC 优势。详见[结论](./20260813-1812-capacity-MoE-smoke结论.md)。

### 3.7 Capacity-MoE 必要性验证（dense-continued 公平对照）

- 代码：`main_moe_capacity.py` 加 `--train-dense-ref` 臂（dense 继续训练 16 epoch 剥离"继续训练红利"）；
  `scripts/run_necessity_matrix.sh` 两卡并行调度（seed42 卡0 / seed123 卡1）。
- 主指标：`Δ_necessity = MoE_best − dense_cont_best`（逐 seed 同卡配对）。结果 **Δ_necessity = [−0.0019, −0.0006]**，
  2 seed 均为负；对照 `Δ_smoke = [+0.0018, +0.0017]`（对冻结 dense）。
- 关键发现：① dense-continued 同样过拟合（valid 0.7791→0.6711，第 1 epoch 即 +0.0036）；② MoE sparse 段
  valid AUC 单调崩溃（0.76→0.67），且 **router 越特化（熵 0.74）AUC 越差**——特化与 AUC 反相关。
- 判定：**INCONCLUSIVE（方向为负）**——必要性不成立；机制哨兵（router 特化 + dispatch）单独 PASS。
- 结论边界：原 "+0.0025" 是微调红利，不是 MoE 结构红利；当前规模下稀疏化代价 > 容量收益。不得据此宣称
  AUC 优势，也不得扩 3-seed。详见[结论](./20260814-1239-capacity-MoE-必要性结论与scaling迁移.md)。

### 3.8 AdaTask 三模式 × capacity MoE（真实稀疏下的路由特化分析）

- 代码：`main_adatask.py` 加 `--arch capacity` 分支（upcycle + warmup/sparse 调度 + `routing_snapshot` 记录熵/利用率）。
- 三模式熵轨迹（seed 42，log K=1.3863）：`suppress(0.694~0.735) < encourage(0.750~0.773) < none(0.772~0.804)`，
  三模式均离开 log K（特化成立），但**方向反直觉**——suppress 最特化、none 最不特化。
- 关键发现：① 真实稀疏下**未激活专家无梯度 → AU 冻结**（弱专家 AU≈0.01、强专家≈0.06，与 dispatch 正相关），
  旧全连接 MoE 无此现象；② 第一层专家坍缩（E2 dispatch 0.34~0.42），lb=0.001 未阻止；③ AUC 三模式均低
  （mean ~0.69），encourage/suppress 相对 none 均负向（−0.0035 / −0.0008），调制无益。
- 结论边界：机制（稀疏特化 + AU 冻结）成立；"encourage 促进/suppress 抑制特化"的旧叙事在 capacity 稀疏下**降级**；
  AdaTask 调制**未改善 AUC**。仅单 seed，方向证据。详见[结论](./20260814-1522-AdaTask-capacity-MoE-结论.md)。

### 3.9 专家无收益：根因定位、三处修复与决定性容量实验

- 代码：`model.py`（lb 向量化 + 去 host sync）、`dataset.py`（`GpuBatches` GPU 常驻数据表）、
  `train.py`（`infer_gpu`/`evaluate_gpu`）、`main_moe_capacity.py`（`--freeze` / `--lr-router` /
  `--full-batch-loss` / `--gpu-resident-data` / `--reinit-cross` / best-state 精确化）；
  `scripts/run_moe_fix_matrix.sh`、`scripts/run_reinit_capacity_matrix.sh`。
- **三个根因**：① 占总参数 **97.9%** 的 84M embedding 表在 `lr=1e-3` 下全程解冻 → dense 与 MoE 双双
  从 epoch1 崩到 0.67（**共因**，原"稀疏化崩 AUC"结论被此污染）；② **真 bug**：`lb_loss` 未按
  `|B_s|/|B|` 缩放，被 8 个 scenario 重复累加 ~8 倍（0.0807 → 修复后 0.0081）；③ 性能：`.item()`/
  `.any()` 每专家 3 次 host sync + `Dataset.__getitem__` 逐样本 `iloc.to_dict()` → host-bound，
  双卡 util 仅 13%/33%。
- **修复效果**：Δ_smoke +0.0018 → **+0.0037**；Δ_necessity −0.0019 → **−0.0003**（进入噪声地板 =
  统计平局）；wall 5–13× 加速（MoE 60s→10.9s/epoch）；双卡 util → **100%/100%**；
  `GpuBatches` 等价性 |ΔAUC| = 0.00e+00。
- **决定性收益分解**（reinit-cross 8 run，cross 层随机重置制造真实 headroom）：
  `top_k=K=4`（零稀疏代价、纯 4× 容量）Δ = {+0.0002, +0.0002, +0.0003, −0.0004} → **跨 seed 变号、
  噪声内 → 容量收益 ≈ 0**；`top_k=2`（真实稀疏）Δ = {−0.0010, −0.0008, −0.0009, −0.0015} → **4/4
  一致为负 → 稀疏化代价 ≈ −0.001**。峰值永远在 soft/warmup 阶段，稀疏阶段从不产出最优模型。
- 判定：**INCONCLUSIVE（容量收益无可测效应）+ FAIL（稀疏净代价一致为负）**。
- 结论边界：cross 层 MoE 在本口径下**上限就是打平 dense**，继续调 K/lb/warmup/lr 不可能翻盘；瓶颈在
  embedding（97.9% 参数）而非 cross（0.46%）。不扩 3-seed。
  详见[结论](./20260814-2111-专家无收益根因定位与决定性容量实验.md)。

### 3.10 dense-widened 4× 对照：容量瓶颈的独立证伪（收官）

- 代码：`model.py`（`DenseWidenedCrossLayer`/`DenseWidenedDCNv2`）、`main_dense_widened.py`、
  `scripts/run_widen_matrix.sh`。
- 设计：cross 层改 `x0⊙(W2·ReLU(W1·x))+x`，W1:[360,720] W2:[720,360]，参数量 1,558,440 =
  **精确 4.00×** 单层 cross；无路由/无稀疏/无 lb。关键：不能简单平均 4 个 Linear（会线性坍缩回 1 个）。
- 结果（2 seed × 2 lr × 30 epoch，同 seed 同卡，freeze sparse）：
  Δ_capacity = B(widened 4×) − A′(dense 1×) = **{+0.0005, −0.0003, +0.0001, −0.0005}**，
  峰值 valid 差 = {−0.0001, +0.0007, +0.0006, −0.0000}——**跨 seed/lr 变号、|Δ| ≤ 0.0007 < 噪声地板**。
- 判定：**给 cross 层 4× 真实容量（加宽+非线性、零路由零稀疏）换不到可测 AUC 收益**，独立证实
  "cross 层容量不是瓶颈"。据此把 capacity-MoE 失败钉死为：**容量收益 = 0（非 MoE 独有）+
  稀疏代价 ≈ −0.001（MoE 独有）→ 净效应必为负**。capacity-MoE 路线在本口径下**正式关闭**，
  且是结构上无利可图，非实现问题。
  详见[结论](./20260814-2111-专家无收益根因定位与决定性容量实验.md) §六.5。

### 3.11 E0/E0b embedding「伪瓶颈」证伪（零训练成本，修订 §3.9/§3.10 的末句推断）

- 代码：`scripts/diagnose_embedding_capacity.py`、`scripts/diagnose_field_ablation.py`；
  证据：`cache/embedding_capacity/diagnosis.json`、`cache/embedding_capacity/field_ablation.json`（均不可覆盖）。
- **参数集中度**：`video_id`+`author_id`+`music_id` = **83,984,250 参数 = embedding 的 99.96%**；
  其余 33 个字段合计仅 **33,140**。样本计数冻结基线：train 9,281,007 / valid 1,230,368 / test 1,201,670。
- **监督稀薄 + OOV**：`video_id` 平均 **2.6** 次曝光/ID、79.83% 的 ID 曝光≤2 次、
  **test 74.19% 样本的 video_id 在 train 从未出现**；`music_id` OOV 47.58%；
  **17.53%（14,725,500）的 embedding 参数从未收到梯度**。
- **推理期字段消融**（基线 `dcnv2_vanilla.pt`，test AUC 0.777490）：三张 ID 表**同时清零**
  → 0.777297，**Δ = −0.000193（噪声地板 0.001 内）**；对照 `tag` −0.006555、
  `upload_type`（**320 参数**）−0.005251、`user_id` −0.004879；top-8 依赖字段合计 29,500 参数
  （embedding 的 0.035%）。
- 判定：**"瓶颈在 embedding"被证伪**。84M ID 参数是**死重 + 过拟合负担**，不是瓶颈；
  原证据"冻结 embedding 后 AUC 更高"的正确读法是 burden 而非 bottleneck。
- 结论边界：消融测的是**推理期依赖度**，系统性**高估**重要性 → 只支持"Δ≈0 ⇒ 不可能是容量瓶颈"
  这一个方向；"84M 可删"须由 E1 重训练确认。子群 AUC（video_id seen 0.776274 / OOV 0.777041）
  不做跨子群比较。
- 方法论产出：**禁止用参数占比推断瓶颈**；长跑前先跑零成本诊断（本轮据此撤销了一个已写进
  路线图的 embedding-widened 长跑）。
  详见[预注册+审计](./20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md)。

### 3.12 E1：ID embedding 死重的重训练确认（PASS）

- 代码：`main_field_ablation.py`（三臂 + 内置哨兵）、`scripts/run_field_ablation_matrix.sh`、
  `scripts/summarize_field_ablation.py`（判定阈值硬编码）。
- 设计：from-scratch 三臂，`full`(36 字段, dim=360) / **`idzero`**(三张 ID 表置零+冻结，与 full
  **架构完全同构**、非 sparse 参数量同为 655,215) / `iddrop`(真移除, dim=330)。
  一进程一 seed → 三臂同卡，配对由构造保证。口径 `lr=1e-3`、batch 10000、15 epoch、
  best=精确 argmax(valid)、GpuBatches；**不使用 freeze sparse**（embedding 即被测对象）。
- 哨兵 **10/10 PASS**：样本计数守恒（9,281,007/1,230,368/1,201,670）、置零与冻结（前后 maxabs=0）、
  通道恒零、可训练参数差 = **83,984,250**（精确）、非 sparse 参数量相同。
- 结果（同 seed 同卡配对 test AUC）：

  | seed | full | idzero | iddrop | Δ_id | Δ_drop |
  |---|---|---|---|---|---|
  | 42 | 0.780961 | 0.781649 | 0.781523 | **+0.000688** | +0.000561 |
  | 123 | 0.781830 | 0.782051 | 0.781753 | **+0.000220** | −0.000077 |

  peak valid Δ(idzero−full) = {+0.001134, +0.000522}；参数 84,672,605 → **584,255（−99.31%）**；
  wall/epoch 10.6s → 7.1s（**1.48×**）。
- 判定：**PASS（死重成立）** —— 2/2 seed |Δ_id| < 噪声地板 0.001。机器字段
  `verdict=PASS`、`unlock_feature_information_track=true`、`unlock_id_capacity_track=false`。
- 结论边界：**"无可测差异"≠"更好"**（Δ 虽同为正但在地板内，不得写成提升）；预算限于
  lr=1e-3/15 epoch/2 seed；`iddrop` 只是工程对照（dim 变化连带缩小 cross 层）不作主判定；
  结论限于"裸 ID 查表"这一建模方式，不等于"ID 信息无用"（E2 待测）。
  本轮 full 臂 test AUC（0.7810/0.7818）高于历史 `dcnv2_vanilla.pt`（0.777490），
  原因是 best-state 精确 argmax 修复 + shuffle 训练，属口径改进，两者不可混用作对照。
  详见[E1 结论](./20260814-2225-E1结论-ID-embedding死重确认.md)。

## 4. 当前执行边界

**capacity-MoE（cross 层）路线已正式关闭**（`done / INCONCLUSIVE(容量收益无效应) + FAIL(稀疏净代价为负)`）：
机制跑通但收益为负，dense-widened 4× 对照独立证伪"cross 层容量是瓶颈"。**不再调 K/lb/warmup/lr/top_k，
不扩 3-seed。** 详见[结论](./20260814-2111-专家无收益根因定位与决定性容量实验.md) §六.5。

**"瓶颈在 embedding（97.9% 参数）"已于 2026-08-14 22:12 证伪并撤回**（§3.11–3.12）：那 84M 参数是
**死重**（E1 重训练确认 PASS，Δ_id 2/2 seed 在噪声地板内），不是瓶颈。**embedding-widened 加宽对照
实验撤销**（前提不存在）；**禁止再用"参数占比"推断瓶颈位置**。
**后续实验默认基线改为 `iddrop`（0.58M 参数、7.1s/epoch）**，可负担 3 seed × 多配置。

持续学习与稀疏扩展旧路线继续 `blocked`：

1. shared-path necessity、shared LR 与 blockwise LR 分配；
2. continual router LR、完整 continual baseline matrix 与 alignment-aware 消融；
3. sparse scale-out 与 same-latency 叙事；
4. 将 TCMR 或 G3 的跨 seed 变号结果写成稳定收益。

新下游路线已 `done / G1 FAIL`：`scripts/run_downstream_transfer_matrix.py` 完成 G1（seed 42 验证门控）后按协议停止，未续 seeds 123/456；正式结论见[G1 结论](archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md)。

**下一阶段方向**：**特征信息侧**（E1 PASS 已解锁）——E2 长尾 ID 的可泛化表示 / E3 交互表达 /
E4 天花板定位。见 [`NEXT.md`](NEXT.md) §3.2 与 [`HANDOFF.md`](HANDOFF.md) §4。

## 5. 统一测量口径

- 默认 seeds：42/123/456；跨变体使用同 seed 配对差。
- 同一 seed 的全部路由模式固定同卡；不同 seed 才可跨设备并行。
- 除被测项外，seed、batch size、LR、设备和 loss weighting 必须一致。
- 跨 scenario 只比较相对排序，不比较绝对 AUC。
- pooled AUC 为主指标时同时报告 macro/逐场景保护指标。
- sample weighting 的 `R_gain` 与同 seed残差口径 `R_resid` 不得混称。
- 单 seed 结果只作研究动机，不能进入高置信结论。

## 6. 归档与 provenance

历史材料按用途归档：

- `archive/drivers/`：已完成或失效的阶段驱动；
- `archive/conclusions/`：阶段结果与最终关卡判定；
- `archive/analysis/`：调研、审计、方案推导与旧综合分析；
- `archive/operations/`：运行维护说明；
- `archive/run_journal/`：既有运行日志索引。

TCMR 与共享残差的不可变 manifest 仍记录整理前的原始驱动路径及 SHA-256。归档文件保持原字节内容，因此原 hash 仍可核验；本节提供旧路径到归档位置的权威迁移映射，不改写历史 manifest。
