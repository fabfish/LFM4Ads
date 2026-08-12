# 实验驱动表（描述性命名，禁止 D 加数字等无全称代号）

> 本表记录当前有效实验。旧文档已整体移入 `docs/archive/`，不再引用任何旧代号。
> 命名规则：脚本 / 产物 / 实验名一律用描述性全称；模型变体称
> 「全路由混合专家模型 / 部分路由加共享专家模型 / 基准稠密模型」，不称 V1/V2。

---

## 一、从零预训练与上游改进判定

- **实验名**：混合专家从零预训练与上游改进判定
- **脚本**：`run_moe_pretrain_from_scratch.py`
- **三类 backbone（均随机初始化，不加载任何预训练权重）**：
  - 全路由混合专家模型（`model.py` 中 `DCNv2MoE`，K=4，数据级路由，top_k=K 全软）
  - 部分路由加共享专家模型（`model.py` 中 `DCNv2MoE_V2`，K=4，1 共享 + 4 路由，top_k=K 全软）
  - 基准稠密模型（`model.py` 中 `DCNv2`）——**同协议从零重训 3 种子**
    （旧 `cache/dcnv2_vanilla.pt` 已弃用：系统性低估 pooled 0.0032 / 按场景 0.0126）
- **产物**：`cache/moe_{fully_routed,partial_shared}_seed{seed}.pt`、
  `cache/vanilla_from_scratch_seed{seed}.pt`、`cache/moe_pretrain_summary_*.json`
- **状态**：✅ 已完成（seed 42/123/456）

## 二、子任务 路由网络参数 / 被路由专家权重 / 常驻共享专家权重 交叉促进-抑制 调制

> 术语更正：“路由专家”统一指“被路由专家权重”，不是负责路由的专家；当前代码没有独立的领域专属专家，也没有共享专家 router。

- **实验名**：子任务三目标交叉促进抑制调制观察
- **脚本**：`run_moe_subtask_modulation.py`、`scripts/run_subtask_{grid,worker}.sh`
- **旋钮**：`--router-mode` × `--expert-mode` × `--shared-mode`，
  各取 `none|encourage|suppress`。三者是**相乘的独立维度**（交叉设计），
  非「每次只激活一个目标」的边际设计（该错误设计已回退）
- **规模**：全路由 3×3=9 + 部分路由加共享 3×3×3=27 = 36 配置 × 3 种子 = **108 次运行**
- **附加**：alpha ∈ {0.25, 0.5, 1.0} 敏感性扫描
- **状态**：✅ 已完成（结论见 `docs/20260811-1450-子任务调制交叉网格阶段结论.md`）

## 三、表征用法传导性与机制诊断

- **实验名**：特征级表征用法传导性对照 与 路由门控机制诊断
- **脚本**：`run_moe_representation_usage.py`、`run_gate_diagnostics.py`、
  `scripts/run_stage4.sh`
- **传导性**：调制后 backbone vs 未调制 backbone，2 模型 × 8 场景 × 5 融合方法
  = **160 次下游训练**；结论：**不成立**（符号检验 p=0.081 / 0.636）
- **机制诊断**：前向门控分布三指标（均匀度 / 按场景分工度 / 专家间不均衡度）。
  刻意不用 `dominance_matrix`（基于梯度平方累计，而调制正在缩放梯度 → 循环论证）
- **状态**：✅ 已完成

## 四、判别性实验：冻结路由 与 训练过程对齐

- **实验名**：冻结路由等价性检验 与 按场景训练过程混淆修正
- **脚本**：`run_moe_pretrain_from_scratch.py --freeze-router`、`--vanilla-per-scenario`
- **核心发现（两条，均反转了既有结论）**：
  1. **冻结路由（门控恒均匀、零调制）等价于最优调制**
     （全路由 0.7714 vs 0.7713，部分路由 0.7656 vs 0.7663，差值远小于噪声地板）
     → 整套子任务调制的增益可被一行冻结代码平替
  2. **训练过程混淆**：混合专家走按场景训练（每场景等权）、
     稠密基准走整批训练（每样本等权），而 pooled AUC 是样本加权。
     该过程差异值 **-0.0140**，是任何结构差异（≤0.0047）的 3 倍。
     对齐后：**全路由 + 冻结路由 +0.0047 优于同过程稠密基准**
- **专家数扫描**：K=2/4/8 在冻结路由下无实质差异（跨度 0.0017，与噪声同量级）
- **状态**：✅ 已完成

## 五、按场景训练代价消除与混合专家可竞争性验证

- **实验名**：按场景损失加权方式对齐 与 混合专家全局可竞争性验证
- **驱动文档**：`docs/20260811-2010-按场景训练代价消除与混合专家可竞争性验证.md`
- **核心问题**：按场景训练的 -0.0140 代价是「按场景梯度」的内生代价，
  还是当前实现（场景内均值 loss 累加）的缺陷？
- **方案**：新增 `--scenario-loss-weighting {equal, sample}`。
  `sample` 把子批量 loss 乘 `mask.sum()/tab_batch.numel()`；不能使用字典的 `len(sub)/len(batch)`，后者得到的是字段数。
  该权重使总目标严格等于整批训练，同时**保留按场景独立前向反向**（调制所需梯度仍可得）
- **关卡实验**：稠密基准 + 按场景 + 样本加权 × 3 种子，
  与 0.7807（整批上界）/ 0.7666（等权下界）比较。仅通过才铺开
- **状态**：✅ 已完成（PASS，2026-08-11）；**审计状态：`auth=done`**
  （sample-weighting 数值不变量 PASS + 路由器不变量 PASS；增益恢复率
  `R_gain≈0.999`，同 seed 配对残差恢复率 `R_resid≈0.992`，两者禁止混称）
- **结论文档**：`docs/20260811-2242-按场景训练代价消除结论.md`
- **关键结果**：
  - `equ_swg_status.json` 当前机读状态为 PASS（`loss_abs_err=0.0`，梯度容差 ≤1e-6/1e-5）；
    历史文档中的 5.96e-08 只作为早期运行记录，不冒充当前 marker 数值
  - 稠密 sample 三种子精确均值 **0.780663205**，整批稠密三种子精确均值
    **0.780676857**（|Δmean|=1.37e-5）；增益恢复率 **R_gain≈0.999**，
    同 seed 配对残差恢复率 **R_resid≈0.992** → 关卡 **PASS**
  - 铺开 9 次 MoE（frout/nrout/pshr × 3 seed）全部落在 0.7803–0.7808，
    与 sample-dense / full-batch-dense 差距 ≤0.0004 ≪ 噪声地板 0.0036；
    但**逐 seed 配对差在三组均跨 seed 变号** → **降级为「未稳定匹配、存在 seed 级反转」**，不称「等价/匹配」
  - normal-router 与 frozen-router 逐 seed 配对差跨 seed 变号（42:+0.0006 / 123:-0.0003 / 456:+0.0005）
    → **旧 router 调制 36 配置网格永久关闭**（只能称「无稳定方向性差异」）
  - **V2 冻结路由器噪声已修复**：原 `CrossExpertLayerV2._gate` 在 `w_noise` 冻结为 0 时仍注入
    `F.softplus(0)+1e-2` 量级噪声；现 `freeze_routers` 同时置 `_router_noise_enabled=False`，
    冻结即「门控恒 uniform、零噪声」（`docs/20260811-2300-…HY3审计.md` fix#4）
  - **含义**：混合专家在 KuaiRand-1K 同容量下对稠密无 AUC 增益；先前 +0.0047 全是未修复训练坑的残差
- **结论边界 6 规则**：见 `docs/20260811-2242-…结论.md` §8 与 `docs/20260811-2300-…HY3审计.md`；
  启动后不得擅自改判更弱。
- **向后兼容**：默认 `--scenario-loss-weighting equal` 保持历史行为。历史简单 run-code 仅用于追溯；
  新实验的参数、产物和文档必须使用已展开的明确英语缩写，并以完整描述性英文 run name 防覆盖。

## 六、MoE + AdaTask 持续学习研究总纲

- **讨论文档**：`docs/20260811-2050-MoE-AdaTask-持续学习问题定义与研究方案.md`
- **四参数块建议**：共享/专精路径分配门、常驻共享专家权重、领域条件专精路由器、被路由专精专家权重。
- **关键修正**：AU 只衡量更新幅度，后续需联合跨任务梯度方向；真正分离 task-wise optimizer state，不能把 AdamW 前梯度缩放直接称为任务特殊学习率。
- **进入条件**：第五节 sample-weighting 关卡已 **PASS**（2026-08-11），进入条件满足。
- **重要约束**：关卡显示 MoE 在 KuaiRand-1K 同容量下对稠密**无 AUC 增益**。
  same-FLOPs 只能说明理论计算预算；same-latency 必须实测且 Stage B 已失败。
  当前只允许继续检查任务条件路由结构，不得外推持续学习能力。
- **状态**：进入条件已满足；**审计状态：`auth=done`**（九次训练全部 succeeded，结论已定稿）。
  驱动文档：`docs/20260811-2310-StageB-MoE-九次训练驱动.md`；结论：`docs/20260812-0103-StageB-MoE-九次训练结论.md`。
- **Stage B 结论（同 FLOPs 比较）**：低秩全维专家 MoE（frozen / soft）相对
  same-FLOPs dense 的逐 seed 配对差均为 `[-,-,+]`，只能称**未稳定超过、存在 seed 级反转**，
  不能称“匹配”。soft routing 学到非均匀门控，但未转化为 AUC 收益。
  manifest 记录的 wall-clock 显示 MoE 明显慢于 dense，因此 **same-latency 不成立**，也无效率收益。
  ⇒ **触发失败关闭条件：禁止 sparse scaling / same-latency 叙事**。后续只允许进入
  Task-Conditioned Mixture Routing（TCMR，任务条件混合路由）的静态路由关卡；持续学习、AdaTask 更新机制和真实稀疏计算继续 blocked。

## 七、Task-Conditioned Mixture Routing（TCMR）静态路由关卡

- **驱动文档**：`docs/20260812-1139-Task-Conditioned-Mixture-Routing-驱动.md`
- **结论文档**：`docs/20260812-1703-TCMR-结论.md`
- **状态**：`auth=done`；15/15 完成；关卡判定 **INCONCLUSIVE（不可晋级）**。
- **矩阵**：Frozen Uniform Routing（FUR）、Data-Only Routing（DOR）、Task-Only Routing（TOR）、
  Data-and-Task Routing（DATR）、Data-and-Task Consistency Routing（DATCR）× seeds 42/123/456。
- **主结果**：DATR−FUR 为 `[+0.000619,+0.000559,-0.000090]`；DATR−DOR 为
  `[+0.000180,-0.000077,+0.000561]`，两组同 seed pooled AUC 配对差均跨 seed 变号。
- **稳定性门槛**：平均配对差 `0.000363 / 0.000221`，均未超过锚点三 seed 极差
  `0.001819 / 0.002016`；`stable_improvement_claim=NOT_SUPPORTED`。
- **边界**：只验证静态任务条件路由；没有解锁 AdaTask 更新、持续学习、BWT 与 sparse scaling。
- **命名**：所有新 run / 日志 / manifest / summary 使用完整描述性英文名称，禁止简单 stage code。

## 八、MoE 学习率分配与持续学习研究

- **审计与总驱动**：`docs/20260812-1723-MoE-学习率分配审计与实验计划.md`
- **核心纠正**：当前仓库没有独立 router/shared/specialist/task LR 的已完成实验；现有 AdaTask 是
  单一 AdamW 前的梯度乘因子，不能称 task-specific learning rate。
- **数学对象**：allocation gate、always-on shared expert weights、task-conditioned specialist router、
  routed specialist expert weights；当前 TCMR 只具备后两者和共享 backbone，不能冒充完整四块。
- **学习率先验（待检验，非结论）**：`η_expert=5e-4` 候选锚点；shared 为 `0.05–0.1×`；
  global router 初始冻结；新任务 router 行为 `0.1–0.25×`，旧任务行严格冻结且不做 weight decay。
- **新结构路线**：TCMR 的 `INCONCLUSIVE/no-unlock` 不自动解锁持续学习；现改为从函数保持的
  `shared residual + fully-routed specialists` 重新建立授权链。
- **当前关卡**：G1 `Function-Preserving Dense-to-MoE Upcycling Gate` + G2
  `Learning-Rate Semantics Invariant Gate` 已实现；正式运行前必须由 invariant report 判定 PASS。

## 九、共享残差 MoE 函数保持与专家持续学习筛选

- **正式驱动**：`docs/20260812-1807-共享残差混合专家-函数保持与持续学习-驱动.md`
- **入口**：`bash scripts/shared_residual_experiment.sh start`
- **状态**：G1/G2=`done/PASS`，G3=`not-started/authorized/NOT_EVALUATED`；结论见
  `docs/20260812-1832-共享残差混合专家-G1G2不变量结论.md`；授权只覆盖 G3 的 24 次 two-task screen。
- **架构**：dense Cross 逐位复制为 always-on shared path；K=4 低秩全维 specialists 的 up projection
  零初始化；router frozen uniform；15 个 task head 拆为独立参数，旧任务 head 不进入 optimizer。
- **矩阵**：`2 orders × 3 seeds × (1 head-only + 3 specialist LR)=24 trials`；每 task 固定 2 epochs，
  禁用 early-stop；head LR 始终固定 `5e-4`，仅 specialist LR 扫描 `2e-4/5e-4/1e-3`。
- **双卡**：seed 42/456→`cuda:0`，seed 123→`cuda:1`；同 seed 的所有配对同卡，每卡单 trial 串行。
- **失败隔离**：单 trial 失败写入 `failed_trials` 后自动运行下一项；immutable run/log 不覆盖；
  任何缺失或失败使汇总 Gate=`BLOCKED`。
- **解锁边界**：只有 G3 双 order、三 seed 的 BWT/LA 机器规则 PASS，才允许另立 shared-path Gate；
  shared LR、router LR、完整 baseline matrix 与 sparse scale-out 当前均 blocked。

---

## 测量口径（通用）

- **种子**：42 / 123 / 456（3 种子；单种子结论不可信——
  上一阶段 34 个单元中 11 个（32%）在三种子间变号）
- **判定标准**：3 种子 Δ 同号 **且** |Δ均值| 超过同变体基线的种子间极差（噪声地板）。
  已知噪声地板：稠密整批 0.0017 / 稠密按场景 0.0036 /
  全路由正常 0.0036 / 全路由冻结 0.0010 / 部分路由正常 0.0054 / 部分路由冻结 0.0013
- **强制前置检查**：任何跨模型比较前，必须先确认两者训练过程
  （损失归约方式、优化器、早停、洗牌）完全一致，否则测到的是过程差异而非模型差异
- **双指标并列**：pooled AUC（主）与按场景 AUC 均值（辅）必须同时报告；
  两者常出现系统性分歧（pooled 涨而按场景跌 = 牺牲小场景换大场景）
- **设备**：2 × NVIDIA H20（97GB），同配置独占单卡；
  双卡并行时按「模型 × 种子 × 参数子集」预划分任务，避免 skip-if-exists 竞态
- **路由**：数据级路由（`routing='data'`）
- **超参**：预训练 batch=10000、lr=1e-3、AdamW；子任务调制 batch=16384、
  1 轮按场景调制训练、alpha=0.5、beta=0.99

## 产物命名防覆盖规则

默认口径不加后缀（兼容既有产物），非默认值必须加后缀，否则扫描会覆盖既有结果：

| 旋钮 | 默认 | 非默认后缀 |
|---|---|---|
| `--alpha` | 0.5 | `_a{alpha}` |
| `--K` | 4 | `_K{K}` |
| `--freeze-router` | 关 | `_frozenrouter` |
| `--vanilla-per-scenario` | 关 | `_perscenario` |
| `--scenario-loss-weighting` | equal | `_samplew` |
