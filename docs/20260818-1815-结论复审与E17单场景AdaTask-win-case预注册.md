# 结论复审与 E17 预注册：优先建立单场景 baseline win 和 AdaTask win-case

- 创建时间：2026-08-18 18:15
- 状态：**`auth=planned / gate=not-started`**；实现审计未通过前禁止长跑
- 用户目标：重新审视当前结论，并优先产生「一个 win 场景的 baseline + 一个 AdaTask case」
- 本文效力：复审并部分取代 [F1 结论](./20260818-1305-F1结论-优化器状态共享度baseline.md)、[F1 生命周期总结](./20260818-1512-上下文生命周期总结-F1降级与MoE降遗忘方向.md) 中过强的路线关闭表述；**supersede E16 的四臂设计**，但不改判 E10/E11
- 上游证据：[E10 结论](./20260816-1950-E8910结论-硬路由topk2最终形态.md)、[E16 旧预注册](./20260817-1410-E16-AdaTask对照预注册.md)、site B 分支 `842b8aa/fbe401f/560cc9a`

---

## 0. 术语与度量定义

| 术语 | 可执行定义及在流程中的作用 |
|---|---|
| **目标场景** | 本实验开跑前冻结为 `tab=0`（下文 `s0`）。训练仍使用全部 macro 场景；仅主检验固定为 s0，禁止看新结果后换成 s3/s5/s8。 |
| **baseline win** | 纯任务状态隔离臂 `T` 相对共享 AdamW 臂 `S` 的同 seed s0 test AUC 配对差 \(\Delta_T=AUC_{s0}(T)-AUC_{s0}(S)\)。 |
| **AdaTask win-case** | 真 task-axis、post-preconditioner AU 调制臂 `A` 相对 `T` 的同 seed s0 test AUC 配对差 \(\Delta_A=AUC_{s0}(A)-AUC_{s0}(T)\)。 |
| **AU** | 对每个参数块 p、任务组 t 维护 \(AU_{p,t}\leftarrow0.99AU_{p,t}+0.01\operatorname{mean}(g_{p,t}^2)\)。它在每个 batch 的任务方向合并前生成学习率因子。 |
| **task-axis** | 固定参数块 p，跨任务组比较 AU；不是旧实现「固定场景、跨专家」归一化。旧实现回答的是专家平衡，不等价于 task-specific learning rate。 |
| **post-preconditioner** | 先由任务独立 Adam 状态得到方向 \(d_{p,t}=\hat m_{p,t}/(\sqrt{\hat v_{p,t}}+\epsilon)\)，再乘 AU 因子；禁止在 AdamW 前只做 `grad.mul_(f)`。 |
| **one-vs-rest** | 为降低 15 次 VJP 的成本，把优化任务冻结为两组：`t=target` 是 s0，`t=rest` 是训练集中全部 `tab!=0` 样本（含不进入 macro 评估的场景）；训练样本不删除。结论只适用于「s0 vs 其余 14 场景」二任务设定。 |
| **macro AUC** | 对 `{0,1,2,3,4,5,6,8}` 各自算 AUC 后等权平均。本文用 valid macro 选 epoch/early-stop，用 test macro 作保护指标；其余场景只参与训练。 |
| **端点** | 决策点一：三臂都按同一个 valid macro 精确 argmax 选权重；决策点二：选定权重后，用 test s0 AUC 判 win。三臂端点和模型选择规则完全对称。 |
| **配对差** | 同 seed、同 site、同卡、同初始化和数据顺序下实验臂减对照臂；禁止跨 site、跨 seed 均值相减。 |
| **噪声地板** | baseline 比较用 4 个新 seed 的 S 臂计算 `floor_T=2×sample_SD(S)`；AdaTask 比较用 T 臂计算 `floor_A=2×sample_SD(T)`，s0 与 macro 各自独立计算。筛选阶段不得借用 E10 的 macro 地板作正式判定。 |
| **判定枚举** | 只使用 `PASS / INCONCLUSIVE / FAIL`。单 seed、短训、缺 raw JSON、符号混合或未越地板一律不得写 PASS。 |
| **参数守恒** | S/T/A 模型参数、初始化逐位一致；优化器状态不计模型参数，但必须单独报告显存。A 相对 T 不新增模型参数。 |
| **哨兵** | 开长跑前必须通过的数值不变量：组梯度等价、`f=1` 更新等价、task-axis、只缩放 post-Adam 方向、数据计数与模型 hash。 |

---

## 1. 复审后的严格结论

### 1.1 仍成立：E10/E11 的 MoE 总体结论

E10 的 27K、macro、轻量硬路由 MoE(K=5, top_k=2) 相对 dense：macro Δ=`+0.005012`、4/4 seed 正、3.70×地板；E11 full-ID 复核 Δ=`+0.004329`、4/4 正。F1 的优化器结果不能改判这两条 PASS。

### 1.2 F1 不能关闭「全部状态隔离路线」

严格判定如下：

| 对象 | 现有证据 | 复审判定 |
|---|---|---|
| DualOptim / `task_state_uniform` | 3.9% 数据、5 epoch：macro Δ=`+0.009352/-0.002900`，异号；两臂 best epoch 均触及上限 5 | **INCONCLUSIVE** |
| DualOptim+ | 同口径 Δ=`-0.017600/-0.015807` | **INCONCLUSIVE（强负向预警）**，非正式 FAIL |
| `role_isolated` 完整 bundle | 短训 Δ=`-0.002325/+0.000729`；site B 全数据 seed42 `0.723993-0.733109=-0.009116` | **INCONCLUSIVE（负向单-seed）** |
| 全部状态隔离路线 | `role_isolated` 同时改变状态隔离、角色 LR、router 一阶矩、head/router weight decay；不是单变量 | **不能关闭**；只暂停当前完整 bundle |

site B 的三个全数据 raw JSON 已存在于分支提交 `842b8aa`，所以 `-0.009116` 不是编造；但它仍只有一个 seed，且是多变量 bundle。正确结论是「子采样乐观外推被撤销」，不是「状态隔离已被证伪」。

### 1.3 历史 AdaTask 从未回答当前问题

历史 `AdaTaskOptimizer` 有三处决定性错位：

1. `grad.mul_(f)` 发生在 AdamW 预条件之前；因子趋稳后被一、二阶矩近似抵消。
2. `model.py::_expert_factors` 的均值轴是「同一场景内跨 K 个专家」，不是「同一参数块跨任务」。它回答专家平衡，不回答 task-specific LR。
3. 旧实验为 1K、单 seed、K=4 旧架构、valid 并入 train；不能迁移到当前 27K 硬路由口径。

因此「AdaTask 无效」应撤销为：**真 task-axis、post-preconditioner AdaTask 尚未被测，状态为 NOT_EVALUATED。**

### 1.4 E16 旧设计需要 supersede

E16 虽提出 post-preconditioner，但仍沿用 `expert_mode` 的跨专家归一化，并在 balanced/equal-task 训练上叙述「救小场景」，机制与目标不完全对齐；其 15 场景逐任务 backward 成本也被低估。E17 改为：

- 目标场景开跑前固定；
- one-vs-rest 两任务，成本从约 15 VJP/batch 降到 2 VJP/batch；
- T/A 使用完全相同的任务独立 Adam 状态；
- A 相对 T 的唯一变量是 **task-axis AU 因子是否乘在预条件后方向上**。

### 1.5 F2 EWC 暂时降为第二优先级

F2 仍是有效的持续学习问题，但它是顺序训练/BWT 设定，与当前联合训练的 AdaTask win-case 不同。按用户本轮优先级：E17 完成筛选前不启动 F2，不把两者指标混表。

---

## 2. 为什么目标场景冻结为 s0，而不是事后挑最好看的格子

### 2.1 已有结构 baseline（仅用于选候选，不直接充当新判定）

E10 中，s0 的硬路由 MoE−dense 配对差为：

| seed | 42 | 123 | 456 | 789 | 均值 |
|---|---:|---:|---:|---:|---:|
| Δ_s0 | +0.009581 | +0.009370 | +0.004543 | +0.006622 | **+0.007529** |

四个 seed 全正，说明 s0 已是可复现的 MoE win 场景之一。

### 2.2 与待测优化器最直接的候选证据

F1 短训中，纯任务状态隔离 `T` 相对共享状态 `S` 的 s0 test AUC：

| seed | S | T | T−S |
|---|---:|---:|---:|
| 42 | 0.635803 | 0.656739 | **+0.020936** |
| 123 | 0.614486 | 0.656441 | **+0.041955** |

这是目前唯一对「纯 task-state」2/2 同号且量级大的场景证据，但训练仅 3.9%×5 epoch，且 best epoch 触顶；只能用于**预先选 s0**，不能当作 PASS。

### 2.3 防止选择偏差

正式验证不再使用已参与选场景的 seeds 42/123；新 seed 冻结为 `101/202/303/404`。s3 虽有更强的 E10 结构收益，但缺少同样直接的纯状态隔离正向证据，列入后续复现而非本轮主终点。

---

## 3. E17 设计：三个臂，先 baseline，再 AdaTask 增量

### 3.1 共同口径

| 项 | 冻结值 |
|---|---|
| site | A；产物 namespace `cache/adatask_win_s0_27k_siteA` |
| 数据 | KuaiRand-27K；train/valid/test 原切分不变 |
| 架构 | 轻量 `DCNv2MoE(K=5, top_k=2)`，869,750 参数 |
| 场景 | 训练使用全部 15 个 tab；优化任务组为 `{s0, rest(tab!=0)}`；macro 仍只评估冻结的 8 场景 |
| loss | `L=(L_s0+L_rest)/2`；三臂完全一致；这是目标场景 case，不声称复现 F1 的 15-task 等权目标 |
| batch / lr | 10,000 / `5e-4`（复现 F1 候选口径） |
| epoch | max 20，patience 10 |
| 模型选择 | valid macro 精确 argmax；三臂相同 |
| seeds | 101/202 为预注册筛选；通过后追加 303/404，正式表只报四 seed |
| 设备 | 101/303→cuda:0；202/404→cuda:1；同 seed 三臂同卡顺序运行 |

`L_rest` 是 batch 内全部 `tab!=0` 样本合并后的逐样本 BCE 均值。若某 batch 缺 s0 或 rest，跳过该 batch并记录；跳过规则三臂相同。不得把 rest 再拆成多个任务，否则改变预注册问题与计算成本。

### 3.2 三臂

| 臂 | 更新定义 | 回答的问题 |
|---|---|---|
| **S：shared baseline** | `AdamW((g_s0+g_rest)/2)` | 普通共享优化器基线 |
| **T：task-state baseline** | `(Adam_s0(g_s0)+Adam_rest(g_rest))/2`；统一 lr/betas/wd | 纯任务状态隔离能否在 s0 形成 baseline win |
| **A：AdaTask case** | 与 T 状态完全相同，但 sparse-expert 的每任务预条件方向乘 task-axis AU 因子后再平均 | 真 AdaTask 在 T 之上是否有增量 |

A 臂对每个 sparse-expert 参数块 p：

\[
 f_{p,t}=\operatorname{clip}\left[
 \left(\frac{\operatorname{mean}_{t'\in\{s0,rest\}}AU_{p,t'}}{AU_{p,t}+10^{-12}}\right)^{0.5},
 \frac13,3
 \right]
\]

\[
 u_p^A=\frac12\sum_{t\in\{s0,rest\}}f_{p,t}
 \frac{\hat m_{p,t}}{\sqrt{\hat v_{p,t}}+10^{-8}}
\]

- `alpha=0.5`、`beta_AU=0.99`、clip `[1/3,3]` 启动前冻结。
- 只调 `sparse_expert`；router、head、embedding、shared backbone 的 A 更新与 T 完全相同。
- 某任务未激活某专家时，该 `(p,t)` 不更新 AU、不产生方向；AU 均值只在本 batch 对该专家有方向的任务中计算。
- A 是「弱 AU 方向增步长」的 suppress 式 case；本轮不扫 encourage/alpha/clip，避免结果后调参。

---

## 4. 启动前审计（全部 PASS 才能把 gate 改成 running）

必须生成不可覆盖的 `cache/audit/adatask_win_s0_27k_siteA/prelaunch_audit.json`，至少包含：

1. **样本计数**：原 train/valid/test 总数与 macro 场景数匹配冻结 JSON；one-vs-rest 分组不漏、不重。
2. **组梯度等价**：S 臂收集的两个梯度合成后，与直接对 `(L_s0+L_rest)/2` backward 的相对误差 ≤`5e-6`。（2026-08-18 启动前修订：原定 1e-6，实测 fp32 下 jacrev 与 autograd 的归约顺序噪声为 1.2e-6，语义错误的量级为 ≥1e-3；修订发生在任何训练 run 之前）
3. **`f=1` 哨兵**：关闭 A 因子时，A 与 T 单步参数/状态 max-abs ≤`1e-7`。
4. **post-preconditioner 哨兵**：A 与 T 的 `m/v` 在同输入下逐位一致；差异只出现在合并前的 direction multiplier。
5. **task-axis 哨兵**：人工令 `AU_s0=4×AU_rest` 时，suppress 因子满足 `f_s0<f_rest`；交换 AU 后方向交换。禁止跨专家轴求均值。
6. **角色隔离**：仅 `sparse_expert` 的 applied update 可因 f 改变；其他角色 A/T 逐位一致。
7. **参数与初值**：S/T/A 模型参数量相同；同 seed 初始 state dict hash 相同；数据生成器初始状态相同。
8. **路由语义**：top_k=2；每 epoch 记录专家覆盖、负载、task/expert AU、f 分布、clip 率。
9. **效率 smoke**：100 batch 实测外推；单 run 预算超过 12h 时停止，不以 3.9% 短训替代正式结果。

若任一 epoch 超过 30% 的有效 f 触及 clip 边界，该 run 标记 `clip_dominated=true`，只能判 INCONCLUSIVE；本轮不得看结果后改 clip 重跑。

---

## 5. 分阶段运行与冻结门控

### Stage 0：实现与 smoke

- 状态：**`done`（2026-08-18 21:50，审计 10/10 全 PASS）**
- 产物：`cache/audit/adatask_win_s0_27k_siteA/prelaunch_audit.json`（all_pass=true）、smoke run json 同目录
- 关键审计读数：组梯度等价 max_rel_err=1.20e-6（容差 5e-6，fp32 归约噪声）；f=1 哨兵 max-abs=0.0；A 步后 m/v/steps 与 T 逐位一致；非专家参数逐位一致、12 个激活专家张量受因子影响；效率 0.037 s/batch → 单臂 20 epoch ≈ 5.4h < 12h 预算
- 实现备注：S3/S4/S6 采用「同对象快照还原」法——跨模型对象的前向存在合法 fp32 逐位差异（内存对齐影响 kernel 选择），不能用于状态不变量检验
- smoke：`--max-batches 5` 全链路通过（训练→valid/test 评估→JSON 落盘；A 臂因子统计 mean_f_target=0.711 / mean_f_rest=1.911，方向符合设计）
- **断点续跑 bitwise 验证（2026-08-18 22:50，用户要求）**：`scripts/verify/verify_resume_bitwise.py`，S/T/A 三臂各对比「连续 3 mini-epoch」vs「2 epoch 存 ckpt → 新对象恢复 → 续跑第 3 epoch」，判定参数/优化器状态（S: AdamW exp_avg/exp_avg_sq/step；T/A: m/v/steps/au）/数据 generator/CPU RNG/逐步 loss 全部逐位一致。结果 **3/3 PASS**，证据 `cache/audit/adatask_win_s0_27k_siteA/resume_bitwise_audit.json`。入口 `--resume` 从每 epoch 落盘的 `ckpt_e17_*.pt` 恢复；此能力对机器迁移/抢占续跑生效，**今后所有实验入口必须内置每 epoch ckpt + 已验证的 resume 语义**。

### Stage 1：新 seed 的全数据三臂筛选

- 状态：**`done / gate=no-go`（2026-08-19 11:50 起六臂 S/T/A × 101/202 全部完成；判定 INCONCLUSIVE，不追加 303/404）**
- 判定结果（详见 [E17 Stage 1 gate 判定](./20260819-1150-E17-Stage1-gate判定-单场景AdaTask无baseline-win.md)）：
  - Δ_T(s0) 变号：101 = −0.002450，202 = +0.000170（go 条件 1 不满足）→ baseline win 不成立；
  - Δ_A(s0) 2/2 正但均值 +0.000528 < 参考地板 0.000832 → AdaTask 增量未达地板；
  - 撤销 F1 短训 +0.021/+0.042 的子采样乐观外推。
- **并行授权（用户 2026-08-18）**：seed 101（cuda:0）与 seed 202（cuda:1）同时启动，各自 S→T→A 同卡顺序执行；若 101 的 gate 判负向终止条件，则终止 202。两 seed 均属预注册 seed 集，不引入选择偏差。
- 观察哨：A 臂各 epoch 的 clip_rate 若 ≥0.3，按冻结规则该 run 标 clip_dominated（smoke 首批 0.34 为 AU 冷启动暂态，以全 epoch 均值为准）
- 只作 go/no-go，不作正式 PASS
- 继续到 Stage 2 的条件同时满足：
  - `Δ_T(s0)>0`；
  - `Δ_A(s0)>0`；
  - A−T 的 test macro 不低于 `-0.002`；
  - 三臂无 clip_dominated、无路由覆盖≤2、无 provenance 失败。
- 其他情况停止为 `INCONCLUSIVE`（若两项 s0 差均 <0 且绝对值 >0.002，登记负向筛选，但仍不写正式 FAIL）。

### Stage 2：追加 303/404，四 seed 正式判定

- 状态：`not started（Stage 1 no-go，本实验收尾）`
- 不改任何超参数；seeds 101/202 已在 Stage 1 并行完成（用户授权），原样纳入
- 正式结果必须四 seed 一起汇总，单 seed 不单独宣传为 win

---

## 6. 判定规则（启动前冻结）

### 6.1 baseline win：T vs S

- **PASS**：四个 `Δ_T(s0)` 全正，均值 `>floor_T_s0`；且 `mean[macro(T)-macro(S)]≥-floor_T_macro`。
- **FAIL**：四个 `Δ_T(s0)` 全负，绝对均值 `>floor_T_s0`。
- **INCONCLUSIVE**：其余全部情况。

### 6.2 AdaTask win-case：A vs T

- **PASS**：四个 `Δ_A(s0)` 全正，均值 `>floor_A_s0`；且 `mean[macro(A)-macro(T)]≥-floor_A_macro`；无 clip_dominated。
- **FAIL**：四个 `Δ_A(s0)` 全负，绝对均值 `>floor_A_s0`，且无审计失败。
- **INCONCLUSIVE**：其余全部情况。

只有 §6.1 与 §6.2 同时 PASS，才可写「在预注册 s0 上同时得到 baseline win 和 AdaTask 增量 win」。若只有 §6.1 PASS，只能写 baseline case；若只有 §6.2 PASS，只能写 AdaTask 相对 T 有增量，不能声称 T 相对普通优化器有效。

### 6.3 结论四元组模板

- baseline 因果断言：唯一变量=`共享 vs 两任务独立 Adam 状态`；runs=`同 seed T−S`；证据=`run_e17_{S,T}_s*.json`；规则=§6.1。
- AdaTask 因果断言：唯一变量=`sparse-expert 预条件后 task-axis AU factor`；runs=`同 seed A−T`；证据=`run_e17_{A,T}_s*.json`；规则=§6.2。

---

## 7. 产物与实现边界

| 项 | 预定路径 |
|---|---|
| 核心模块 | `adatask_win_case.py`（根目录，新增；两组梯度收集 + T/A 优化器 + task-axis 因子纯函数） |
| 入口 | `experiments/main_adatask_win_case.py`（新；S 臂复用 `SharedAdamWBatchOptimizer`；不修改历史 `main_adatask.py` 与 `main_task_role_optimizer.py`） |
| 优化器依赖 | 复用 `task_role_optimizer.py` 的 `ParameterRoleRegistry`/`SharedAdamWBatchOptimizer`，不改动该文件 |
| runner | `scripts/matrix/run_adatask_win_case.py` |
| 审计 | `scripts/verify/verify_adatask_win_case.py`（九项哨兵）；`scripts/verify/verify_resume_bitwise.py`（断点续跑 bitwise，3/3 PASS） |
| 汇总 | `scripts/summarize/summarize_adatask_win_case.py`（Stage 1 出结果后实现），阈值硬编码 |
| raw JSON | `cache/adatask_win_s0_27k_siteA/run_e17_{S,T,A}_s{seed}.json` |
| 判定 | `cache/adatask_win_s0_27k_siteA/e17_decision.json` |
| 结果 CSV | `results/adatask_win_case/`；完成后登记 `results/INDEX.md` |

禁止项：

1. 不以 3.9%/5 epoch 再次替代全数据结论；
2. 不把 site B `role_isolated` bundle 的 `-0.009116`归因到纯状态隔离；
3. 不把 s0 换成新结果里最好的场景；
4. 不用 seeds 42/123 作为正式验证 seed；
5. 不混用 pooled 与 s0/macro Δ；
6. 不启动 F2/E12/E13/E14 抢占本实验设备，直到 E17 Stage 1 给出 go/no-go。

---

## 8. 当前任务队列

1. **立即**：修订 F1/INDEX/NEXT 的过强路线关闭措辞，标 E16 `superseded`。
2. **实现**：two-group gradient collector + task-axis post-direction AdaTask 子类 + 三个数值哨兵。
3. **preflight**：site A 数据/参数/数值环境检查 + Stage 0 审计落盘。
4. **优先长跑**：seed101 的 S/T/A 全数据三臂；得到 gate 后再决定是否扩四 seed。
5. **后置**：E17 完成后再恢复 F2 EWC；E12/E13/E14 重新排期。
