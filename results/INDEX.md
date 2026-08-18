# 结果大清单（Results Index）

> 本清单是**结果文件 → 实验 → 判定 → 结论**的集中索引，10 秒看懂全局。
> 判定与一句话结论**直接取自** [`docs/DRIVERS.md`](../docs/DRIVERS.md) 路线表与各结论文档，本清单不重算任何指标。
> 实时证据（json/判定）在 `cache/`，历史文档在 `docs/archive/`。

---

## 0. 速览（当前状态）

**唯一活跃路线**：27K 数据上的 scenario-routed MoE。**最终形态 = 硬路由 top_k=2**。

- **E7（PASS）**：软路由 MoE 相对 dense 稳定胜出，Δ=+0.001881（地板 1.39×），4/4 seed 正。首个 work case。
- **E10（PASS）**：**硬路由 top_k=2，Δ=+0.005012（地板 3.70×）**，4/4 seed 正，是软路由的 2.7 倍——**当前最优**（macro 0.735388）。稀疏不损失收益（wall 0.98×，尚未省算力）。
- **E11（PASS）**：**加回 5.5 亿 ID 参数后 Δ=+0.004329（地板 3.24×）**，4/4 seed 正 → **公平性质疑排除**。副产结论：加回 ID 表 macro 降 **0.0055**（4/4 负）→ **ID 表有害**。
- **E8（INCONCLUSIVE）**：pooled 训练下 5/6 seed 正、1 微负；绝对成绩 moe+pooled 最高，实用推荐 pooled 训练。
- **E9（撤销）**：K 扫描 20 run 按用户授权缩减不执行（top_k 维度比 K 粒度重要）。
- **突破归因**：换 27K（地板 ↓8.4×）+ 换 macro 端点（对准收益位置）+ 硬路由（放大 2.7×），**三者缺一不可**；删 ID 表只是工程使能条件。见 [`突破归因`](../docs/20260817-1208-突破归因-公平比较下的正面结果.md)。
- **E15（PASS ×2）**：换一套 epoch 选择规则（macro→pooled），MoE 收益**方向不变**（4/4 同号正），但地板倍数 3.70→**1.05**——方向稳健、统计强度不稳健；复现哨兵 8/8 逐位一致。见 [`E15 结论`](../docs/20260817-1500-E15结论-选择端点敏感性.md)。
- **下一步（最高优先，planned）**：E17 先在预注册场景 s0 建立纯 task-state baseline win，再测一个真 task-axis、post-preconditioner AdaTask 增量 case；Stage 0 审计未通过前禁止长跑。见 [`E17 复审与预注册`](../docs/20260818-1815-结论复审与E17单场景AdaTask-win-case预注册.md)。E12/E13/E14 与 F2 暂后置。
- **site B 分支（长程已完成，单 seed）**：`site-b/task-role-optimizer` 的完整角色隔离 bundle 在全数据 seed42 上为 `0.723993-0.733109=-0.009116`；raw JSON 在提交 `842b8aa`。该 bundle 同时改变状态、角色 LR、router 动量和 weight decay，故只能登记 **INCONCLUSIVE（负向单-seed）**，不能归因或关闭全部状态隔离路线。
- **F1 优化器状态共享度 baseline（探索，严格判定 INCONCLUSIVE）**：DualOptim 两 seed 异号；DualOptim+ 是强负向预警但仍属 3.9%×5 epoch；`role_isolated` 两 seed 异号。s0 上纯状态隔离已有 `+0.020936/+0.041955` 的候选信号，E17 用新 seeds 前瞻复核。历史 AdaTask 是 pre-Adam、跨专家调制，真 task-axis AdaTask 尚未测试。
- 证据：`cache/macro_auc_27k/`；结论 [`E8910`](../docs/20260816-1950-E8910结论-硬路由topk2最终形态.md)、[`E7`](../docs/20260816-1237-E7结论-27K场景内MoE首个work-case.md)

**三条已关闭的路线**（一句话）：

| 路线 | 为何关闭 |
|---|---|
| cross 层容量 / capacity-MoE | 4× 容量收益 ≈ 0 + 稀疏代价 ≈ −0.001，dense-widened 独立证伪 |
| embedding 容量 | 84M ID 表是死重（E1：删掉 AUC 不变） |
| 下游留出域迁移 | G1 FAIL，moe-router 1/3 target 胜 |

**核心教训**（写进所有后续设计）：① 加容量前先证伪"容量缺口"；② pooled AUC 测不到 MoE 收益（macro 的 Δ 是 pooled 的 2–7 倍）；③ 1K 小场景只有 70 个正样本，测不出任何效应，需 27K。

---

## 1. 结果文件索引（按实验分组）

### 活跃路线

| 实验 | 状态 | 结果文件 | 判定 | 一句话结论 | doc |
|---|---|---|---|---|---|
| **E7 场景内泛化 MoE（27K）** | **活跃** | `cache/macro_auc_27k/`（json 证据，18 run） | **PASS** | Δ_moe +0.0019，4/4 seed 正，首个 work case | [`E7 结论`](../docs/20260816-1237-E7结论-27K场景内MoE首个work-case.md) |
| **E10 硬路由 top_k=2（27K）** | **活跃（最优）** | `cache/macro_auc_27k/run_s6sparse_*.json` | **PASS** | Δ=+0.005012（地板 3.70×），4/4 正，macro 0.735388 | [`E8910 结论`](../docs/20260816-1950-E8910结论-硬路由topk2最终形态.md) |
| **E11 full-ID 公平复核（27K）** | **活跃** | `cache/macro_auc_27k/run_s8full_*.json`（8 run） | **PASS** | 加回 551M ID 后 Δ=+0.004329（4/4 正）→ 公平性排除；ID 表**有害** −0.0055 | [`突破归因`](../docs/20260817-1208-突破归因-公平比较下的正面结果.md) |
| **E15 选择端点敏感性（27K）** | **活跃** | `cache/macro_auc_27k/run_e15_*.json`（8 run） | **PASS ×2** | 换 pooled 选 epoch 后 Δ 方向不变（4/4），但地板倍数 3.70→1.05；复现哨兵 8/8 逐位一致 | [`E15 结论`](../docs/20260817-1500-E15结论-选择端点敏感性.md) |
| **E8 pooled 组合（27K）** | 已复盘 | `cache/macro_auc_27k/run_s7pool_*.json` | INCONCLUSIVE | 5/6 正、绝对成绩最高，实用推荐 pooled 训练 | [`E8910 结论`](../docs/20260816-1950-E8910结论-硬路由topk2最终形态.md) |
| E5 场景内泛化 MoE（1K） | 已复盘 | `cache/macro_auc/`（json 证据，58 run） | INCONCLUSIVE | 1K 测不出（小场景 70 正样本），换 27K 后成立 | [`E5/E6 结论`](../docs/20260815-1508-E5E6结论-场景内泛化MoE与隔离上界.md) |

### 已关闭路线

| 实验 | 状态 | 结果文件 | 判定 | 一句话结论 | doc |
|---|---|---|---|---|---|
| **capacity-MoE（cross 层）** | 关闭 | `results/capacity_moe/`（57） | INCONCLUSIVE + FAIL | 容量收益≈0 + 稀疏代价≈−0.001 | [`根因定位`](../docs/20260814-2111-专家无收益根因定位与决定性容量实验.md) |
| **dense-widened 4×** | 关闭 | `results/dense_widened/`（8） | FAIL | 4× 容量收益≈0，证伪"容量瓶颈" | [`同上 §六.5`](../docs/20260814-2111-专家无收益根因定位与决定性容量实验.md) |
| **下游迁移** | 关闭 | `results/downstream/`（18） | G1 FAIL | moe-router 仅 1/3 target 胜，停止 | [`G1 结论`](../docs/archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md) |
| **Stage B same-FLOPs MoE** | 关闭 | `results/stage_b/`（9） | FAIL | 同 seed 差 [-,-,+]，wall-clock 更慢 | [`StageB 结论`](../docs/archive/conclusions/20260812-0103-StageB-MoE-九次训练结论.md) |
| **TCMR 静态任务路由** | 关闭 | `results/tcmr/`（1） | INCONCLUSIVE | 跨 seed 变号、未越噪声地板 | [`TCMR 结论`](../docs/archive/conclusions/20260812-1703-TCMR-结论.md) |
| **共享残差持续学习** | 关闭 | `results/shared_residual/`（1） | G3 INCONCLUSIVE | specialist screen 无 winner，gate=false | [`G3 结论`](../docs/archive/conclusions/20260812-1956-共享残差specialist-screen-INCONCLUSIVE结论.md) |

### 已完成 / 已降格（结论已固化，不再演进）

| 实验 | 状态 | 结果文件 | 判定 | 一句话结论 | doc |
|---|---|---|---|---|---|
| **E1 ID embedding 死重** | 已完成 | `results/field_ablation/`（6） | **PASS** | 84M ID 表无净贡献，参数 −99.31% | [`E1 结论`](../docs/20260814-2225-E1结论-ID-embedding死重确认.md) |
| **按场景 sample weighting** | 已完成 | `results/sample_weighting/`（12） | **PASS** | 样本加权恢复 full-batch 目标，R_gain≈0.999 | [`swg 结论`](../docs/archive/conclusions/20260811-2242-按场景训练代价消除结论.md) |
| **MoE v2 系列** | 降格 | `results/moe_v2/`（30） | 见 Stage B | 早期 v2 探索，多数被 Stage B 结论覆盖 | [`StageB 结论`](../docs/archive/conclusions/20260812-0103-StageB-MoE-九次训练结论.md) |
| **早期 MoE 探索（k1/k2/k4/k8/d10-d20）** | 归档 | `results/moe_exploration/`（47） | superseded | 旧网格关闭，单 seed 不作强结论 | [`阶段结论`](../docs/archive/conclusions/20260811-1450-子任务调制交叉网格阶段结论.md) |
| **vanilla dense baseline** | 归档 | `results/vanilla_baseline/`（3） | baseline | dense 从零预训练与逐场景 eval | — |
| **main 入口产物** | 归档 | `results/main_pretrain/`（1） | baseline | 早期 `main.py` 三级下游的历史产物 | — |

---

## 2. 重要提醒（口径，读数字前必看）

1. **跨场景只比较相对排序**，绝对 AUC 不可跨 scenario 比较。
2. **跨数据集绝对 AUC 不可比**（1K 的 0.73 与 27K 的 0.73 数据分布不同）——只信**同 seed 配对差**。
3. **macro AUC 与 pooled AUC 是不同口径**，不得混入同一张表。
4. **单 seed 只作动机**，不进高置信结论；跨 seed 变号不得用均值覆盖。
5. 旧产物可能含未冻结 embedding 的过拟合（修复前口径），引用时注意。

---

## 3. 目录速查

- 结果 csv：`results/<实验>/`
- 证据 json / 判定：`cache/<实验>/`（`macro_auc`=1K，`macro_auc_27k`=27K）
- 权重 `.pt`：`cache/` 顶层（gitignore，未纳入版本管理）
- 日志：`logs/`
- 结论文档：`docs/`（近期）+ `docs/archive/conclusions/`（历史）
