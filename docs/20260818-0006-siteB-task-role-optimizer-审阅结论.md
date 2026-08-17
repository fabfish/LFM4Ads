# site B 分支审阅：按参数角色隔离子任务优化器（task-role-optimizer）

- 创建时间：2026-08-18 00:06
- 类型：**代码审阅结论（site A 作为合并审阅方）**
- 审阅对象：`origin/site-b/task-role-optimizer`（分叉自 `41cee0a`，21 个提交，最新 `08f7b71`）
- 上游：[两端协作分离设计](./20260817-1400-两端协作分离设计与实验分工.md)、
  [site B 预注册](./20260817-1821-按参数角色隔离子任务优化器预注册-siteB.md)（在分支上）
- 审阅范围：`task_role_optimizer.py`（988 行）、`task_role_optimizer_protocol.py`、
  `experiments/main_task_role_optimizer.py`（444 行）及 matrix/summarize/verify 脚本

---

## 一、结论一句话

**实现语义正确、站得住脚，审阅通过核心正确性；但有两处需在合并前登记的边界，且当前短测证据仅"值得复核"、尚不足以作正式判定。**

## 二、审阅要点（逐项）

### 2.1 核心数学：这是「真 per-task 自适应学习率」，不是梯度缩放的伪实现

`TaskRoleOptimizer.step`（`task_role_optimizer.py:359-448`）的更新语义，与历史 AdaTask（已被证伪的"梯度乘因子"）**有本质区别**：

| 优化器 | 数学 | 说明 |
|---|---|---|
| 对照 `SharedAdamWBatchOptimizer` | \(\mathrm{AdamW}(\mathrm{mean}_s\, g_s)\) | 先对各场景梯度求平均，再进一个共享 AdamW |
| 实验 `TaskRoleOptimizer` | \(\mathrm{mean}_s\,\mathrm{AdamW}_s(g_s)\) | **每个场景用自己的独立一阶/二阶矩算方向，再对各场景方向求平均** |

代码实证（`step` 内 `accumulated.add_(direction, alpha=1.0/task_count)`，每个 `direction` 来自 `_dense_direction(spec, item.task_id, ...)` 的**每场景独立状态**）。这与 AdaTask / MultiAdam 的原始语义一致：梯度大的场景二阶矩大 → 其方向被自身二阶矩缩小 → 各场景更新量更均衡 → 小场景贡献相对放大。**这是"任务自适应学习率"的正确实现，不再是恒等式抵消的梯度缩放。**（对比：[E16 预注册 §1](./20260817-1410-E16-AdaTask对照预注册.md) 用 200 步数值实验证明了历史实现的空操作性。）

### 2.2 参数角色划分：正确、完备、互斥

`ParameterRoleRegistry.from_model`（`task_role_optimizer.py:120-165`）用正则把每个可训练参数映射到 6 个角色，`assert_complete`（`167-179`）硬校验"完备 + 互斥"。经逐项比对 `model.py` 实际命名，**全部一致**：

| site B 正则 | model.py 实际结构 | 角色 |
|---|---|---|
| `sparse\.tables\.(.+)\.weight` | `Sparse/SubsetSparse.tables = ModuleDict(field→Embedding)` | SHARED_EMBEDDING |
| `cross_layers\.(\d+)\.experts\.(\d+)\.` | `CrossExpertLayer.experts = ModuleList(Linear)` | SPARSE_EXPERT |
| `cross_layers\.(\d+)\.router\.embed\.weight` | `ScenarioRouter.embed = Embedding` | ROUTER |
| `cross_layers\.(\d+)\.shared\.` | （model.py 无此结构） | ALWAYS_ON_SHARED_EXPERT（**空，预留**） |
| `dnn\.` | `DCNv2MoE.dnn = ModuleList(Linear×2)` | SHARED_BACKBONE |
| `head\.` | `DCNv2MoE.head = Linear(dim,15)` | TASK_HEAD |

### 2.3 row_sparse 语义：正确区分了两种"行"

`head` / `router` 标 `row_sparse=True`（行 = 场景 id，共 15 行），`sparse.tables.*` 也标 `row_sparse=True`（行 = 特征 id）。`_row_direction` 用 `item.active_rows[spec.name]` 取活跃行、用 `(task_id, row)` 索引状态，两种语义都被正确处理。`head` 按场景行独立状态尤其精确——每个场景只回传 `head.weight[s]` 那一行。

### 2.4 角色差异化配置：有据、且被短测部分验证

`default_role_hyperparameters`（`85-108`）：expert lr 0.001、router = 0.05×lr 且 `use_first_moment=False`（RMSProp 风格，避免动量干扰离散路由，符合 AdaTask 设计）、shared backbone = 1.0×lr、head weight_decay=0。短测已扫 expert lr {0.0002, 0.0005, 0.001}、router ratio {0.02, 0.05, 0.1}、shared ratio {0.5, 1.0}，选出"0.001 / 0.05 / 1.0"为最佳非坍缩配置。

---

## 三、必须在合并前登记的三处边界

### 3.1 full-ID 模型不可用此优化器（显存上界）

`_new_state`（V1 `_new_state` / V2 `moment_shape`）按 `num_tasks=15` 维度展开**每个参数**的完整状态。轻量模型（869,525 参数）下：非 sparse 参数 551,115 × 15 × 2（一阶+二阶）× 4B ≈ 66MB，可接受；但 full-ID（551M 参数）会达到 **~66GB 状态**，直接 OOM。**故该优化器当前只适用于轻量模型**，full-ID 公平复核（E11 口径）无法用它做。

### 3.2 「SHARED_EMBEDDING」命名有误导性

embedding 表的**参数**是各场景共享的（没错），但它的**优化器状态**按 `(场景, 特征id)` 隔离（`row_sparse` + `(task_id, row)` 索引）。角色名 `SHARED_EMBEDDING` 容易让人误以为"embedding 完全共享、无场景隔离"。**合并时应在文档澄清：参数共享、状态按场景隔离。**

### 3.3 实验臂 vs 对照臂是多变量混合（消融存在，但主候选不是纯单变量）

主候选"完整角色隔离"相对对照"共享普通状态"同时改变了：① 场景状态隔离、② 角色差异化 lr、③ router 无动量（RMSProp）、④ head/router 无 weight_decay。**这不是"唯一变量"**。但 site B 短测里保留了"场景独立但角色同规则"臂（`task_state_uniform`，只隔离状态、角色同 lr），可单独归因状态隔离；其余三个变量尚未逐一消融。合并时须在结论中显式标注，不得表述为"单一变量"。

---

## 四、当前进展与证据强度

| 项 | 状态 |
|---|---|
| 短测（9 方向，单 seed 42，3 epoch，~3.9% 数据，valid） | **完成**，方向性正向：完整角色隔离 +0.0150（5/5 专家覆盖、负载比 1.5） |
| 三臂长程（seed 42，共享基线 / 完整角色隔离 / 冻结路由器诊断，三卡 H20） | **进行中，无结果提交**（最新提交 `08f7b71` 仍是"启动长程"） |
| 正式判定（4 seed + test + 噪声地板） | **未做**——site B 自己明确短测"只能写值得复核，不能下正式结论" |

**证据强度结论**：当前短测是**单 seed + 短训 + valid**，按仓库判定标准（4 seed + test + 噪声地板）**尚不能作正式 PASS/FAIL**。这正是留待补齐的缺口。

---

## 五、合并风险（重要）

site B 分叉点 `41cee0a` 在我 E15 收尾提交**之前**，故其分支相对 main 会把以下视为"删除"或"回退"：

- `cache/macro_auc_27k/run_e15_*.json`（8 个）—— 分支上显示为 `D`
- `docs/20260817-1500-E15结论-选择端点敏感性.md` —— 分支上显示为 `D`
- 突破归因 / 两端协作 / 交接文档 / E15 预注册 / INDEX / AGENTS / .gitignore / HANDOFF —— 双方各自基于旧点修改，**合并必须逐文件解决语义冲突，不得静默覆盖任一方**。

---

## 六、建议的后续（按依赖顺序）

1. **等 site B 三臂长程结果提交**（若单 seed 长程为负，4 seed 价值存疑；若为正，4 seed 才有意义）。
2. **合并**：以"审阅通过 + 三处边界登记"为前提，逐文件解决 §五 的冲突；E15 证据与结论必须保留。
3. **4 seed 正式判定**（site A 的 2 卡空闲，可并行）：复用已合并的 `task_role_optimizer.py`，跑"完整角色隔离主候选"的 4 seed + test macro + 噪声地板，把"值得复核"升级为登记的 `PASS/INCONCLUSIVE/FAIL`。
4. 登记 full-ID 限制与 SHARED_EMBEDDING 语义澄清进 `results/INDEX.md` 与实现文档。
