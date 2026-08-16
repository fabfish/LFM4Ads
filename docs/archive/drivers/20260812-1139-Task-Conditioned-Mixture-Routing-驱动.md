# Task-Conditioned Mixture Routing（TCMR）实验驱动

- 创建时间：2026-08-12 11:39
- 当前状态：**`auth=not-started`**
- 当前允许范围：代码审查、CPU smoke、文档与计划；**禁止在本轮启动十五次长程训练**
- 上游证据：`docs/20260812-0103-StageB-MoE-九次训练结论.md`
- 总驱动：`docs/DRIVERS.md`

## 一、命名约定（强制）

新实验禁止使用 `stgc`、`dt`、`fu` 等无法从名称直接还原语义的简单代号。

本文只使用以下有明确英文展开的缩写：

- **TCMR**：Task-Conditioned Mixture Routing，任务条件混合路由；
- **FUR**：Frozen Uniform Routing，冻结均匀路由；
- **DOR**：Data-Only Routing，仅数据路由；
- **TOR**：Task-Only Routing，仅任务标识路由；
- **DATR**：Data-and-Task Routing，数据与任务联合路由；
- **DATCR**：Data-and-Task Consistency Routing，带一致性约束的数据与任务联合路由；
- **MoE**：Mixture of Experts，混合专家；
- **AUC**：Area Under the Receiver Operating Characteristic Curve。

缩写只用于正文表格。脚本参数、run name、日志、manifest、summary、checkpoint 和 CSV
一律使用完整的描述性英文名称。例如：

`task-conditioned-mixture-routing-data-and-task-routing-seed-42`

历史 `swg-*` / `stgb-*` 只作为不可改写的旧产物主键，不得复制到新实验。

## 二、研究问题与证据边界

### 2.1 已冻结事实

1. sample weighting 修复了按场景等权导致的训练目标失配。三种子精确口径为：
   - `R_gain≈0.999`；
   - 同 seed 配对 `R_resid≈0.992`；
   - 两者公式和含义不同，禁止合并成单一恢复率。
2. 低秩全维 MoE 的 frozen / soft 路由相对 same-FLOPs dense 均跨 seed 变号，
   竞争性主张 FAIL；只能称“未稳定超过”，不能称“匹配”或“等价”。
3. soft routing 学到了非均匀 gate，但没有转化为 AUC 收益；manifest 时间显示
   MoE 慢于 dense，same-latency 不成立。
4. sparse scaling、真实稀疏 FLOPs、AdaTask 更新机制、持续学习与 BWT 结论保持 blocked。

### 2.2 本关卡只回答

在相同低秩全维专家、相同初始化、相同数据顺序与相同训练预算下，显式加入任务标识
是否能让 DATR 相对 FUR 和 DOR 产生跨三 seed 方向一致的 pooled AUC 改进，同时不引发
路由塌缩或 macro / 逐场景系统性恶化。

本关卡是**静态多任务路由实验**，不是 AdaTask 更新实验，也不是持续学习实验。

## 三、实现冻结

### 3.1 模型

- 主体：`model.py::DCNv2MoE_TaskConditionedLowRank`；
- 每个交叉层：4 个低秩全维专家，`dim=360`、`rank=45`；
- 路由器：`TaskConditionedRouter`，每种模式均构造 data projection 与 task embedding，
  再按模式冻结不用的路径；这保证模型构造消耗相同随机数序列；
- 历史 `DCNv2MoE_LowRank` 不修改语义，旧 checkpoint 保持兼容；
- 每个 run 在训练前记录排除 router 后的 `shared_initialization_sha256`；同 seed 五变体
  hash 不一致时，汇总器必须判 `blocked`。

### 3.2 五种路由

| 缩写 | `--routing-mode` 完整值 | 可训练输入 | 主/次级角色 |
|---|---|---|---|
| FUR | `frozen-uniform-routing` | 无；恒 `1/K` | 主锚点 |
| DOR | `data-only-routing` | 数据表示 | 主锚点 |
| TOR | `task-only-routing` | `tab` 任务标识 | 次级消融 |
| DATR | `data-and-task-routing` | 数据 logits + 任务 logits | **预注册主候选** |
| DATCR | `data-and-task-consistency-routing` | DATR + 对称 KL | 次级机制分析 |

DATCR 的一致性项冻结为：

\[
L_{consistency}=\frac{1}{2}\left[KL(p_{data}\Vert p_{data+task})+
KL(p_{data+task}\Vert p_{data})\right],\qquad \lambda=0.01
\]

基础 BCE 与一致性项都按 `|B_task|/|B|` 加权。不得看结果后修改 \(\lambda\)。

### 3.3 训练协议

| 项 | 冻结值 |
|---|---|
| device | `cuda:0`（默认）；允许 `--device cuda:0 cuda:1` 多卡并行 |
| seeds | 42 / 123 / 456，同 seed 配对 |
| batch size | 10000 |
| learning rate | 1e-3 |
| Adam beta2 | 0.999 |
| shuffle | `True` |
| maximum epochs | 300 |
| early stop | valid AUC 提升阈值 0.001，与现有上游协议一致 |
| scenario loss weighting | `sample` |
| optimizer step | 每个原始 batch 恰好一次 |
| training task scope | `Split("all")` 训练集全部 15 个 `tab` |
| formal report scope | `[0,1,2,3,4,5,6,8]` 共 8 个场景 |

8 场景 macro AUC 不得表述为全部 15 个训练任务的 macro AUC。

## 四、十五次矩阵与不可变 run name

每个变体按 seed 42、123、456 串行执行：

1. `task-conditioned-mixture-routing-frozen-uniform-routing-seed-42`
2. `task-conditioned-mixture-routing-frozen-uniform-routing-seed-123`
3. `task-conditioned-mixture-routing-frozen-uniform-routing-seed-456`
4. `task-conditioned-mixture-routing-data-only-routing-seed-42`
5. `task-conditioned-mixture-routing-data-only-routing-seed-123`
6. `task-conditioned-mixture-routing-data-only-routing-seed-456`
7. `task-conditioned-mixture-routing-task-only-routing-seed-42`
8. `task-conditioned-mixture-routing-task-only-routing-seed-123`
9. `task-conditioned-mixture-routing-task-only-routing-seed-456`
10. `task-conditioned-mixture-routing-data-and-task-routing-seed-42`
11. `task-conditioned-mixture-routing-data-and-task-routing-seed-123`
12. `task-conditioned-mixture-routing-data-and-task-routing-seed-456`
13. `task-conditioned-mixture-routing-data-and-task-consistency-routing-seed-42`
14. `task-conditioned-mixture-routing-data-and-task-consistency-routing-seed-123`
15. `task-conditioned-mixture-routing-data-and-task-consistency-routing-seed-456`

任何 run directory、manifest、日志或结果已存在时必须停止，禁止覆盖或自动删除后重跑。

## 五、指标与流式诊断

### 5.1 精度

- pooled AUC：主指标；
- 8 个正式报告场景的 macro AUC：保护指标；
- 8 个场景逐场景 AUC：保护指标。

### 5.2 路由

每层流式累计，不保存样本级 gate：

- mean gate / expert load；
- mean routing entropy；
- `soft_task_expert_mutual_information_nats`：以 `one_hot(task)^T @ soft_gate` 构造
  soft 路由期望联合分布，使用自然对数的 plug-in mutual information，单位 nats；
- `hard_argmax_route_churn`：固定 validation prefix 上相邻 epoch 的 argmax expert 变化率。

互信息不是 hard-route MI，也不是连续 gate 向量 MI；churn 不是历史文档中的 JSD 口径，禁止混称。

路由塌缩预注册定义：任一层满足以下任一条件：

- mean entropy `< 0.25 × ln(K)`；
- 最大 mean gate `> 0.95`；
- 最小 mean gate `< 0.01`。

FUR 的确定性均匀路由不按“缺少分工”判塌缩；上述塌缩保护用于可训练路由变体。

## 六、关卡判定（不得事后降级）

### 6.1 主比较

只允许以下两项决定晋级：

1. DATR − FUR 的同 seed pooled AUC 配对差；
2. DATR − DOR 的同 seed pooled AUC 配对差。

TOR 和 DATCR 只作次级解释，不能替换失败的主终点。

### 6.2 四态规则

- **PASS（仅探索关卡解锁）**：两项主比较在三个 seed 均严格为正；所有可训练路由不塌缩；macro AUC
  不对任一锚点三 seed 全负；不存在某正式报告场景对锚点三 seed 全负且平均下降至少 0.001。
  该 PASS 只允许规划后续机制关卡，不自动支持“稳定改进”论文主张。
- **INCONCLUSIVE**：任一主比较在三 seed 跨正负变号，或没有达到三 seed 严格为正且
  又不满足明确 FAIL。
- **FAIL**：任一锚点比较在三个 seed 全部 `≤0`；或主比较虽全正但触发路由塌缩、
  macro 系统性恶化、逐场景系统性恶化保护。
- **blocked**：不变量失败、共享初始化 hash 不一致、产物覆盖风险、运行异常或证据不全。

均值不能覆盖逐 seed 方向；不得追加少量 seed 重开 sparse scaling 或持续学习叙事。

### 6.3 稳定改进主张的额外门槛

总驱动的通用噪声规则继续有效。即使探索关卡 PASS，只有当 DATR 对两个锚点的平均同 seed
配对差都超过对应锚点三 seed AUC 极差时，才可将 `stable_improvement_claim` 标为
`SUPPORTED`；否则必须标为 `NOT_SUPPORTED`。汇总器同时输出探索关卡 verdict 与该强主张状态，
禁止二选一或事后改口径。

## 七、交付给执行模型的机械步骤

执行模型不得修改冻结协议，只按顺序执行：

### 7.1 阅读和静态检查

```bash
python -m py_compile model.py train.py task_conditioned_mixture_routing_protocol.py \
  run_task_conditioned_mixture_routing.py \
  scripts/verify/verify_task_conditioned_mixture_routing.py \
  scripts/matrix/run_task_conditioned_mixture_routing_matrix.py \
  scripts/summarize/summarize_task_conditioned_mixture_routing.py
python -c "import torch, torcheval; print(torch.__version__, torcheval.__version__)"
```

当前交付环境的静态编译已通过；纯 CPU smoke 已通过五种路由语义、路径隔离、对称 KL 梯度、
sample-weighted 单 optimizer step 和共享专家初始化 hash 检查；矩阵计划数量已核对为 15。
当前 shell 缺少 `torcheval`，所以正式训练运行时前置检查尚未满足。执行模型必须进入项目原有完整训练环境
或补齐项目依赖；不得把依赖错误改写为实验 FAIL。

### 7.2 只展示计划，不训练

```bash
python scripts/matrix/run_task_conditioned_mixture_routing_matrix.py plan --device cuda:0
```

必须人工核对输出恰好 15 个完整描述性 run name，且顺序与第四节一致。

### 7.3 正式前置门控

```bash
python scripts/verify/verify_task_conditioned_mixture_routing.py --write-report
```

预期产物：
`cache/audit/task_conditioned_mixture_routing/task_conditioned_mixture_routing_invariants.json`。
状态不是 `pass` 时将本关卡置为 `blocked`，禁止训练。报告不可覆盖；矩阵执行器若发现报告已存在，
只读取并验证其中的源码和驱动 SHA-256，不会重复写；报告不存在时才自动生成一次。

### 7.4 机器授权与十五次并行训练（按 seed 组分卡）

当前 `scripts/task_conditioned_mixture_routing_authorization.json` 在收到用户明确指令后改为：

- `status="authorized"`；
- `driver_path` 精确指向本文档；
- `driver_sha256` 等于本文档当时 SHA-256；
- `authorized_scope="fifteen-trial-multi-device-seed-group-parallel"`。

矩阵脚本会同时检查显式命令行确认和机器授权；任一缺失都拒绝训练：

```bash
python scripts/matrix/run_task_conditioned_mixture_routing_matrix.py execute \
  --device cuda:0 cuda:1 --acknowledge-document-driver
```

并行规则（用户 2026-08-12 放宽，已同步 AGENTS.md §二）：

- **同一 seed 的 5 个路由模式必须跑在同一张卡上**，保证 DATR 对该 seed 的 FUR/DOR 配对差不引入设备混杂因子；
- 不同 seed 组分布到不同设备并发（runner 通过 `seed_device_map` 固定分配）；
- 禁止把同一 seed 的不同路由模式拆到不同设备；
- 禁止直接调用 `run_task_conditioned_mixture_routing.py`；单次 runner 只接受已授权矩阵传递的执行上下文；
- 任一 trial 非零退出立即停止整个矩阵；
- 禁止删除失败 manifest 后静默重跑；
- 禁止边跑边改 `model.py`、`train.py`、runner 或判定阈值。

### 7.5 汇总

```bash
python scripts/summarize/summarize_task_conditioned_mixture_routing.py --write-results
```

若不足 15/15，汇总器只能输出 `not-started/running` 与 `INCONCLUSIVE`，禁止写结论。
若 15/15 完整，输出：

- `result_task_conditioned_mixture_routing.csv`；
- `cache/task_conditioned_mixture_routing/gate_decision.json`。

随后新建实际时间命名的 TCMR 结论文档，逐 seed 回填主比较和保护指标，并更新
`docs/DRIVERS.md`；不得在结论文档中引入本驱动未定义的简单代号。

## 八、产物结构

```text
cache/task_conditioned_mixture_routing/
└── task-conditioned-mixture-routing-<full-routing-name>-seed-<seed>/
    ├── manifest.json
    ├── summary.json
    ├── trial_result.csv
    └── model_state.pt

cache/manifests/task_conditioned_mixture_routing/
└── matrix_state.json

logs/
└── task-conditioned-mixture-routing-<full-routing-name>-seed-<seed>.log
```

每个 manifest 必须含配置、训练 15 个 tab 的样本计数、8 场景报告范围、git commit、
dataset SHA-256、源文件 SHA-256、共享初始化 SHA-256 和不可变产物路径。

## 九、本轮明确不做

- 不启动十五次正式训练；
- 不重跑或改写 Stage A/B 历史产物；
- 不启用旧 AdaTask AU 梯度调制；
- 不实现 task-wise optimizer state；
- 不做持续学习、BWT、任务顺序扫描；
- 不做 top-k 或真实 sparse dispatch；
- 不声称 same-latency、稀疏 FLOPs 或部署效率收益。
