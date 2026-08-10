# 实验驱动表（描述性命名，禁止 D 加数字等无全称代号）

> 本表记录当前有效实验。旧文档已整体移入 `docs/archive/`，不再引用任何旧代号。
> 命名规则：脚本 / 产物 / 实验名一律用描述性全称；模型变体称「全路由混合专家模型 / 部分路由加共享专家模型 / 基准稠密模型」，不称 V1/V2。

## 一、从零预训练与上游改进判定

- **实验名**：混合专家从零预训练与上游改进判定
- **脚本**：`run_moe_pretrain_from_scratch.py`
- **三类 backbone（均随机初始化，不加载任何预训练权重）**：
  - 全路由混合专家模型（`model.py` 中 `DCNv2MoE`，K=4，数据级路由，top_k=K 全软）
  - 部分路由加共享专家模型（`model.py` 中 `DCNv2MoE_V2`，K=4，1 共享 + 4 路由，top_k=K 全软；**修正后路由器在全软模式下也激活**）
  - 基准稠密模型（`model.py` 中 `DCNv2`）——**复用仓库已有 `cache/dcnv2_vanilla.pt`，不重新从零训练**
- **产物**：`cache/moe_fully_routed_seed{seed}.pt` / `moe_partial_shared_seed{seed}.pt`；`result_moe_fully_routed.csv` / `result_moe_partial_shared.csv`；`cache/moe_pretrain_summary_*.json`
- **上游判定**：vanilla 复评 + 两类 MoE 的 pooled / 按场景 AUC 对比（结论见结果文档）
- **状态**：✅ 已完成（seed 42，单种子）

## 二、子任务 路由网络 / 专家 促进-抑制 调制（核心 10 配置）

- **实验名**：子任务路由网络与专家促进抑制调制观察
- **脚本**：`run_moe_subtask_modulation.py`
- **旋钮**：`--target router|expert` × `--direction 0|+1|-1`（0=不调制基线 / +1=促进 / -1=抑制）
- **核心 10 配置** = 2 模型 ×（2 调制目标 × {促进, 抑制}）+ 2 基线 = 8 调制 + 2 控制
- **产物**：`result_moe_subtask_modulation.csv`；`cache/subtask_modulation_{model}_{target}_{direction}_seed{seed}.json`
- **状态**：✅ 已完成（seed 42，单种子）

## 三、条件扩展（待启动，按信号条件执行）

- **实验名**：显著单元多种子验证 与 论文三种表征用法扩展
- **内容**：对显著单元（当前最强信号为「抑制路由器」对两类模型均提升 pooled AUC）补多种子验证；按需扩展到论文三种表征用法（特征级 `FeatureUsage` / 模块级 `ModuleUsage` / 模型级 `ModelUsage`）
- **状态**：⬜ 待启动（需先确定范围：多种子数量、是否做表征扩展）

## 测量口径（通用）

- 种子：当前均用 seed 42（单种子广扫；用户要求「所有先跑 1 种子」）
- 设备：2 × NVIDIA H20（97GB），变体串行、单卡独占
- 路由：数据级路由（`routing='data'`）
- 预训练 batch=10000、lr=1e-3、AdamW；子任务调制 batch=16384、1 轮按场景调制训练
- 论文三种表征用法：特征级 / 模块级 / 模型级（`FeatureUsage` / `ModuleUsage` / `ModelUsage`）
