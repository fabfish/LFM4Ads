# 共享残差 Specialist-Only Continual Adaptation Screen — INCONCLUSIVE 结论

- 记录时间：2026-08-12 19:56
- 实验入口：`scripts/shared_residual_experiment.sh`（双卡矩阵 runner）
- 驱动文档：`docs/archive/drivers/20260812-1807-共享残差混合专家-函数保持与持续学习-驱动.md`
- Gate 判定文件：`cache/shared_residual_continual/specialist_screen_gate_decision.json`
- 结果表：`result_shared_residual_specialist_screen.csv`

## 1. 目标

在 G1/G2（函数保持 upcycling、不变量）已 PASS 的前提下，跑 G3 的第一道 screen：
**Specialist-Only Continual Adaptation Screen** —— 检验「共享残差 MoE 的 specialist-only
持续适配」是否相对「仅训 task-head」基线有可证明的优势。若通过，则解锁
Shared-Path Necessity Gate（进而允许探索 shared-LR / router-LR / sparse scale-out）；
若不通过或 INCONCLUSIVE，则停在授权边界内。

## 2. 环境与测量口径（统一，控制变量）

| 项 | 值 |
|----|----|
| 模型 | DCNv2MoE_SharedResidual，dim=360，K=4 expert，r=45 rank，num_tasks=15 |
| 路由 | frozen-uniform-fully-routed（不可训练） |
| 设备↔seed 绑定 | seed 42 → cuda:0，seed 123 → cuda:1，seed 456 → cuda:0（同 seed 全路由模式同卡） |
| 固定步数预算 | 每 scenario 24 步优化，epochs_per_task=2，early_stopping=false |
| batch_size | 10000，num_workers=4 |
| 训练顺序 | order A = scenario 0 → 3；order B = scenario 3 → 0 |
| learning rate 扫描 | specialist-only: {2e-4, 5e-4, 1e-3}；task-head-only 基线固定 5e-4 |
| seed | {42, 123, 456}（3 seed，跨 scenario 仅看相对排序） |
| 矩阵规模 | 2 orders × 3 seeds × (1 task-head + 3 specialist-LR) = **24 trial** |
| 失败隔离 | 单 trial 失败记 failed 后续跑；矩阵级异常终止全部子进程 |

## 3. 结果

### 3.1 运行结果

- 矩阵 `status = done`，**24/24 完成，0 失败**。
- 单 trial 墙钟 ~61–70s（其中 GPU 净训练仅 ~8s，其余为数据重建 / torch 导入 / 10s poll 延迟；
  利用率低为「小模型 + 固定步数 + 每 trial 独立进程失败隔离」的固有特征，非异常）。

### 3.2 Gate 判定

- `verdict`：**INCONCLUSIVE**
- `reason`：paired BWT direction / practical-significance rules are not jointly met
- `winning_specialist_learning_rate`：**null**（4 个 LR 无一满足联合规则）
- `unlock_shared_path_necessity_gate`：**false**

### 3.3 指标表（learning_accuracy / backward_transfer）

| arm | LR | order | seed | learning_acc | backward_transfer |
|-----|----|--------|----|----|----|
| task-head-only (基线) | 5e-4 | 0→3 | 42 | 0.738424 | 0.0 |
| task-head-only (基线) | 5e-4 | 0→3 | 123 | 0.720408 | 0.0 |
| task-head-only (基线) | 5e-4 | 0→3 | 456 | 0.715204 | 0.0 |
| specialist-only | 2e-4 | 0→3 | 42 | 0.733516 | -1.2e-5 |
| specialist-only | 2e-4 | 0→3 | 123 | 0.722129 | +1.9e-5 |
| specialist-only | 2e-4 | 0→3 | 456 | 0.719139 | -8.2e-6 |
| specialist-only | 5e-4 | 0→3 | 42 | 0.733036 | -5.3e-5 |
| specialist-only | 5e-4 | 0→3 | 123 | 0.722436 | +5.3e-5 |
| specialist-only | 5e-4 | 0→3 | 456 | 0.719757 | -3.4e-5 |
| specialist-only | 1e-3 | 0→3 | 42 | 0.732186 | -1.5e-4 |
| specialist-only | 1e-3 | 0→3 | 123 | 0.723762 | +9.1e-5 |
| specialist-only | 1e-3 | 0→3 | 456 | 0.720382 | -1.1e-4 |
| task-head-only (基线) | 5e-4 | 3→0 | 42 | 0.738707 | 0.0 |
| task-head-only (基线) | 5e-4 | 3→0 | 123 | 0.720581 | 0.0 |
| task-head-only (基线) | 5e-4 | 3→0 | 456 | 0.715234 | 0.0 |
| specialist-only | 2e-4 | 3→0 | 42 | 0.738789 | -1.18e-2 |
| specialist-only | 2e-4 | 3→0 | 123 | 0.721246 | +3.06e-3 |
| specialist-only | 2e-4 | 3→0 | 456 | 0.715965 | +5.74e-3 |
| specialist-only | 5e-4 | 3→0 | 42 | 0.739326 | -1.38e-2 |
| specialist-only | 5e-4 | 3→0 | 123 | 0.721601 | +2.07e-3 |
| specialist-only | 5e-4 | 3→0 | 456 | 0.716372 | +6.47e-3 |
| specialist-only | 1e-3 | 3→0 | 42 | 0.739912 | -1.64e-2 |
| specialist-only | 1e-3 | 3→0 | 123 | 0.721995 | +1.19e-3 |
| specialist-only | 1e-3 | 3→0 | 456 | 0.716759 | +7.47e-3 |

## 4. 解读

1. **specialist-only 相对 task-head-only 基线的 learning_accuracy 差异在 1e-3 量级**，
   落在 3-seed 噪声地板内 —— 既无一致正向、也无一致负向。
2. **paired backward_transfer 方向不一致**（同一 LR 下有的 seed 正、有的负），
   因此「方向一致 + 实际显著（practical-significance）」的联合规则无法满足 → INCONCLUSIVE。
3. 含义：在当前固定步数预算（每 scenario 24 步）+ 3-seed 探索口径下，
   **没有证据支持需要走 shared-residual 的 specialist 适配路径**。
4. `unlock_shared_path_necessity_gate = false` → 按授权边界应**停止在该 screen 内**，
   不得扩展到 shared-LR / router-LR / sparse scale-out。

## 5. 代码改动清单

本轮无源码改动，仅执行了既有的双卡矩阵（提交 `a678866` 已包含其代码、脚本、文档与不可变证据）。
实验运行期间 GPU 利用率偏低（~15–20%）为设计固有特征，非缺陷；优化数据准备缓存需改动
`dataset.py`/trial 脚本从而改变 `source_sha256`，会破坏 G3 gate 要求的「24 trial 同源」不变量，
故未在中途实施（且剩余墙钟仅数分钟，收益可忽略）。

## 6. 踩坑与修复

- `bash scripts/shared_residual_experiment.sh summarize` 末尾报
  `FileExistsError: refusing to overwrite aggregate results` —— 这是**保护性拒绝覆盖**，
  非实验错误：聚合文件（`result_shared_residual_specialist_screen.csv`、
  `specialist_screen_gate_decision.json`、`specialist_screen_result_manifest.json`）
  已在首次汇总（19:52）写出。判定结果本身正确且已落盘。若需强制重算，需先手动删除这三个文件。

## 7. 后续 backlog

- **选项 A（停在授权边界）**：接受 INCONCLUSIVE，不扩展路径；本 screen 结论即终点。
- **选项 B（提口径）**：增大 seed 数（如 5–10 seed）或放大每 scenario 步数预算，
  看差异是否脱离噪声地板；若仍 INCONCLUSIVE，则更坚定地停在边界内。
- **选项 C（未来更大实验预埋）**：若要做更大规模实验，从一开始把 `Split` 数据准备缓存做进
  `dataset.py`，并配套重跑 G1/G2 verify，保持 24 trial 同源 provenance。
- 无论选哪个，均**不得越界**到 shared-LR / router-LR / sparse scale-out（gate 未解锁）。
