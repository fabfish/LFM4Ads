# 真实任务占卡守护（gpu_work_keeper）

日期：2026-08-19
主题：政策更新后，用「真实实验任务」替代「假负载」占卡

## 0 背景与政策

2026-08-19 政策更新：**之后不能用简单占卡脚本（假负载/矩阵乘心跳）跑在 GPU 上；GPU 占用只能来自真实、有用的任务。**

据此：
- 原 `scripts/gpu_keeper.py`（`--fake` 假负载）与 `scripts/gpu_keeper_siteB.py` 已停用（2026-08-19，cron 看门狗已移除，见
  [占位守护停用记录](./archive/operations/20260812-1130-gpu-keeper-占位守护.md)）。
- 新增 `scripts/gpu_work_keeper.py`：**真实任务占卡守护**——空闲 GPU 上派的是真实实验任务（满利用率 + 产出 run json），
  而非假负载。

## 1 设计

| 特性 | 说明 |
|---|---|
| 真实任务来源 | 复用 `scripts/matrix/run_macro_auc_matrix.py` 的预注册 stage（`build_tasks`），只跑「已预注册、有授权」的真实实验，绝不为了占卡乱造配置 |
| 真实任务入口 | `experiments/main_macro_auc.py`（27K + macro 端点主线入口） |
| 空闲自检 | 派活前 `nvidia-smi` 确认该卡无计算进程，不抢占手动任务、不叠卡 |
| 断点续跑 | `run_<tag>.json` 已存在则跳过，可随时 kill/重启 |
| 失败隔离 | 单个 run 崩溃记录后继续下一个，不拖垮守护 |
| 终止条件 | 任务表耗尽即自然退出（「无真实任务则不占卡」） |

与 gpu_keeper 的对应关系：
- `gpu_keeper`：空闲 → 派 `--fake` 假负载（占卡，无产出）
- `gpu_work_keeper`：空闲 → 派真实训练任务（占卡 + 产出真实结果）

## 2 用法

```bash
# 环境变量（与矩阵 runner 相同，缺一会静默用错数据集或直接 S1 FAILED）
export LFM_DATASET=$PWD/dataset_27k.feather
export LFM_VOCAB_JSON=$PWD/cache/fields_27k.json
export LFM_SAMPLE_COUNTS_JSON=$PWD/cache/sample_counts_27k.json
export LFM_MACRO_OUT=$PWD/cache/macro_auc_27k          # 27K 证据目录，勿与 1K 混
export LFM_SITE=A

# dry-run：只打印任务分配，不派活
python scripts/gpu_work_keeper.py --devices cuda:0,cuda:1 --stages s1,s2sent --dry-run

# 占 cuda:0/cuda:1，跑预注册主线 stage
nohup python scripts/gpu_work_keeper.py \
    --devices cuda:0,cuda:1 --stages s1,s2sent \
    > logs/gpu_work_keeper.log 2>&1 &
```

`--stages` 可填矩阵 runner 里任意已预注册 stage（`s1` / `s2sent` / `s6sparse` / `s7pool` / `s8full` / `s9sel` / `b0repro` / `b1topk` 等）。

## 3 现状核对（2026-08-19）

- `cache/macro_auc_27k/` 与 `cache/macro_auc/` 的所有预注册主线 stage `run_*.json` 已齐全，`matrix_state.json` 里
  `failed=[]`、`not_started=[]`——**site A 所有预注册真实任务均已 done**。
- 因此 `gpu_work_keeper` 用默认 `--stages s1,s2sent` 启动会因断点续跑全部 skip，打印「无待跑真实任务…退出」——这是符合政策的预期行为（没有真实任务就不占卡）。
- GPU 当前空闲（0% / 0 MiB）是「任务跑完」的自然结果，无需人为占卡。

## 4 后续约定

- GPU 只跑真实实验任务；有新的预注册实验要跑时，用 `gpu_work_keeper.py --stages <stage>` 占卡 + 产出结果。
- 新实验仍需先预注册（冻结哨兵/判定），不得为了占卡临时造配置。
