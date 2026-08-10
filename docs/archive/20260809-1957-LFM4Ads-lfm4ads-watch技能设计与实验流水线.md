# LFM4Ads lfm4ads-watch 技能设计与实验流水线

> 日期：2026-08-09 19:57 | 驱动编号：D8 | 状态：设计定稿 / 技能待创建
>
> 本文统辖 `.codebuddy/skills/lfm4ads-watch/` 的创建与验证，声明技能需求、架构、与 D0–D7 的依赖关系，并记录 AdaTask vs 原文的 7 项口径差异作为后续公平实验的准绳。
>
> **关联代码**：`main.py` / `main_adatask.py` / `main_moe.py` / `main_moe_v2.py` / `main_continual.py` / `dataset.py` / `model.py` / `train.py`
>
> **数据源**：`cache/dcnv2_vanilla.pt`、`cache/vanilla_pretrain.pt`、`cache/adatask_results.csv`、`cache/moe_pretrain.log`、`docs/DRIVERS.md`、`docs/20260809-1905-*.md (D7)`

---

## 目标

1. **健全化 long-run-watch 为 LFM4Ads 专属技能**：创建 `.codebuddy/skills/lfm4ads-watch/`，使 AI Agent 能自循环执行数小时实验（预训练 / 下游评估 / 消融矩阵），无需人工介入
2. **记录 AdaTask 口径差异**：将 D7 Q2 中 5 项差异扩展为 7 项，逐项标注是否影响可比性，给出统一公平口径
3. **承接 D7 路线图**：为 P0（多 seed 复跑）和 P1（基座统一重做）提供自动化执行基础设施

---

## 格局

### AdaTask vs 原文：7 项差异详录

> 来源：`main_adatask.py:85-86` vs `main.py:22` / `train.py`。用户指出与原文设计不完全一致，要求之后尽量按公平口径。

| # | 差异项 | 原文 pre-train | AdaTask D4 | 是否影响可比性 | 风险说明 |
|---|--------|:-------------:|:----------:|:-------------:|----------|
| 1 | **训练集** | `train_set` only (`date < 20220503`) | `pd.concat([train_set, valid_set])` (`< 20220506`) | ❌ **严重影响** | 多吃 3 天数据（~1.7M 样本），趋势更有利，跨场景对比不可比 |
| 2 | **验证集** | `valid_set` → early-stop 判断 | **无**（valid 被吃进训练） | ❌ **严重影响** | 无 early-stop 无法截断过拟合；1 epoch 缓解但不可忽略 |
| 3 | **Epoch 数** | `while True` + early-stop（通常 ≥2） | 固定 `1` | ⚠️ 中等 | 1 epoch 欠训练风险；原文 2 epoch 后 often converged |
| 4 | **Batch size** | `10000`（AdamW 默认） | `16384`（+64%） | ⚠️ 中等 | 大 batch → 更稳梯度，可能抑制 per-scenario 噪声 → AU 调制精度更好 |
| 5 | **基座 checkpoint** | `cache/dcnv2_vanilla.pt` | `cache/vanilla_pretrain.pt` | ❌ **严重影响** | 两份权重不等——AdaTask Phase 0 独立训练 vanilla（`main_adatask.py:36-48`），其 `dataset.py:train()` 用的 `pd.concat([train_set, valid_set])` 数据+1 epoch，等效于不同的种子/数据路径 |
| 6 | **LR** | AdamW 默认 `1e-3` | 显式 `1e-3` | ✅ 无影响 | 数值一致，差异仅显式 vs 默认 |
| 7 | **Optimizer 封装** | 原始 `AdamW`（`train.py:6`） | `AdaTaskOptimizer` 包装 `AdamW` + Hook | ✅ 无影响（none 模式） | none 模式下 `AdaTaskOptimizer` 行为应等价 AdamW；需 smoke 验证 |

> **直接后果**：D4 的 `none = 0.7087` 与 D3 的 `MoE = 0.7218` **不能直接对比**（D7 已标注，D8 首条确认）。D4 内部 3 模式可互比（同数据+同权重），但跨 D3/D4 的任何数字比较均无效。

### 统一公平口径

> 来源：`docs/20260809-1905-...md (D7)` §2.2 统一口径方案。D8 在此基础上追加星号标注的细项。

| 参数 | 统一值 | 依据 |
|------|--------|------|
| 基座 checkpoint | `cache/dcnv2_vanilla.pt` | 所有变体必须共享同一份权重；放弃 `vanilla_pretrain.pt` |
| 训练集 | `train_set` only (`date < 20220503`) | 与原文及 D3 一致 |
| 验证集 | `valid_set` → early-stop 判断 | 恢复原文流程 |
| lr | `1e-3` | 一致 |
| batch_size | `16384` | 沿用较大 batch（稳定）+ 原文需重新跑一次统一 batch |
| epoch | 2 | 充分训练，后续可消融 1 vs 2 |
| weight_decay | `0.01` | 统一（AdaTask 已有此值） |
| seed | 42 起步 → 扩展至 3 seed | 单 seed 达标再扩展 |
| 下游评估 | 每次 evaluation 后显式写 `result.csv` | 便于 watch_tick.sh 自动计数 |

**路径区分**：统一重做实验产物存 `cache/retest_vanilla.pt` / `cache/retest_moe_*.pt` / `cache/retest_adatask_*.csv` 等，与 D0–D4 的旧产物**物理隔离**，避免数据源混淆（DRIVERS.md §3 已为此预留映射行）。

---

## 设计

### 3.1 技能结构

```
.codebuddy/skills/lfm4ads-watch/
├── SKILL.md                          # 核心技能文件（YAML frontmatter + 工作流）
├── scripts/
│   ├── watch_tick.sh                 # [复用] 直接拷贝自 long-run-watch，纯通用 bash
│   ├── checkpoint_commit.sh          # [复用] 直接拷贝自 long-run-watch，纯通用 bash
│   ├── launch_experiment.sh          # [新增] LFM4Ads 实验 detached 启动器
│   └── smoke_test.sh                 # [新增] 前向/反向快速验证
├── references/
│   └── repo_probes.md                # [重写] OLMoE → LFM4Ads 专属探针
└── assets/                           # 预留（暂空）
```

#### 各模块设计意图

| 文件 | 角色 | 设计决策 |
|------|------|----------|
| `SKILL.md` | 技能本体，AI Agent 加载时延展为行为准则 | 嵌入 AGENTS.md 约束（串行/控制变量/AUC/文档规范），触发条件为"跑实验/盯着训练/长程执行/挂起等"，工作流 6 步（Preflight → Launch → Tick Loop → Report → Checkpoint → Terminate） |
| `watch_tick.sh` | 前台阻塞 + 进程探测 + GPU 状态 + 日志增量 + stall 检测 | 直接复用 long-run-watch 版本，通过 `WATCH_*` 环境变量配置，无需改一行；tick ≤900s 上限 |
| `checkpoint_commit.sh` | 按 cadence 自动 git commit（不 push） | 复用；仅提交 `WATCH_COMMIT_PATHS` 白名单内文件，不碰用户其他 staged 修改 |
| `launch_experiment.sh` | 通用 detached 启动器 | 接受 `--name` / `--script` / `--device` / `--seed` / `--extra-args`，preflight 检查（vanilla ckpt 存在 / nvidia-smi 无冲突 / Smoke 通过），然后 `setsid nohup python <main_script> <args> >/tmp/lfm4ads_<name>.log 2>&1 < /dev/null &` |
| `smoke_test.sh` | 前向/反向验证，AGENTS.md §5 要求 | 加载 DCNv2 → 构造 mini batch(320 样本) → forward → loss → backward，确认不 OOM 不 NaN |
| `repo_probes.md` | LFM4Ads 专属探针，替代 OLMoE 内容 | 5 节：入口映射 / WATCH_* 就绪配置 / 日志规则 / 失败签名 / checkpoint 范围 |

### 3.2 自循环工作流

```
Preflight (声明终止条件 + pgrep/nvidia-smi 冲突检测)
   │
   ▼
Launch (launch_experiment.sh → setsid nohup)
   │   30 s tick 验证 liveness
   ▼
Tick Loop ──────────────────────────────────────────────────┐
   │  watch_tick.sh <interval>                                │
   │  输出：进程状态 / GPU / artifact 计数 / 日志增量 /       │
   │        delta vs 上一 tick / stall 警告 (≥3 ticks no change)
   │                                                          │
   ▼                                                          │
Report (≤5 行分析：增量/速率/异常/下一步)                      │
   │                                                          │
   ├─ 每 ~6 ticks (≈1 h) 或状态迁移 ──► Checkpoint Commit     │
   │                                      checkpoint_commit.sh │
   │                                                          │
   └─ 终止条件未满足 ────────────────────────────────────────┘
   │
   ▼
Terminate (验证产物完整性 → 最终 commit → 总结报告)
```

失败处理（loop 内自愈，不退出）：
- OOM → 移到空闲设备或等 holder 退出
- Crash → 读 log tail → 修复 → relaunch → 重置 tick 30s
- Stall ≥3 ticks → 停止 sleep 主动诊断（`py-spy dump` / `nvidia-smi` / file mtime）
- pgrep 匹配不到但 log 在写入 → pattern 错误而非 job 死
- 三次不同修复仍失败 → 向用户升级，带诊断

### 3.3 入口映射表（repo_probes.md §1 核心内容）

| 入口脚本 | 命令 | 角色 | 预估 wall time (1 seed) | 关键产物 |
|----------|------|------|:----------------------:|----------|
| `main.py` | `python main.py cuda:0` | Vanilla 预训练 + 三级下游 | ~30 min | `cache/dcnv2_vanilla.pt` + `result.csv` |
| `main_moe.py` | `python main_moe.py cuda:0` | MoE V1（soft routing, K=4） | ~2.5 h | `cache/dcnv2_moe_k4.pt` + `result_moe.csv` + `result_moe_downstream.csv` |
| `main_moe_v2.py` | `python main_moe_v2.py cuda:0` | MoE V2（shared + top-k） | ~3 h | `cache/dcnv2_moe_v2_k4.pt` + `result_moe_v2.csv` |
| `main_adatask.py` | `python main_adatask.py cuda:0` | AdaTask 3 modes | ~3.5 h | `cache/adatask_results.csv` + `cache/adatask_au_*.json` |
| `main_continual.py` | `python main_continual.py cuda:0` | 持续学习 | ~20 min | `cache/continual_results.json` |

### 3.4 失败签名（repo_probes.md §4 核心内容）

| 签名 | 诊断 | 动作 |
|------|------|------|
| `torch.OutOfMemoryError` | GPU 内存不足 | 检查 nvidia-smi 持有者，等或搬 |
| `loss = NaN` | 梯度爆炸 | kill + 检查 LR/数据 + relaunch |
| 进程 alive，GPU util 0%，log 停滞 ≥3 tick | 真正 hang | `py-spy dump` → kill → 修复 → relaunch |
| pgrep 无匹配但 log 数秒前有写入 | pattern 错误，job 没死 | 修正 `WATCH_PROC_PAT` |
| `SIGKILL`（log 无 Traceback，突然消失） | 外部 kill（OOM killer / 用户释放 GPU） | retrain |
| `result.csv` 行数不变但 log 在推进 | 下游评估阶段可能在 accumulate（正常），或评估 hang | 对比上一次 checkpoint 确认 |

---

## 结果（设计规格，非实验结果）

### 4.1 SKILL.md 核心约束（嵌入 AGENTS.md）

```
- 设备：cuda:0，单卡独占，变体串行
- 切分：train date<20220503 / valid [20220503,20220506) / test ≥20220506
- 指标：AUC (torcheval BinaryAUROC)，写入 result.csv
- 探索阶段：3–5 seed 快速验证
- 公平对比：除被测项外 seed/batch_size/lr 完全一致
- 文档记录：YYYYMMDD-HHMM-主题.md，含目标/环境/结果/改动清单/踩坑/backlog
```

### 4.2 launch_experiment.sh 设计规格

```
用法: launch_experiment.sh --name <exp_name> --script <main_*.py> --device <cuda:N> [--seed 42] [--extra-args "..."]
Preflight 检查:
  1. vanilla checkpoint 是否存在（否 → 报错并要求先跑 vanilla）
  2. nvidia-smi 显示目标 GPU 上是否有其他 python 进程（是 → 报冲突）
  3. smoke_test.sh 通过（否 → 拒绝启动）
输出: PID + log path
```

### 4.3 smoke_test.sh 设计规格

```
1. import model.py 中所有类（DCNv2 / DCNv2MoE / DCNv2MoE_V2 / FeatureUsage / ModuleUsage / ModelUsage）
2. import fields.py 确认字段数 = 36
3. 构造 mini batch（320 样本 × 36 fields）→ forward → BinaryCrossEntropy loss → backward
4. 确认: (a) no OOM (b) no NaN in loss (c) grad norm > 0 (d) result.csv 可写
5. 任一失败 exit 1，全通过 exit 0
```

---

## 分析

### 5.1 为什么复用 watch_tick.sh / checkpoint_commit.sh 而不是重写

`watch_tick.sh` 的核心逻辑是纯粹的进程/日志/文件系统探针——它通过 `WATCH_*` 环境变量接收所有配置，本身不包含任何项目特定逻辑。重写只会引入 bug。`checkpoint_commit.sh` 同理。

LFM4Ads 的适配全部承载在 `repo_probes.md`（WATCH_* 配置）和 `launch_experiment.sh`（启动逻辑）中。

### 5.2 为什么 repo_probes.md 需要整文件重写

原文件是 OLMoE 专属（入口在 `training/main.py`、产物在 `outputs_prototype/...`、日志格式为 tqdm progress bar）。LFM4Ads 入口完全不同（`main.py` 等 5 个入口、产物在 `cache/*`、日志无 tqdm），逐节替换比就地修改更干净。保留 5 节结构但全部内容重写。

### 5.3 为什么 AdaTask 差异不写入技能本身

技能是对所有实验入口的通用操作层，AdaTask 差异是特定实验的上下文约束。差异记录在 D8 驱动文档（本文），技能内引用 D8 即可。

---

## 缺陷

| ID | 摘要 | 状态 |
|:---|------|:---:|
| D8-A | 技能创建完成后需在 D7 P0 实验中实际验证自循环稳定性 | ⏳ |
| D8-B | `launch_experiment.sh` 目前无 lock 机制：两个 Agent 同时启动可能打架 | 📝 设计已知/当前单 Agent 场景暂不实现 |
| D8-C | `main_adatask.py` 内嵌 Phase 0 vanilla 训练，无法直接指定外部 `dcnv2_vanilla.pt` → 统一重做时需要改造入口或另写 wrapper | 📝 见 Backlog P1-1 |

---

## Backlog

### 技能创建（本驱动文档执行后）

- [ ] **D8-1**：使用 skill-creator 初始化 `.codebuddy/skills/lfm4ads-watch/`
- [ ] **D8-2**：拷贝 `watch_tick.sh` + `checkpoint_commit.sh` 自 `long-run-watch/scripts/`
- [ ] **D8-3**：重写 `repo_probes.md`（入口映射/WATCH_*/日志/失败签名）
- [ ] **D8-4**：创建 `launch_experiment.sh` + `smoke_test.sh`
- [ ] **D8-5**：编写 `SKILL.md`
- [ ] **D8-6**：smoke 验证 + dry-run 检查

### 技能就绪后 — 承接 D7 路线图

- [ ] **P0-1**：Vanilla baseline 3 seed 复跑（用 lfm4ads-watch 自循环）
- [ ] **P0-2**：MoE V1 3 modes 3 seed
- [ ] **P1-1**：改造 `main_adatask.py` 支持统一口径（外部指定 vanilla ckpt + train only + 2 epoch + early-stop）
- [ ] **P1-2**：E1–E6 全部实验（统一口径，3 seed）

### 统一口径实验矩阵（D7 E1–E6）

> 来源：D7 §2.2。D8 承接执行。

| 实验 | 入口模块 | 预估时间 (1 seed) | 状态 |
|------|---------|:-----------------:|:----:|
| E1 Vanilla 统一口径 | 新 wrapper（复用 `train.py:train`） | ~20 min | ⏳ |
| E2 MoE baseline 统一口径 | 新 wrapper（`DCNv2MoE.load_pretrained(dcnv2_vanilla.pt)`） | ~2.5 h | ⏳ |
| E3 AdaTask none 统一口径 | `main_adatask.py` 改造后 | ~2.5 h | ⏳ |
| E4 AdaTask encourage | 同上 | ~2.5 h | ⏳ |
| E5 AdaTask suppress | 同上 | ~2.5 h | ⏳ |
| E6 encourage+suppress 混合 | 同上（可选，D7 建议先 α sweep） | ~2.5 h | ⏳ |

---

> **下游依赖**：本文创建后，需回到 DRIVERS.md 登记——D8 已创建，注册表 + 依赖图 + 数据源映射 + G-3 状态更新 4 项变更见 DRIVERS.md。
