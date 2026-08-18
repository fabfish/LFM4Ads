# HANDOFF — 交接一页纸（给后续执行者）

> 本文是给接手实验的模型/人读的**速览**。完整证据见 [`DRIVERS.md`](DRIVERS.md)，
> 已可靠结论见 [`ANALYSIS.md`](ANALYSIS.md)，后续授权见 [`NEXT.md`](NEXT.md)。

---

## 0. 一句话状态

**MoE 路线已拿到公平比较下的正面结果，当前形态 = 27K + macro 端点 + 硬路由 top_k=2。**

| 结论 | 数字 | 状态 |
|---|---|---|
| MoE 有效（轻量模型） | Δ=**+0.005012**，4/4 seed 正，地板 3.70× | **PASS**（E10） |
| MoE 有效（加回 5.5 亿 ID 参数） | Δ=**+0.004329**，4/4 seed 正，地板 3.24× | **PASS**（E11，公平性质疑排除） |
| 收益来自路由而非容量 | 参数守恒 +225；frozen-router 哨兵 −0.0003/+0.0001 ≈0 | 已验证 |
| ID 表**有害** | 加回后 macro 降 **0.0055**，4/4 seed 负 | **PASS**（E11） |

**当前最优配置**：轻量模型（去三大 ID 表，dim=330，0.87M 参数）+ scenario-routed MoE(K=5, **硬路由 top_k=2**)，
macro AUC = **0.735388**。

**突破归因（缺一不可，三者叠加）**：
1. **换 27K**（必要）：1K 小场景 test 正类仅 70 个、噪声地板 0.0113 > 待测效应 → 统计上不可能测出；
   27K 地板降到 0.00135（**低 8.4×**）。
2. **换 macro 端点**（必要）：pooled AUC 里一大半是"认出场景"白拿的，MoE 收益全在场景内排序；
   同样 27K 下只换回 pooled 就退回 INCONCLUSIVE（E8）。
3. **硬路由 top_k=2**（放大器）：效应从 +0.0019（地板 1.39×）放大到 +0.0050（3.70×），结论从脆弱变稳健。
4. 删 ID 表只是**工程使能条件**（模型 ÷634、快 6 倍，让 80+ run 矩阵跑得起），**不是收益来源**（E11 已证）。

详见[突破归因](./20260817-1208-突破归因-公平比较下的正面结果.md)。
**当前最高优先级（2026-08-18 18:15）**：E17 在预注册 s0 上先做 task-state baseline，再做真 task-axis、post-preconditioner AdaTask 增量；见[复审与预注册](./20260818-1815-结论复审与E17单场景AdaTask-win-case预注册.md)。原 E12/E13/E14 与 F2 后置。

---

## 0.5 历史包袱（读旧文档时注意）

- **cross 层 capacity-MoE 已关闭**：4× 容量收益≈0、稀疏代价≈−0.001。
  该结论限于 **pooled 端点 + 1K + 容量维度**，与 E7/E10 的 PASS **不冲突**——
  前者证伪"加容量"，后者证实"改分配"。
- **"瓶颈在 embedding"已证伪撤回**：84M ID 表在 1K 上是死重（E1 PASS），在 27K 上进一步证明**有害**（E11）。
- **禁止用参数占比推断瓶颈**（97.9% 参数不仅不是瓶颈，加回还有害）。

---

## 1. 已可靠知道什么（初级结论，按证据强度分级）

### A 级（多 seed + 配对 + 决定性对照，高置信）

0. **【当前主线】scenario-routed MoE 在 macro 端点上有可复现收益**：
   - 硬路由 top_k=2，轻量模型 Δ=**+0.005012**（4/4 seed 正，地板 3.70×）；
   - full-ID（551M 参数）复核 Δ=**+0.004329**（4/4 seed 正，地板 3.24×）→ **公平性质疑排除**；
   - hard−soft 配对差 +0.003131（4/4 正）→ 硬路由的放大作用是真的；
   - 参数守恒（moe−dense=**+225** router）+ frozen-router 哨兵 ≈0 → **收益来自路由，非容量、非实现差异**。
1. **cross 层容量不是 AUC 瓶颈**（dense-widened 4× 加宽+非线性、零路由零稀疏，Δ_capacity 跨 seed 变号、
   |Δ|≤0.0007 < 噪声地板）。这是 capacity-MoE 关闭的直接证据。
   → 与第 0 条**不冲突**：前者证伪"加容量"，后者证实"改分配"。
2. **capacity-MoE 收益 = 容量收益(≈0) + 稀疏代价(≈−0.001)**，净效应必为负，不是实现问题。
   → 该分解限于 **pooled 端点 + 1K**；在 27K + macro 端点下稀疏（硬路由）反而是**正收益**（第 0 条）。
3. **84M ID embedding 是死重，不是瓶颈**（2026-08-14 修订，原第 3 条"embedding 是瓶颈"已证伪）：
   - 三张 ID 表占 embedding 的 **99.96%**（83,984,250 / 84,017,390），其余 33 个字段合计仅 **33,140** 参数；
   - 推理期把这 84M 全部清零，test AUC `0.777490 → 0.777297`，**Δ = −0.000193**（噪声内）；
   - **E1 重训练确认 PASS**：from-scratch 剔除后 Δ_id = {+0.000688, +0.000220}（2/2 seed 地板内），
     参数 **−99.31%**、wall **1.48×**；哨兵 10/10 PASS；
   - **27K 上更强一档（E11）**：加回 5.5 亿 ID 参数后 macro **降 0.0055**（4/4 seed 负，4 倍地板）
     → 不只是"死重"，而是**有害**；
   - 依赖度最高的字段是 `tag`(−0.0066) / `upload_type`(−0.0053，仅 320 参数) / `user_id`(−0.0049)；
   - `video_id` 平均每 ID 仅 **2.6** 次曝光、79.8% 的 ID 曝光≤2 次、**test 74.19% 样本的 video_id
     在 train 中从未出现**；**17.53% 的 embedding 参数永远收不到梯度**；
   - 原证据"冻结 embedding 后 AUC 反而更高"的正确读法是 **burden（负担）**，不是 bottleneck（瓶颈）。
4. **三个实现 bug 已修复并固化**（见 §3）。
5. **噪声地板必须实测、且随端点与数据集变化**：pooled@1K 0.001 / macro@1K **0.0113** /
   macro@27K **0.00135** / pooled@27K 0.00102。**开跑前先算地板**；地板 > 预期效应则实验设计无效。

### B 级（机制成立，但≠收益）

5. **router 特化机制是真的**：熵从 log K=1.386 离开到 0.74，真实 sparse dispatch 校验通过（12/12 哨兵 PASS）。
6. **真实稀疏引入"AU 冻结"**：未激活专家无梯度 → AU 不更新 → 专家坍缩（L0 的 E2 吃 0.34–0.42 dispatch）。
   这是旧全连接 MoE 没有的现象，但当前是"代价"而非"收益"。

### C 级（不成立/降格）

7. MoE 相对 dense 的静态 AUC 优势（+0.0025 是微调红利，非结构红利）。
8. same-FLOPs / same-latency 效率优势（实测 wall-clock 更慢）。
9. TCMR 任务条件路由、specialist-only 持续适配、下游留出域迁移优势（均 INCONCLUSIVE 或 FAIL）。

---

## 2. 已关闭的路线（不要再碰）

| 路线 | 状态 | 一句话 |
|---|---|---|
| capacity-MoE（cross 层，**加容量**） | **正式关闭** | 容量收益≈0 + 稀疏代价≈−0.001（限 pooled@1K） |
| **embedding 容量（加宽）** | **正式关闭** | 现有 84M 容量本就没被用上；27K 上加回 ID 表还**有害**（−0.0055，4/4 负） |
| Stage B same-FLOPs MoE | FAIL | 同 seed 差 `[-,-,+]`、wall-clock 更慢 |
| TCMR 静态任务条件路由 | INCONCLUSIVE | 跨 seed 变号、未越噪声地板 |
| 共享残差持续学习 | blocked | G3 INCONCLUSIVE、gate=false |
| 下游留出域迁移 | G1 FAIL | moe-router 1/3 target 胜、停止 |
| E9 K 粒度扫描 | 撤销 | 已授权缩减（top_k 维度比 K 粒度重要） |

> **注意别误读**：关闭的是"**加容量**"和上面这些具体路线，**不是 MoE 本身**。
> MoE 在 27K + macro 端点 + 硬路由下是 PASS（§1 第 0 条）。

**核心教训（写进后续所有设计）**：

1. 加容量/加专家前，必须先用"widened 对照"证明**存在容量缺口**。否则就是 cross 层 MoE 的重演。
2. 判断"瓶颈在哪"**不能用参数占比**。先做零训练成本诊断（参数集中度/曝光频次/OOV 率/推理期消融）。
   本轮靠它在 1 分钟内挡掉了一个已写进路线图的 embedding-widened 长跑。
3. **"没有效应"≠"测不出效应"**（2026-08-17 新增，最贵的一课）。1K 上所有 MoE 实验全 INCONCLUSIVE，
   曾被解读为"MoE 没用"；实际是小场景 test 正类仅 70 个、噪声地板 0.0113 > 待测效应。
   → **开跑前先算噪声地板**（dense 臂跨 seed SD 或 Hanley-McNeil SE）；地板 > 预期效应则设计无效，别花钱。
4. **主终点必须对准收益的作用位置**（2026-08-17 新增）。pooled AUC 看似正统，但对 MoE 收益近乎不敏感
   （macro 的 Δ 是 pooled 的 2–7 倍）。先问"收益会出现在哪个子群"，再选端点。
5. **性能优化先 benchmark 再动手**（2026-08-17 新增）。为 full-ID 写的 sparse embedding + SparseAdamW
   实测比 dense AdamW **慢 2.5 倍**（30.9ms vs 12.3ms），且有"未访问行动量不衰减"的语义差异无法严格
   对齐，方案作废。

---

## 3. 已固化的代码资产（直接复用）

| 资产 | 文件 | 说明 |
|---|---|---|
| `GpuBatches` | `dataset.py` | 全表 1.83GB 常驻 GPU，逐 batch `index_select`；评估 14×、训练吞吐 11× |
| `infer_gpu` / `evaluate_gpu` | `train.py` | GPU 常驻数据的等价推理/评估（已验证 |ΔAUC|=0） |
| `--freeze sparse` | `main_moe_capacity.py` | 冻结 84M embedding，两臂同等施加 |
| `--full-batch-loss` | `main_moe_capacity.py` | 单次全批前反向（sample 加权等价，R_gain≈0.999） |
| `--gpu-resident-data` | `main_moe_capacity.py` | 开关 GPU 常驻数据 |
| `--reinit-cross` | `main_moe_capacity.py` | 两臂 cross 层同时随机重置（制造真实 headroom） |
| `DenseWidenedDCNv2` | `model.py` | cross 层加宽 720 维+ReLU，参数量精确 4.00×，无路由 |
| `main_dense_widened.py` | — | dense-widened 三臂对照入口 |
| **零成本诊断（新）** | `scripts/diagnose/diagnose_embedding_capacity.py` | 参数集中度 / 曝光频次 / OOV 率，无需 GPU |
| **零成本诊断（新）** | `scripts/diagnose/diagnose_field_ablation.py` | 推理期逐字段 embedding 置零，测依赖度，~30s |
| **E1 入口（新）** | `experiments/main_field_ablation.py` | `full`/`idzero`/`iddrop` 三臂 from-scratch + 内置哨兵 |
| **E1 判定（新）** | `scripts/summarize/summarize_field_ablation.py` | 阈值硬编码防漂移，产出 `e1_decision.json` |
| **【当前主线入口】** | `experiments/main_macro_auc.py` | macro 端点 + `--arch dense/moe` + `--K` + `--top-k` + `--loss` + `--full-embeddings` + `--freeze-router` |
| **【当前主线矩阵】** | `scripts/matrix/run_macro_auc_matrix.py` | `--stages s1,s2sent,s6sparse,s7pool,s8full`，断点续跑、失败隔离、预算守卫、自动汇总 |
| **【当前主线判定】** | `scripts/summarize/summarize_macro_auc.py`、`summarize_macro_auc_stage2.py` | 阈值硬编码，实测噪声地板 |
| **27K 数据构建** | `scripts/build_27k_dataset.py` | 3.22 亿行、12.7min；输出 `dataset_27k.feather` + `cache/fields_27k.json` |
| **轻量基线（推荐默认）** | `iddrop` / `--full-embeddings` 关闭 | 27K 下 **0.87M 参数、3.4min/epoch**；full-ID 是 551M、20min/epoch 且 AUC 更差 |

**当前主线实验口径**（27K + macro，后续所有公平对比照此执行）：
```
LFM_DATASET=dataset_27k.feather LFM_VOCAB_JSON=cache/fields_27k.json \
LFM_SAMPLE_COUNTS_JSON=cache/sample_counts_27k.json LFM_MACRO_OUT=cache/macro_auc_27k \
python experiments/main_macro_auc.py <device> --arch moe --loss balanced --K 5 --top-k 2 \
  --lr 1e-3 --batch-size 10000 --max-epochs 20 --patience 10 --seed <42|123|456|789>
```
配对规则：同 seed 固定同卡（42/456/101→cuda:0，123/789/202→cuda:1）。

**历史口径**（capacity-MoE 时代，仅读旧文档时参考）：
`--freeze sparse --full-batch-loss --gpu-resident-data --lr 2e-4 --lr-router 1e-3 --batch-size 10000`

**关键模型文件**：`model.py`（`ScenarioRouter`/`DataRouter`/`CrossExpertLayer(top_k)`/`DCNv2MoE`/
`SubsetSparse`/`DenseWidenedDCNv2`/`DCNv2CapacityMoE`）。
**数据**：`dataset_27k.feather`（主线，3.22 亿行）、`dataset.feather`（1K，历史）；
切分见 `dataset.py:Split`（date<20220503 train / [20220503,20220506) valid / ≥20220506 test，两数据集相同）。

---

## 4. 下一步方向（有证据支持，按顺序）

**当前主线**：27K + macro 端点 + 硬路由 MoE 已 PASS（E10/E11）。三个已识别缺口 → 三个开放实验，
预注册见[开放实验计划](./20260817-1215-开放实验计划-下一个关键MoE改动.md)。

| # | 实验 | 缺口 | 成本 | 状态 |
|---|---|---|---|---|
| **E12** | **专家利用率诊断** | 不知道专家是否真分化（可能坍缩到 2/5） | **50 min** | `planned`（**必做，决定 E13**） |
| **E14** | **top_k 单调性**（top_k=1） | 5→2 收益 ×2.7，→1 更好还是回落？ | 4h | `planned`（独立，可先跑） |
| **E13** | **router 粒度升级**（scenario→data/hybrid） | `ScenarioRouter` 只有 **15 种 gate 模式**，样本级路由未测 | 12h | `planned`（**关键 MoE 改动**） |
| E12b | load-balance loss | 条件解锁：E12 判定坍缩时 | 8h | `blocked` |

**决策树**：E12 → 坍缩则先 E12b；已分化则直接 E13。E14 与两者无依赖，可插空跑。

**已完成的历史步骤**（不必重跑）：

1. **E1（`done / PASS`）— ID embedding 死重的重训练确认**：三臂 from-scratch，
   `full`(36 字段) / `idzero`(三张 ID 表置零+冻结，架构与 full 完全同构) / `iddrop`(真移除，dim=330)。
   Δ_id = {+0.000688, +0.000220}（2/2 seed 在 0.001 地板内）→ **PASS（死重成立）**；
   参数 84,672,605 → **584,255（−99.31%）**，wall/epoch 10.6s → 7.1s。
   入口 `experiments/main_field_ablation.py`，实际成本 8 min。
   → **后续实验默认基线改为 `iddrop`（0.58M 参数）**，3 seed × 多配置已变得廉价。
   结论见 [E1 结论](./20260814-2225-E1结论-ID-embedding死重确认.md)。
2. **E5/E6（`done / INCONCLUSIVE`）— 1K 上的场景内泛化 MoE + 隔离上界**：58+16 run 全 INCONCLUSIVE，
   根因是 1K 小场景 test 正类仅 70 个、地板 0.0113 → 换 27K。
3. **E7（`done / PASS`）— 27K 软路由 MoE**：Δ=+0.0019，4/4 seed 正 → 首个 work case。
4. **E8（`done / INCONCLUSIVE`）— pooled loss 组合**：5/6 正；绝对成绩 moe+pooled 最高。
5. **E10（`done / PASS`）— 硬路由 top_k=2**：Δ=+0.0050（地板 3.70×）→ **当前最优**。
6. **E11（`done / PASS`）— full-ID 公平性复核**：加回 551M 后 Δ=+0.0043（4/4 正）→ 公平性排除；
   副产：ID 表**有害**（−0.0055，4/4 负）。

**～～已弃用的旧路线（E2/E3/E4「特征信息侧」）～～**：当时基于"84M 是死重、瓶颈在信息侧"的判断提出，
但 E7/E10/E11 证明**路由/隔离维度仍有可挖收益**，优先级高于特征工程。E2/E3/E4 未执行，
如需重启须重新预注册。

所有新实验必须预注册（用 `lfm4ads-experiment-audit-planner` skill 冻结哨兵/判定），
同 seed 同卡配对差 + 4 seed；**先算噪声地板、先跑零成本诊断，再决定是否开 GPU**。

---

## 5. 必须遵守的规则（AGENTS.md 摘要）

- 文档放 `docs/`，命名 `YYYYMMDD-HHMM-主题.md`；**每轮对话结束前 `git commit` + `push`，确保工作区干净**（每完成一个逻辑单元 commit 一次，见 `AGENTS.md §5`）。
- 结果 CSV 放 `results/<实验>/`，证据 json 放 `cache/<实验>/`；新结果同步登记 `results/INDEX.md` 大清单。
- 实验入口在 `experiments/`，工具脚本在 `scripts/{matrix,summarize,verify,diagnose}/`，
  核心库与 `*_protocol.py` 留根目录（`train.py` 模块级依赖 protocol，勿移）。
- 同 seed 全配置同卡；不同 seed 才可跨卡并行。
- 除被测项外 seed/batch/lr/device/loss weighting 全一致；跨 scenario 只看相对排序；
  **跨数据集不比绝对 AUC**（只信同 seed 配对差）；**macro 与 pooled 端点不混入同一张表**。
- 单 seed 只作动机，不进高置信结论；跨 seed 变号不得用均值覆盖。
- 每个新结论先登记 claim/evidence/verdict/可降级边界，再跑。
- `.codebuddy/` 勿删。
