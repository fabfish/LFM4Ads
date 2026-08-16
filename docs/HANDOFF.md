# HANDOFF — 交接一页纸（给后续执行者）

> 本文是给接手实验的模型/人读的**速览**。完整证据见 [`DRIVERS.md`](DRIVERS.md)，
> 已可靠结论见 [`ANALYSIS.md`](ANALYSIS.md)，后续授权见 [`NEXT.md`](NEXT.md)。

---

## 0. 一句话状态

**cross 层 MoE 路线已正式关闭**：机制（router 特化 + 真实稀疏 dispatch）完全跑通，但收益为负——
**4× 容量收益 ≈ 0，稀疏化代价 ≈ −0.001**。

**2026-08-14 22:25 更新**：原先写的"瓶颈在 embedding 表（97.9% 参数）"是**过度推断，已被证伪**。
把占总参数 **99.19%** 的三张 ID 表（`video_id`/`author_id`/`music_id`，83,984,250 参数）：
- 推理期**全部清零**，test AUC 只掉 **0.000193**（噪声地板 0.001 内）；`upload_type`（**320 参数**）
  单独清零却掉 0.0053；
- **从零重训练时彻底剔除（E1，2 seed × 3 臂，哨兵 10/10 PASS）**，Δ_id = **{+0.000688, +0.000220}**
  → **PASS（死重成立）**，同时**参数 −99.31%（84.7M→0.58M）、每 epoch 快 1.48×**。

即：**84M 参数是死重与过拟合负担，不是瓶颈**。整个模型的预测能力来自约 **3 万参数**的 33 个低基数字段。

下一步方向因此从"给 embedding 加容量"改为 **特征信息侧**（E2/E3/E4，见 §4）。
证据：[E1 结论](./20260814-2225-E1结论-ID-embedding死重确认.md)、
[预注册+审计](./20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md)。

---

## 1. 已可靠知道什么（初级结论，按证据强度分级）

### A 级（多 seed + 配对 + 决定性对照，高置信）

1. **cross 层容量不是 AUC 瓶颈**（dense-widened 4× 加宽+非线性、零路由零稀疏，Δ_capacity 跨 seed 变号、
   |Δ|≤0.0007 < 噪声地板）。这是 capacity-MoE 关闭的直接证据。
2. **capacity-MoE 收益 = 容量收益(≈0) + 稀疏代价(≈−0.001)**，净效应必为负，不是实现问题。
3. **84M ID embedding 是死重，不是瓶颈**（2026-08-14 修订，原第 3 条"embedding 是瓶颈"已证伪）：
   - 三张 ID 表占 embedding 的 **99.96%**（83,984,250 / 84,017,390），其余 33 个字段合计仅 **33,140** 参数；
   - 推理期把这 84M 全部清零，test AUC `0.777490 → 0.777297`，**Δ = −0.000193**（噪声内）；
   - **E1 重训练确认 PASS**：from-scratch 剔除后 Δ_id = {+0.000688, +0.000220}（2/2 seed 地板内），
     参数 **−99.31%**、wall **1.48×**；哨兵 10/10 PASS；
   - 依赖度最高的字段是 `tag`(−0.0066) / `upload_type`(−0.0053，仅 320 参数) / `user_id`(−0.0049)；
   - `video_id` 平均每 ID 仅 **2.6** 次曝光、79.8% 的 ID 曝光≤2 次、**test 74.19% 样本的 video_id
     在 train 中从未出现**；**17.53% 的 embedding 参数永远收不到梯度**；
   - 原证据"冻结 embedding 后 AUC 反而更高"的正确读法是 **burden（负担）**，不是 bottleneck（瓶颈）。
   - 边界：**"无可测差异"≠"删掉更好"**（Δ 虽同为正但在噪声地板内）。
4. **三个实现 bug 已修复并固化**（见 §3）。

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
| capacity-MoE（cross 层） | **正式关闭** | 容量收益≈0 + 稀疏代价≈−0.001 |
| **embedding 容量（加宽）** | **正式关闭** | 现有 84M 容量本就没被用上（清零 Δ=−0.0002、剔除 Δ 在地板内），加宽前提不存在 |
| Stage B same-FLOPs MoE | FAIL | 同 seed 差 `[-,-,+]`、wall-clock 更慢 |
| TCMR 静态任务条件路由 | INCONCLUSIVE | 跨 seed 变号、未越噪声地板 |
| 共享残差持续学习 | blocked | G3 INCONCLUSIVE、gate=false |
| 下游留出域迁移 | G1 FAIL | moe-router 1/3 target 胜、停止 |

**核心教训（写进后续所有设计）**：加容量/加专家前，必须先用一个"widened 对照"证明**存在容量缺口**。
否则就是 cross 层 MoE 的重演——白送 4× 容量都换不到收益。

**核心教训 2（2026-08-14 新增，代价更低）**：判断"瓶颈在哪"**不能用参数占比**。
先做**零训练成本诊断**（参数集中度 / 曝光频次 / OOV 率 / 推理期字段消融），再决定是否值得开 GPU。
本轮就是靠它在 **1 分钟内**挡掉了一个已经写进路线图的 embedding-widened 长跑实验。

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
| **E1 入口（新）** | `main_field_ablation.py` | `full`/`idzero`/`iddrop` 三臂 from-scratch + 内置哨兵 |
| **E1 判定（新）** | `scripts/summarize/summarize_field_ablation.py` | 阈值硬编码防漂移，产出 `e1_decision.json` |
| **轻量基线（新，推荐默认）** | `iddrop` 臂 | **0.58M 参数、7.1s/epoch**，AUC 与 84.7M 全模型无可测差异 |

**默认实验口径**（后续所有公平对比照此执行）：
```
--freeze sparse --full-batch-loss --gpu-resident-data \
--lr 2e-4 --lr-router 1e-3 --batch-size 10000
```

**关键模型文件**：`model.py`（`CapacityCrossExpertLayer`/`DCNv2CapacityMoE`/`DenseWidenedDCNv2`）。
**数据**：`dataset.feather`，切分见 `dataset.py:Split`（date<20220503 train / [20220503,20220506) valid / ≥20220506 test）。

---

## 4. 下一步方向（有证据支持，按顺序）

**方向已随 §1.3 的修订改写**：不再"给 embedding 加容量"（前提已被证伪），改为**特征信息侧**。
预注册见 [20260814-2212 文档](./20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md)。

1. **E1（第一步，已完成 `done / PASS`）— ID embedding 死重的重训练确认**：三臂 from-scratch，
   `full`(36 字段) / `idzero`(三张 ID 表置零+冻结，架构与 full 完全同构) / `iddrop`(真移除，dim=330)。
   Δ_id = {+0.000688, +0.000220}（2/2 seed 在 0.001 地板内）→ **PASS（死重成立）**；
   参数 84,672,605 → **584,255（−99.31%）**，wall/epoch 10.6s → 7.1s。
   入口 `main_field_ablation.py`，矩阵 `scripts/run_field_ablation_matrix.sh`，实际成本 8 min。
   → **后续实验默认基线改为 `iddrop`（0.58M 参数）**，3 seed × 多配置已变得廉价。
   结论见 [E1 结论](./20260814-2225-E1结论-ID-embedding死重确认.md)。
2. **E2 长尾 ID 的可泛化表示**（已解锁，下一个要跑的）：ID 不可泛化（test 74% OOV），但其属性可以——
   用频次分桶 + 时序安全的统计编码 + tag/upload_type 交叉替换裸 ID，对照 `iddrop`/`idzero`。
   判据：同 seed 配对差越过 0.001 地板。
3. **E3 特征交互侧表达**：在轻量模型上加 FM/attention 式高阶交叉，参数量匹配对照。
4. **E4 天花板定位**：33 个低基数字段能达到的 AUC 上界，判断 0.78 是"模型不足"还是"信息不足"。

所有新实验必须预注册（用 `lfm4ads-experiment-audit-planner` skill 冻结哨兵/判定），
同 seed 同卡配对差 + 2–3 seed；**且先跑零成本诊断，再决定是否开 GPU**。

---

## 5. 必须遵守的规则（AGENTS.md 摘要）

- 文档放 `docs/`，命名 `YYYYMMDD-HHMM-主题.md`；每逻辑单元完成 `git commit` + `push`。
- 同 seed 全配置同卡；不同 seed 才可跨卡并行。
- 除被测项外 seed/batch/lr/device/loss weighting 全一致；跨 scenario 只看相对排序。
- 单 seed 只作动机，不进高置信结论；跨 seed 变号不得用均值覆盖。
- 每个新结论先登记 claim/evidence/verdict/可降级边界，再跑。
- `.codebuddy/` 勿删。
