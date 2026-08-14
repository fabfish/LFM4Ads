# HANDOFF — 交接一页纸（给后续执行者）

> 本文是给接手实验的模型/人读的**速览**。完整证据见 [`DRIVERS.md`](DRIVERS.md)，
> 已可靠结论见 [`ANALYSIS.md`](ANALYSIS.md)，后续授权见 [`NEXT.md`](NEXT.md)。

---

## 0. 一句话状态

**cross 层 MoE 路线已正式关闭**：机制（router 特化 + 真实稀疏 dispatch）完全跑通，但收益为负——
**4× 容量收益 ≈ 0，稀疏化代价 ≈ −0.001**。瓶颈在 **embedding 表（97.9% 参数）**，不在 cross 层（0.46%）。
下一步唯一有证据支持的方向是 **embedding / 特征交互侧**，且必须先证伪"容量缺口"再加容量。

---

## 1. 已可靠知道什么（初级结论，按证据强度分级）

### A 级（多 seed + 配对 + 决定性对照，高置信）

1. **cross 层容量不是 AUC 瓶颈**（dense-widened 4× 加宽+非线性、零路由零稀疏，Δ_capacity 跨 seed 变号、
   |Δ|≤0.0007 < 噪声地板）。这是 capacity-MoE 关闭的直接证据。
2. **capacity-MoE 收益 = 容量收益(≈0) + 稀疏代价(≈−0.001)**，净效应必为负，不是实现问题。
3. **84M embedding 表是过拟合源**：占 97.9% 参数，`lr=1e-3` 下解冻会让 dense 与 MoE **双双**从 epoch1
   崩到 0.67（共因）。`--freeze sparse` 后两臂绝对 AUC 反而更高。
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
| Stage B same-FLOPs MoE | FAIL | 同 seed 差 `[-,-,+]`、wall-clock 更慢 |
| TCMR 静态任务条件路由 | INCONCLUSIVE | 跨 seed 变号、未越噪声地板 |
| 共享残差持续学习 | blocked | G3 INCONCLUSIVE、gate=false |
| 下游留出域迁移 | G1 FAIL | moe-router 1/3 target 胜、停止 |

**核心教训（写进后续所有设计）**：加容量/加专家前，必须先用一个"widened 对照"证明**存在容量缺口**。
否则就是 cross 层 MoE 的重演——白送 4× 容量都换不到收益。

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

**默认实验口径**（后续所有公平对比照此执行）：
```
--freeze sparse --full-batch-loss --gpu-resident-data \
--lr 2e-4 --lr-router 1e-3 --batch-size 10000
```

**关键模型文件**：`model.py`（`CapacityCrossExpertLayer`/`DCNv2CapacityMoE`/`DenseWidenedDCNv2`）。
**数据**：`dataset.feather`，切分见 `dataset.py:Split`（date<20220503 train / [20220503,20220506) valid / ≥20220506 test）。

---

## 4. 下一步方向（有证据支持，按顺序）

1. **embedding 瓶颈定位对照**（最高性价比）：给 embedding/特征交互侧加容量（更高 dim 或特征交叉扩展），
   做"widened vs dense"对照。不涨 → embedding 也不是容量瓶颈，转向数据/任务；涨 → 找到真缺口，再考虑条件化容量。
2. **特征交互侧表达**：DCNv2 cross 层只做 `x0⊙(W·x)` 低阶交互，未覆盖 FM/attention 式高阶交叉。
3. **数据/任务侧**：AUC 天花板约 0.78，dense 已贴近；模型侧全部证伪则问题在数据/任务。

所有新实验必须预注册（用 `lfm4ads-experiment-audit-planner` skill 冻结哨兵/判定），
同 seed 同卡配对差 + 2–3 seed，先证伪容量缺口再加容量。

---

## 5. 必须遵守的规则（AGENTS.md 摘要）

- 文档放 `docs/`，命名 `YYYYMMDD-HHMM-主题.md`；每逻辑单元完成 `git commit` + `push`。
- 同 seed 全配置同卡；不同 seed 才可跨卡并行。
- 除被测项外 seed/batch/lr/device/loss weighting 全一致；跨 scenario 只看相对排序。
- 单 seed 只作动机，不进高置信结论；跨 seed 变号不得用均值覆盖。
- 每个新结论先登记 claim/evidence/verdict/可降级边界，再跑。
- `.codebuddy/` 勿删。
