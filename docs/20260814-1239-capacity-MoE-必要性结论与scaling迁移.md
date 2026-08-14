# Capacity-MoE 必要性验证 — 结论与 scaling 迁移

- 创建时间：2026-08-14 12:39
- 状态：**`auth=done` / 部分结论已被下游修正**
- 上游驱动：[20260814-1120-capacity-MoE-必要性验证驱动.md](./20260814-1120-capacity-MoE-必要性验证驱动.md)
- 上级总纲：`docs/DRIVERS.md` §3.6

> ## ⚠️ 修正声明（2026-08-14 21:11 追加）
>
> 本文档 §四 的两条机制结论**已被证伪/降级**，请以
> [根因定位与决定性容量实验](./20260814-2111-专家无收益根因定位与决定性容量实验.md) 为准：
>
> 1. **「稀疏化崩 AUC（0.76→0.67）」不成立**——崩溃的真因是占总参数 **97.9%** 的 84M embedding 表
>    在 `lr=1e-3` 下全程解冻造成的过拟合，**dense-continued 臂同样崩溃**（这是共因，与稀疏化无关）。
>    加 `--freeze sparse` 后两臂均不再崩溃。
> 2. **「router 越特化 AUC 越差」应降级**——熵与 AUC 同为"训练进度"的函数，不可读作因果。
> 3. **Δ_necessity 已从 −0.0019 收窄到 −0.0003**（`lr=2e-4` + 冻结 embedding + 修复 lb 8× 放大 bug），
>    即从"明确为负"降级为**统计平局**（噪声地板内）。
> 4. 本文档 §五「冻结的最优设置」与 §六「scaling 迁移规则」**建立在被污染的口径上，已作废**；
>    新的收益分解结论是：**4× 容量收益 ≈ 0，稀疏化代价 ≈ −0.001**。
>
> 下方原文保留作为过程记录，不再作为结论依据。

---

## 一、结论一句话

**capacity MoE 的"改变必要性"在当前规模（360 维、K=4、top_k=2）下不成立：**
剥离"继续训练红利"后，MoE 相对 dense-continued 的同 seed 配对差 **Δ_necessity = [−0.0019, −0.0006]**，
2 seed 均为负。原 smoke 报告里的 "+0.0025 微弱正向" 被证实是 **"1 个 epoch 微调红利"**，而非 MoE 结构红利——
dense 继续训练同样拿到 +0.0023~+0.0036，且拿得比 MoE 更多。

机制层面（router 特化 + 真实稀疏 dispatch）再次确认成立（哨兵 PASS），但**稀疏化本身在伤害模型**：
sparse 阶段 valid AUC 单调崩溃，且"router 越特化、AUC 越差"。

---

## 二、三臂结果（冻结口径）

| 指标 | seed 42 | seed 123 |
|---|---|---|
| A: dense 冻结 test AUC | 0.7775 | 0.7775 |
| A′: dense-continued test AUC（best @ epoch 1） | 0.7811 | 0.7798 |
| B: capacity-MoE test AUC（best @ warmup epoch 1） | 0.7793 | 0.7792 |
| Δ_smoke = B − A（原 smoke 口径） | **+0.0018** | **+0.0017** |
| **Δ_necessity = B − A′（本阶段主指标）** | **−0.0019** | **−0.0006** |

- 三臂共享 `dim=360, K=4, embed_dim=10`，dense 预训练 ckpt 完全一致（`cache/dcnv2_vanilla.pt`）。
- A′ 与 B 训练协议逐字段一致（lr=1e-3、AdamW(0.9,0.999)、batch=10000、`sample` 加权、16 epoch 跑满），
  唯一差异是被测结构；B 额外有 warmup 2 + lb loss（已显式声明）。
- 逐 seed 配对差同卡：seed 42 卡 0、seed 123 卡 1，无设备混杂因子。

### 测量口径说明

- device：seed42→`cuda:0`、seed123→`cuda:1`（同 seed 全臂同卡）。
- lr=1e-3、batch_size=10000、beta2=0.999、loss_weighting=`sample`、`min_epochs=max_epochs=16`（关闭 early-stop）。
- test AUC = best-valid-state 的 test 评估（`best_state` 回溯）。

---

## 三、必要性判定（预注册规则回填）

按驱动 §六冻结规则逐条核对：

| 哨兵条件 | seed42 / seed123 | 结果 |
|---|---|---|
| 前向/反向不崩 | ✓ / ✓ | 通过 |
| dispatch 真实稀疏（实际执行专家 == top_k=2） | True / True | 通过 |
| 末 epoch 熵 ≤ log K − 0.15（≤1.2363） | 0.7451 / 0.7401 | 通过 |
| 2 seed Δ_necessity 均 > +0.001 且同号 | **−0.0019 / −0.0006**（均为负） | **不通过** |

**判定 = INCONCLUSIVE（方向为负）**。

- 机制哨兵（router 特化 + dispatch）单独判 **PASS**：熵从 log K=1.3863 降到 0.74，12/12 结论再确认。
- 必要性主张 **不成立**：2 seed Δ_necessity 均为负，在"偏袒 dense（不折抵 MoE warmup 的 K 倍 FLOPs）"的
  保守口径下，MoE 仍略差于 dense-continued。
- 驱动 §六补充规则命中：「若 2 seed Δ_necessity 均为负，判定 INCONCLUSIVE 且方向为负，结论文档必须显式写
  '必要性不成立'，不得写成微弱正向」——本结论照此执行。

---

## 四、关键机制发现（比"负结论"更有价值的部分）

### 4.1 dense-continued 同样过拟合（"继续训练红利"的真实形状）

dense 继续训练 16 epoch 的 valid AUC 轨迹（两 seed 高度一致）：

```
s42:  epoch1=0.7791 → 4=0.7639 → 8=0.7247 → 12=0.6905 → 16=0.6711
s123: epoch1=0.7783 → 4=0.7631 → 8=0.7205 → 12=0.6875 → 16=0.6710
```

- dense 从冻结态继续训练，**第 1 个 epoch 就 +0.0016~+0.0036**（0.7775 → 0.7791/0.7783），随后**单调过拟合**
  到 0.671。
- 这直接证明原 smoke 的 Δ 口径有缺陷：dense 参照被"冻结"在 0.7775，而 MoE 被允许"多训"，天然占便宜。

### 4.2 MoE sparse 阶段单调崩溃，且"router 越特化、AUC 越差"

MoE（warmup 2 + sparse 16）轨迹（两 seed 高度一致）：

```
        warmup1=0.7783/0.7785 (ent 1.20/1.16)   ← best state 在此
        warmup2=0.7764/0.7759 (ent 1.09/1.06)
sparse: ep3=0.7611/0.7604 (ent 1.16/1.19)
        ep5=0.7382/0.7373 (ent 1.05/1.06)
        ep9=0.7012/0.6995 (ent 0.88/0.89)
        ep13=0.6794/0.6812 (ent 0.78/0.79)
        ep16=0.6738/0.6711 (ent 0.75/0.74)      ← 熵最低、AUC 最低
```

- **router 特化是成立的**（熵 1.386 → 0.74，离开 log K 0.64 nats，dispatch 校验 True）。
- **但特化与 AUC 反相关**：熵最低（router 最特化）的 sparse 末段，valid AUC 最低（0.671~0.674）。
  换句话说，真实稀疏 dispatch 让模型学会了"有偏向地路由"，但这份偏向**没有带来下游收益，反而在毁 AUC**。
- 结论边界：**"router 特化机制成立" ≠ "稀疏化对 AUC 有益"**。二者在当前规模下是反向关系。这是驱动 §5.4
  哨兵明确要测的风险，现已坐实。

---

## 五、冻结的最优设置（scaling 唯一基准）

在"必要性不成立"的前提下，仍需给"若要在更大规模重试"固定一个**唯一起点**，避免以后漫无边际调参：

| 项 | 冻结值 | 依据 |
|---|---|---|
| K | 4 | 12-run + 本矩阵一致 |
| top_k | 2 | 一致优于 top_k=1（Δ_smoke +0.0018 vs +0.0012 量级） |
| noise_scale | 0.1 | upcycle 乘法扰动，打破专家对称性的最优档 |
| lb_alpha | 0.001 | 防坍缩又不阻碍特化的平衡点 |
| warmup_epochs | 2 | 占总 epoch 的 12.5%（2/16） |
| lr / batch / beta2 | 1e-3 / 10000 / 0.999 | 与 dense 全一致 |
| loss 加权 | `sample` | 复用仓库统一口径 |

> 注：该设置是"router 特化最强的可跑配置"，**不是"AUC 最优配置"**——本矩阵已证明它 AUC 上跑不赢
> dense-continued。冻结它只作为 scaling 讨论的机制锚点，不隐含"该设置应被推广"。

---

## 六、scaling 迁移规则（讨论，不跑大模型）

目标：从探索规模（360 维、K=4）迁移到更大规模（更大的 dim / 更大的 K）时，设置如何变。以下规则以
**"熵目标区间"为锚**——因为本阶段发现 router 熵是唯一可稳定观测、且与特化强相关的量。

### 6.1 以"熵目标区间"锚定 lb_alpha

- 把 sparse 稳态熵落在 **`log K − 0.6 ~ log K − 0.35`** 区间作为调参目标：
  - K=4 时该区间 = 0.79 ~ 1.04（本矩阵末段熵 0.74 略低于区间，属"过度特化"，与 AUC 反相关一致）。
- **lb_alpha 起点 ∝ 1/K**：K 增大时专家被均摊的概率下降、负载失衡风险上升，lb 需更小以免压死特化。
  经验起点 `lb_alpha ≈ 0.004 / K`（K=4 → 0.001；K=8 → 0.0005；K=16 → 0.00025），再按熵区间微调。
- **不许**把 lb 调到"熵回升到接近 log K"（那是特化被压死，回到 V2 老问题）。

### 6.2 warmup 占比保持 15%–25% 总 epoch

- 本矩阵 warmup=2/16=12.5%，偏少（warmup 结束即 sparsify 触发崩溃）。更大规模建议 warmup 占总 epoch
  的 15%–25%，让专家在软路由下充分分化后再稀疏化。
- warmup 期间是 K 倍 FLOPs（全专家激活），这是 MoE 的"隐性算力溢价"，迁移时必须像本矩阵一样显式声明、不折抵。

### 6.3 noise_scale 维持 0.05~0.1

- upcycle 乘法扰动的角色是"打破专家对称性"，与规模弱相关。维持 0.05~0.1 即可，无需随 K 放大。

### 6.4 top_k 以 `top_k ≈ K/2` 起步

- K=4 → top_k=2 已是最优。更大 K 建议 top_k 随 K 线性起步（K=8 → 4，K=16 → 8），
  保持"激活容量 / 总容量"比例约 1/2，避免 top_k 过小导致路由方差爆炸。

### 6.5 必须保留的公平口径（任何规模都不变）

- **dense-continued 对照臂**：任何"MoE 是否更好"的主张，必须与"同样继续训练的 dense"比配对差，禁止再犯
  原 smoke 的"对冻结 dense 比"错误。
- **同 seed 同卡配对差 + 多 seed**：探索 2 seed 只能判方向，正式需 3 seed + 统计显著。
- **诚实标注 K 倍容量 + warmup FLOPs**：不把 K 倍参数 / warmup 算力算作"免费"。

### 6.6 一个前置判断（重要）

本矩阵的负结果**不是"MoE 不行"的终局证据**，而是"小规模 + 短训练下 MoE 的稀疏化代价 > 容量收益"的证据。
若要在更大规模重试，**前提是先把"稀疏化为何伤害 AUC"这个机制问题解决**（见 §七 backlog），否则单纯放大
dim/K 只会把同样的崩溃放大。

---

## 七、Backlog（后续关卡，不在本次范围）

1. **诊断"稀疏化为何崩 AUC"**：候选方向——
   (a) top-k 稀疏丢掉了专家间的软加权（STE 硬 gate 的方差）；
   (b) 稀疏化后每专家只看到自己分到的样本子集，专家权重在少数样本上过拟合；
   (c) warmup→sparse 的 top_k 跳变（4→2）造成分布偏移。
   可用"渐进稀疏化"（top_k 逐步降）或"软硬混合 gate"消融定位。
2. **专家利用率度量**：统计每专家实际接收的 token 占比，确认是否有专家被闲置（即便 lb=0.001）。
3. **3-seed 正式矩阵**：仅在上述机制修复后才有意义；当前 2 seed 负向结果下**不授权**扩 3-seed。
4. **AdaTask 三模式**（本计划第二部分）：在 capacity 架构上跑 none/encourage/suppress + 路由特化分析，
   观察 AdaTask 梯度调制能否改善"稀疏化崩 AUC"（见下一步驱动）。

---

## 八、时间成本实测（用户要求透明）

| 阶段 | 实测 wall time |
|---|---|
| dense 预训练 | 0（ckpt 已缓存 338MB） |
| dense-continued 16 epoch | ≈ 19 min（~72s/epoch） |
| MoE warmup 2 + sparse 16 epoch | ≈ 42 min（warmup ~96s、sparse ~160s/epoch） |
| 单卡串行总计 | ≈ 61 min |
| 两卡矩阵（并行） | ≈ 61 min（t+55min 完成，含 ~5min 数据加载/评估开销） |

---

## 九、结论边界登记（audit-planner §5，启动后不得改判更弱）

| # | claim | evidence | 判定 | 可降级为 |
|---|---|---|---|---|
| 1 | MoE 相对 dense 有 +0.0025 微弱正向 | 本矩阵 Δ_smoke=[+0.0018,+0.0017] | **降级**：该 Δ 是对冻结 dense 的差，混入微调红利 | "对冻结 dense 有 ~+0.002 微弱正向，但非结构红利" |
| 2 | MoE 改变必要性成立 | Δ_necessity=[−0.0019,−0.0006] 均为负 | **错误**（方向为负） | "必要性不成立（当前规模下稀疏化代价 > 容量收益）" |
| 3 | router 特化机制成立 | 熵 1.386→0.74、dispatch True | **成立** | 无（机制层面，与 AUC 无关） |
| 4 | 稀疏化对 AUC 有益 | sparse 段 valid AUC 0.76→0.67 单调崩 | **错误** | "稀疏化在当前规模下伤害 AUC" |
| 5 | 更长训练让 MoE 收敛 | sparse 末段 AUC 仍在下行、无企稳 | **错误** | "更长训练下 sparse AUC 持续崩溃" |

---

## 十、回填

- `docs/DRIVERS.md` §3.6：追加必要性验证结论（`auth=done`、verdict=INCONCLUSIVE 方向为负、配对差表）。
- 产物：`result_capacity_moe_s{42,123}_necessity.csv`、`result_dense_cont_s{42,123}_necessity.csv`、
  `cache/capacity_moe_history_s{42,123}_necessity.json`、`logs/capacity_moe_s{42,123}_necessity.log`。
- 代码：`main_moe_capacity.py`（`--train-dense-ref` 臂）、`scripts/run_necessity_matrix.sh`。
