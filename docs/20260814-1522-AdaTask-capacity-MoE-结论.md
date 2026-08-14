# AdaTask 三模式 × capacity MoE — 结论与路由特化分析

- 创建时间：2026-08-14 15:22
- 状态：**`auth=done`**（capacity 架构三模式矩阵完成）
- 上游：`docs/20260814-1239-capacity-MoE-必要性结论与scaling迁移.md`（必要性验证 INCONCLUSIVE 方向为负）
- 上级总纲：`docs/DRIVERS.md` §3.7

---

## 一、结论一句话

在 capacity MoE（full-rank 专家 + 真实 top-k 稀疏 dispatch）上复跑 AdaTask 三模式，**router 特化机制在
三模式下都成立（熵均离开 log K），但 AdaTask 的 encourage/suppress 调制既没有改善 AUC（相对 none 都是负向），
也没有按"encourage 促进特化、suppress 抑制特化"的直觉方向调制 router**——实测方向恰好相反：suppress 熵最低
（最特化），none 熵最高。真实稀疏 dispatch 引入了一个旧全连接 MoE 没有的新现象：**未激活专家无梯度 → AU 冻结**，
使 AdaTask 的 AU 调制因子对弱专家基于"陈旧 AU"，调制语义被稀疏化改变。

---

## 二、实验口径

| 项 | 值 |
|---|---|
| 架构 | `DCNv2CapacityMoE`（K=4 full-rank 专家 + top_k=2 真实稀疏 + STE + lb） |
| 基座 ckpt | `cache/dcnv2_vanilla.pt`（与 necessity 验证一致） |
| seed / device | 42 / `cuda:0`（单 seed） |
| lr / alpha / beta | 1e-3 / 0.5 / 0.99 |
| batch_size | 16384 |
| epochs | 5（warmup 2 全软路由 + sparse 3 真实稀疏） |
| noise_scale / lb_alpha | 0.1 / 0.001 |
| 训练集 | train + valid（沿用 AdaTask 历史口径，非只 train） |
| 调制 | AdaTaskOptimizer 旧接口 `mode`（仅调制 expert，router 不调制） |

---

## 三、路由特化轨迹（主结果）

### 3.1 逐 epoch 熵（三模式，log K = 1.3863）

| epoch | phase | none | encourage | suppress |
|---|---|---|---|---|
| 1 | warmup | 0.923 | 0.862 | 0.780 |
| 2 | warmup | 0.899 | 0.875 | 0.820 |
| 3 | sparse | 0.772 | 0.750 | 0.694 |
| 4 | sparse | 0.804 | 0.773 | 0.707 |
| 5 | sparse | 0.775 | 0.771 | 0.735 |

- 三模式熵均离开 log K（1.3863），router 特化成立（dispatch 校验 True，真实稀疏）。
- **方向反直觉**：熵排序 `suppress < encourage < none`（suppress 最特化、none 最不特化）。
  而 AdaTask 的直觉是 encourage 促进特化、suppress 抑制特化。

### 3.2 专家 dispatch 利用率（最终 sparse epoch）

| mode | L0 | L1 | L2 |
|---|---|---|---|
| none | [0.210, 0.161, **0.384**, 0.246] | [0.275, 0.257, 0.237, 0.231] | [0.270, 0.282, 0.208, 0.241] |
| encourage | [0.263, 0.274, **0.339**, 0.124] | [0.203, 0.276, 0.284, 0.236] | [0.235, 0.288, 0.286, 0.191] |
| suppress | [0.175, 0.083, **0.418**, 0.324] | [0.216, 0.262, 0.229, 0.293] | [0.282, 0.224, 0.235, 0.259] |

- **第一层（L0）专家利用率明显不均衡**：E2 一致接收最多 token（0.34~0.42），E1 最少（0.08~0.27）。
  `lb_alpha=0.001` 没有阻止第一层的专家坍缩倾向。L1/L2 相对均衡（0.2~0.29）。
- **suppress 的不均衡最重**（L0：E2=0.418、E1=0.083）——与它"熵最低（最特化）"一致。

### 3.3 AU 与 dispatch 利用率正相关（none 模式）

| 层 | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| L0 专家均值 AU | 0.02 | 0.01 | **0.06** | 0.05 |
| L1 专家均值 AU | 0.01 | 0.03 | 0.04 | 0.03 |
| L2 专家均值 AU | 0.02 | 0.01 | 0.01 | 0.01 |

- L0 的 E2（dispatch 0.384）AU 最高（0.06），E1（dispatch 0.161）AU 最低（0.01）——**AU 与 dispatch 利用率正相关**。

---

## 四、关键机制发现

### 4.1 真实稀疏下的"AU 冻结"（旧 MoE 没有的新维度）

capacity MoE 的稀疏 dispatch 只对被 top-k 选中的专家执行前向，**未选中专家完全无梯度**，因此
AdaTask 的 AU（梯度平方 EMA）对未选中专家**不更新、保持旧值（接近 0）**。这与旧 `DCNv2MoE`
（全算后加权，所有专家每步都有梯度）有本质区别：

- 旧 MoE：所有专家 AU 每步更新 → 调制因子反映"实时梯度强度"。
- capacity MoE：弱专家 AU 冻结在低值 → 调制因子（`(mean_au/au)^alpha` 或逆）对弱专家基于"陈旧 AU"，
  会把弱专家**永久钉在**低 AU 状态（因为没被选中就没有梯度、没有梯度 AU 就不更新、AU 低就继续不被选中……
  形成自我强化的"专家饿死"循环）。

这解释了 §3.2 的专家坍缩：稀疏化让"被选中的专家越来越强、没被选中的越来越弱"，lb=0.001 不足以打破。

### 4.2 调制方向反直觉的根因（旧接口只调制 expert）

AdaTaskOptimizer 的旧接口 `mode=encourage/suppress` 只设 `expert_mode`，`router_mode` 恒 `none`。
所以 router 的熵**不被直接调制**，只受 expert 权重变化的间接影响（通过 cross 层残差输入传导）。
`suppress` 抑制主导 expert 的更新 → 主导 expert 的权重变化更慢 → 其输出更稳定 → 下游 router 反而更容易
特化（熵更低）。这是"suppress → 更特化"反直觉结果的机制解释，但**仅 1 seed，不足以作稳定结论**。

### 4.3 AUC 层面：三模式均低、调制无益

| mode | mean AUC（8 scenario） | Δ vs none |
|---|---|---|
| none | 0.6907 | — |
| encourage | 0.6872 | −0.0035 |
| suppress | 0.6899 | −0.0008 |

- 三模式 mean AUC 都 ~0.69，与必要性验证的"稀疏化崩 AUC"一致（冻结 dense 0.7775 的对照下全面下坠）。
- encourage / suppress 相对 none **都是负向**（且 encourage 负向更明显），AdaTask 调制在 capacity 稀疏
  架构下**没有兑现任何 AUC 收益**，与旧 MoE 上"encourage 促进特化"的叙事不同。

---

## 五、结论边界登记（启动后不得改判更弱）

| # | claim | evidence | 判定 |
|---|---|---|---|
| 1 | capacity 稀疏下 router 特化成立 | 三模式熵均离开 log K、dispatch True | 成立 |
| 2 | encourage 促进特化 / suppress 抑制特化 | 熵排序 suppress<encourage<none，方向相反 | **需降级**（反直觉，且仅单 seed） |
| 3 | AdaTask 调制改善 capacity AUC | Δ encourage −0.0035 / suppress −0.0008 | **不成立**（均负向） |
| 4 | 稀疏化引入专家 AU 冻结 | 弱专家 AU≈0.01、强专家 AU=0.06，与 dispatch 正相关 | 成立（机制层面） |
| 5 | lb=0.001 阻止专家坍缩 | L0 E2 dispatch 0.34~0.42 | **不成立**（仍坍缩） |

---

## 六、Backlog（与 necessity 验证 §七 汇合）

1. **专家饿死循环是"稀疏化崩 AUC"的首要嫌疑**：稀疏 dispatch + AU 冻结 → 专家坍缩 → 有效容量下降。
   修复方向：(a) 提高 lb（但这会阻碍特化，需回到 necessity 的熵区间锚定重扫）；(b) 加"专家保底探索"
   （ε-greedy dispatch / 最小 dispatch 约束）；(c) 用"软硬混合 gate"替代 STE 硬 gate。
2. **旧 AdaTask 接口对 router 不调制**：若要真正"促进/抑制路由特化"，应改用新接口 `--target router` 或
   直接设 `router_mode`，而非旧 `mode`（只动 expert）。
3. **三模式起始权重不一致**：`upcycle_from_dense` 的噪声随 RNG 状态跨 mode 推进，三模式起始噪声不同。
   若要做严格配对，需在每 mode 前重置 RNG 或固定 upcycle 噪声。
4. **仅 1 seed**：以上反直觉方向与负向 Δ 都是单 seed 动机级证据，不能进高置信结论。

---

## 七、产物与回填

- 代码：`main_adatask.py`（`--arch capacity` 分支 + `routing_snapshot` + `train_adatask_capacity`）。
- 数据：`cache/adatask_results_cap_k4_tk2_s42.csv`、`cache/adatask_au_{mode}_cap_k4_tk2_s42.json`、
  `cache/adatask_capacity_routing_{mode}_cap_k4_tk2_s42.json`、`logs/adatask_capacity_s42.log`。
- 回填：`docs/DRIVERS.md` §3.7 追加 AdaTask 三模式结论。
