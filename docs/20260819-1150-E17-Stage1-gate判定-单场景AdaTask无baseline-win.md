# E17 Stage 1 gate 判定：单场景无 baseline win

- 记录时间：2026-08-19 11:50
- 预注册：[E17 复审与预注册](./20260818-1815-结论复审与E17单场景AdaTask-win-case预注册.md)（设计、判定规则、端点、地板均已冻结，本文不改任何规则）
- 运行日志：[run journal](./run_journal/e17-stage1.md)

## 结论（一句话）

预注册的 gate 双条件（Δ_T(s0)>0 且 Δ_A(s0)>0）未同时满足：
- **baseline win（T vs S）**：Δ_T(s0) 两 seed 异号（−0.002450 / +0.000170），判定 **INCONCLUSIVE**。
- **AdaTask win-case（A vs T）**：Δ_A(s0) 两 seed 同正（+0.000453 / +0.000604），但均值 +0.000528 未越过地板 0.000832，判定 **不 PASS**。

→ gate = **no-go**，不扩 Stage 2（seeds 303/404）。

## 术语定义

- **s0**：目标场景（tab=0），开跑前冻结，禁止看结果后更换。
- **S / T / A 三臂**：S=共享 AdamW（所有场景一套优化器状态）；T=每场景独立 Adam 状态；A=T 的基础上对 sparse expert 的预条件后方向乘 task-axis AU 因子（AdaTask）。三者模型参数、初始化、数据顺序、超参完全一致，唯一变量是优化器状态分配方式。
- **端点**：主端点 test s0 AUC；保护端点 test macro AUC。三臂均按同一端点（valid macro argmax）选 epoch。
- **配对差（Δ）**：同 seed 下实验臂减对照臂。
- **噪声地板**：`2 × sample_SD(对照臂 test 值)`，n=2（1 自由度），仅作筛选参考，非正式阈值。

## 实验口径

| 项 | 值 |
|---|---|
| 数据 | KuaiRand-27K 原切分 |
| 模型 | 轻量 MoE（K=5, top_k=2），869,750 参数 |
| 优化目标 | one-vs-rest：任务组 {s0, rest(tab≠0)}，loss=(L_s0+L_rest)/2 |
| 优化器 | lr=5e-4、betas=(0.9,0.999)、wd=0.01；A 臂 α=0.5、β_AU=0.99、clip=[1/3,3] |
| seed | 101（cuda:0）、202（cuda:1），各 seed 的 S→T→A 同卡顺序 |
| 选择 | valid macro argmax，patience 10，max 20 epoch |

启动前审计 10/10 PASS（组梯度等价、f=1 更新等价、状态不变、角色隔离、task-axis、参数量、路由语义、效率预算等），证据见 `cache/audit/adatask_win_s0_27k_siteA/prelaunch_audit.json`。断点续跑 bitwise 3/3 PASS。

## 结果

### 主端点：test s0 AUC

| arm | s101 | s202 |
|---|---|---|
| S | 0.711821 | 0.709789 |
| T | 0.709371 | 0.709959 |
| A | 0.709824 | 0.710563 |

配对差：

| 差 | s101 | s202 | mean | 方向一致性 |
|---|---|---|---|---|
| Δ_T(s0)=T−S | −0.002450 | +0.000170 | −0.001140 | 异号 |
| Δ_A(s0)=A−T | +0.000453 | +0.000604 | +0.000528 | 2/2 正 |

地板：floor_T_s0=0.002873，floor_A_s0=0.000832。

### 保护端点：test macro AUC

| arm | s101 | s202 |
|---|---|---|
| S | 0.736954 | 0.736184 |
| T | 0.731636 | 0.733267 |
| A | 0.737940 | 0.735813 |

配对差：

| 差 | s101 | s202 | mean | 方向一致性 |
|---|---|---|---|---|
| Δ_T(macro)=T−S | −0.005318 | −0.002917 | −0.004118 | 2/2 负 |
| Δ_A(macro)=A−T | +0.006305 | +0.002546 | +0.004426 | 2/2 正 |

地板：floor_T_macro=0.001089，floor_A_macro=0.002307。A 臂 clip_dominated=False（clip_rate 均值 0.008，远低于 0.3）。

## 判定

按预注册 §6：

1. **baseline win（T vs S）**：Δ_T(s0) 非"全正"也非"全负" → **INCONCLUSIVE**；Δ_T(macro) 2/2 负且越地板 → 登记为负向相关性（纯 task-state 隔离在 s0 无益、在 macro 有害）。
2. **AdaTask win-case（A vs T）**：Δ_A(s0) 2/2 正但 mean +0.000528 < floor_A_s0 0.000832 → **未越地板，不 PASS**；Δ_A(macro) 2/2 正且越地板 → 登记为正向相关性。

**gate = no-go**：预注册 §5 的继续条件（Δ_T(s0)>0 且 Δ_A(s0)>0）不满足，不扩 Stage 2。

## 含义

- 否定了 F1 短训口径的乐观信号（+0.020936/+0.041955）：纯 task-state 隔离在全数据下不成立，且方向为负。这证实了 F1 复审中"子采样短训会误导方向"的判断。
- AdaTask 相对 T 有微小正增量（s0 与 macro 均 2/2 正），但 s0 上的量级低于筛选地板，不能作为该方向可行的证据。
- 详细机制（AdaTask 增益被 rest 场景吸收）与后续选项见 [E17 完整结论](./20260819-1155-E17结论-状态隔离无益AdaTask增益在rest.md)。
