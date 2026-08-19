# E17 Stage 1 gate 判定：单场景 s0 下纯状态隔离无 baseline win，AdaTask 增量未达地板

- 创建时间：2026-08-19 11:50
- 状态：**`done / gate=no-go`**（Stage 1 筛选完成，判定 INCONCLUSIVE；不追加 303/404，Stage 2 不启动）
- 预注册（判定规则来源，本结论文档不改任何规则）：
  [结论复审与 E17 预注册](./20260818-1815-结论复审与E17单场景AdaTask-win-case预注册.md)
- 判定 json：`cache/adatask_win_s0_27k_siteA/e17_decision.json`
- 本实验在 site A（2 卡）实跑；六臂（S/T/A × seed 101/202）全部完成。

---

## §0 术语与度量定义（继承，不重复定义）

- **目标场景 s0**：`tab=0`，开跑前冻结。训练用全部 15 场景，主检验只固定 s0。
- **S/T/A 三臂**：S=共享 AdamW 基线；T=纯任务状态隔离（`(Adam_s0+Adam_rest)/2`）；
  A=在 T 状态之上，对 sparse-expert 预条件后方向乘 task-axis AU 因子（`alpha=0.5, beta_AU=0.99, clip[1/3,3]`）。
- **baseline win**：`Δ_T(s0) = AUC_s0(T) − AUC_s0(S)`，同 seed 配对。
- **AdaTask win-case**：`Δ_A(s0) = AUC_s0(A) − AUC_s0(T)`，同 seed 配对。
- **端点对称性**：三臂都用同一 valid macro 精确 argmax 选 epoch/early-stop，再用 test s0 AUC 判 win；
  三臂端点和模型选择规则完全对称（可核对每个 run json 的 `best_epoch` 与 `selection` 字段）。
- **判定枚举**：只用 PASS / INCONCLUSIVE / FAIL。Stage 1 只作 go/no-go，不作正式 PASS/FAIL。
- **地板（Stage 1 仅 2 seed，参考值）**：`floor_T_s0 = 2×sample_SD(S 两 seed)`，`floor_A_s0 = 2×sample_SD(T 两 seed)`；
  正式地板需 4 seed（§0 预注册），此处只作量级参考，不用于正式判定。

---

## 1. 目标

在预注册的目标场景 s0 上，回答两个问题（Stage 1 筛选，go/no-go）：

1. 纯任务状态隔离（T）相对共享优化器（S）是否形成 **baseline win**（`Δ_T(s0)>0`）；
2. 真 task-axis、post-preconditioner 的 AdaTask（A）在 T 之上是否有 **增量 win**（`Δ_A(s0)>0`）。

go 条件（预注册 §5，需**同时**满足）：
`Δ_T(s0)>0` ∧ `Δ_A(s0)>0` ∧ `A−T test macro ≥ −0.002` ∧ 无 clip_dominated/覆盖≤2/provenance 失败。

## 2. 环境

| 项 | 值 |
|---|---|
| site | A；namespace `cache/adatask_win_s0_27k_siteA` |
| 数据 | KuaiRand-27K，train/valid/test 原切分不变 |
| 架构 | 轻量 `DCNv2MoE(K=5, top_k=2)`，869,750 参数 |
| loss | `L=(L_s0+L_rest)/2`，三臂一致；optimization 任务组 `{s0, rest(tab!=0)}` |
| batch / lr | 10,000 / 5e-4 |
| epoch | max 20，patience 10 |
| seeds | 101→cuda:0，202→cuda:1（预注册筛选 seed） |
| 审计 | Stage 0 十项哨兵 10/10 PASS；断点续跑 bitwise 3/3 PASS |

## 3. 结果（六臂完整读数）

| seed | 臂 | test s0 AUC | test macro AUC | best_valid_macro@ep | clip_dominated | 路由覆盖 |
|---|---|---|---|---|---|---|
| 101 | S | 0.711821 | 0.736954 | 0.743160@8 | False | 5 |
| 101 | T | 0.709371 | 0.731636 | 0.736406@5 | False | 5 |
| 101 | A | 0.709824 | 0.737940 | 0.743308@9 | False | 5 |
| 202 | S | 0.709789 | 0.736184 | 0.740192@8 | False | 5 |
| 202 | T | 0.709959 | 0.733267 | 0.739362@11 | False | 5 |
| 202 | A | 0.710563 | 0.735813 | 0.742271@17 | False | 5 |

配对差（同 seed）：

| seed | Δ_T(s0)=T−S | Δ_A(s0)=A−T | A−T macro |
|---|---|---|---|
| 101 | **−0.002450** | +0.000453 | +0.006305 |
| 202 | +0.000170 | +0.000604 | +0.002546 |
| **mean** | **−0.001140** | +0.000528 | +0.004426 |

地板（2 seed 参考）：`floor_T_s0 ≈ 0.002873`，`floor_A_s0 ≈ 0.000832`。

## 4. Stage 1 gate 判定（照预注册 §5 冻结表）

| go 条件 | 读数 | 满足 |
|---|---|---|
| 1. Δ_T(s0) > 0 | 101 = −0.002450，202 = +0.000170（**变号**） | **否** |
| 2. Δ_A(s0) > 0 | +0.000453 / +0.000604（2/2 正） | 是 |
| 3. A−T macro ≥ −0.002 | +0.006305 / +0.002546 | 是 |
| 4. 无 clip/覆盖≤2/prov 失败 | clip=False、覆盖=5、prov site=A | 是 |

**判定：`no-go（INCONCLUSIVE）`**。go 条件第 1 条不满足（seed 101 的 Δ_T 为负且 |−0.00245|>0.002）。
按 §5「其他情况停止为 INCONCLUSIVE」，不追加 303/404，Stage 2 不启动。

## 5. 科学结论（诚实负向，撤销 F1 子采样外推）

1. **baseline win（T vs S）不成立**：Δ_T(s0) 跨 seed 变号（−0.00245 / +0.00017），均值 −0.00114。
   纯任务状态隔离在全数据 27K、新 seed 下**无 baseline win**。
   这撤销了 F1 短训（3.9% 数据 × 5 epoch）的 +0.021 / +0.042 正向读数——正是预注册 §1.2 明确警告的
   "子采样乐观外推被撤销"，且 §2.2 中 F1 读数 best epoch 触顶，本不可外推。现被全数据新 seed 证伪。
2. **AdaTask 增量（A vs T）方向为正但未达地板**：Δ_A(s0) 2/2 正（+0.00045 / +0.00060），
   但均值 +0.00053 < 参考地板 `floor_A_s0 ≈ 0.00083`，量级不足以支撑"AdaTask 增量 win"。
   A−T macro 为正（+0.0063/+0.0025）说明 A 在 macro 端没有以牺牲整体为代价，但 s0 主终点增量太小。
3. **判定强度**：本结论为 Stage 1 筛选的 no-go，**不写正式 FAIL**（按 §5，负向结果只登记为
   INCONCLUSIVE / 负向筛选，且 Δ_T 并非两 seed 均负，不满足"负向筛选"的严格条件）。

## 6. 下一步

- Stage 2（追加 303/404）**不启动**，本实验按 no-go 收尾。
- 路线含义：在预注册 s0 场景、全数据 27K、lr=5e-4 口径下，"纯任务状态隔离 + task-axis AdaTask"
  未能复现 F1 短训的乐观信号。这与 site B 的 E12（专家利用率 collapsed/负载失衡）相互独立，
  两者共同指向：**优化器侧的任务状态隔离不是当前 MoE 收益的主因**，机制重心仍应在路由侧
  （E12 已判定负载失衡，下一步是 E12b load-balance）。
- 待办：E12b（load-balance）需 site A 实现 `b2lb` stage（涉 site A 独占文件），已在此前
  site B 结论文档中登记移交请求。

## 7. 禁止项遵守声明

- 未把 s0 换成结果更好的场景（目标场景开跑前冻结为 s0）；
- 未用 seeds 42/123 作正式验证 seed（用的是预注册新 seed 101/202）；
- 未混用 pooled 与 s0/macro Δ（本文分列、分表，端点均注明）；
- 未把负向结果写成正式 FAIL（只登记 INCONCLUSIVE / no-go）。
