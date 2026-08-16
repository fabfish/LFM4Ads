# 后续安排与授权边界

> 当前决策（2026-08-14 22:12 更新）：**capacity-MoE（cross 层）路线已正式关闭**——机制跑通但收益为负
> （容量收益≈0 + 稀疏代价≈−0.001）。
>
> **本次修订**：原先"瓶颈在 embedding（97.9% 参数）"的推断**已被零成本证据证伪**——三张 ID 表
> （83,984,250 参数）推理期全部清零，test AUC 仅掉 **0.000193**（噪声内），而 320 参数的 `upload_type`
> 单独清零掉 0.0053。因此 **embedding 加宽对照实验撤销**（前提不存在），下一阶段方向改为
> **特征信息侧**：第一步 E1 是"ID embedding 死重的重训练确认"（三臂 from-scratch），
> 见[预注册文档](./20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md)。
> 旧路线（共享残差持续学习、下游迁移）维持冻结状态不变。
> 状态见 [`DRIVERS.md`](DRIVERS.md)，已可靠结论见 [`ANALYSIS.md`](ANALYSIS.md)，
> 交接一页纸见 [`HANDOFF.md`](HANDOFF.md)。

## 1. 已冻结的旧路线

共享残差最终状态：

- G1 Function-Preserving Upcycling：`done / PASS`；
- G2 Learning-Rate Semantics：`done / PASS`；
- G3 Specialist-Only Screen：`done / INCONCLUSIVE`；
- `winning_specialist_learning_rate=null`；
- `unlock_shared_path_necessity_gate=false`。

因此 shared-path necessity、shared/router LR、完整持续学习矩阵、alignment-aware 消融和 sparse scale-out 均继续 `blocked`。不再扫描 specialist LR，不用单 seed 或跨 seed 均值重解释旧结果。

## 2. 已关闭的下游迁移路线（协议归档）

下游留出域参数高效迁移路线已 `done / G1 FAIL`，完整预注册协议（P0 实现审计 / P1 seed-42 可行性门控 /
P2 正式优势 gate / 禁止项）归档于[下游迁移驱动](archive/drivers/20260812-2303-MoE下游留出域参数高效迁移驱动.md)，
结论见 [G1 结论](archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md)。不再执行，仅保留其通用实验纪律：
不更换判定阈值凑正结果、不按 arm 重采样、不给 MoE 额外参数而不给 dense 匹配对照、不以机制诊断替代 test AUC 主终点。

## 3. 当前可执行动作

### 3.1 已关闭（不再执行）

- 下游迁移路线：`done / G1 FAIL`，12/12 seed-42 trial 完成，G1 验证门控失败，按预注册协议停止、未扩 seeds 123/456；结论见 [G1 结论](archive/conclusions/20260813-1219-MoE下游留出域迁移G1结论.md)。
- capacity-MoE（cross 层）路线：`done / 正式关闭`，容量收益≈0 + 稀疏代价≈−0.001，dense-widened 独立证伪"cross 层容量是瓶颈"。**不再调 K/lb/warmup/lr/top_k，不扩 3-seed。** 结论见 [根因定位与决定性容量实验](./20260814-2111-专家无收益根因定位与决定性容量实验.md)。
- 共享残差持续学习 / sparse scale-out / same-latency 叙事：继续 `blocked`（§1）。

### 3.2 下一阶段方向：特征信息侧（原"embedding 加容量"已撤销）

capacity 路线全部关闭 + embedding 伪瓶颈被证伪后，给出**两个硬约束 + 一个新假设**：

- 硬约束 1：**加容量前必须先证明存在容量缺口**（cross 层的教训）。
- 硬约束 2：**判断瓶颈不能用参数占比**，要用零成本诊断（曝光/OOV/推理期消融）或容量对照
  （embedding 的教训——97.9% 参数贡献 −0.000193）。
- 新假设：瓶颈是**可用输入信息量**。有效依赖集中在 `tag`/`upload_type`/`user_id`/`onehot_feat*` 等
  合计约 **3 万参数**的低基数字段；长尾 ID（`video_id` 平均 2.6 次曝光、test 74% OOV）在当前
  建模方式下不交出可泛化信号。

#### E1（第一步，已完成：`done / PASS`）— ID embedding 死重的重训练确认

- 三臂 from-scratch：`full`(36 字段, dim=360) / **`idzero`**(三张 ID 表置零+冻结，架构与 full 同构) /
  `iddrop`(真移除, dim=330)。一进程一 seed → 三臂同卡，配对由构造保证。
- **结果**：Δ_id = `idzero − full` = **{+0.000688 (s42), +0.000220 (s123)}**，2/2 seed 在噪声地板
  0.001 内 → **PASS（死重成立）**；peak valid Δ = {+0.001134, +0.000522} 方向一致。
  哨兵 **10/10 PASS**（含可训练参数差精确 = 83,984,250）。
- **工程收益**：参数 84,672,605 → **584,255（−99.31%）**、wall/epoch 10.6s → 7.1s（**1.48×**）。
- 口径：`lr=1e-3`（from-scratch 历史口径）、batch 10000、15 epoch、best=精确 argmax(valid)、
  GpuBatches；**不使用 `--freeze sparse`**（embedding 是被测对象）。
- 产物：`main_field_ablation.py`、`scripts/run_field_ablation_matrix.sh`、
  `scripts/summarize/summarize_field_ablation.py`、`cache/embedding_capacity/e1_decision.json`
  （`verdict=PASS`、`unlock_feature_information_track=true`）。
  结论见 [E1 结论](./20260814-2225-E1结论-ID-embedding死重确认.md)。实际成本 8 min。
- **边界**：**"无可测差异"≠"删掉更好"**（Δ 在地板内）；限 lr=1e-3/15 epoch/2 seed；
  `iddrop` 只作工程对照；结论限"裸 ID 查表"这一建模方式。

**新默认基线（授权）**：后续所有实验用 `iddrop`（0.58M 参数、7.1s/epoch）替代 84.7M 全模型，
并因此**恢复 3 seed 口径**（42/123/456）——成本已不再是限制。

#### E2/E3/E4（E1 PASS 已解锁，各自仍需预注册）

1. **E2 长尾 ID 的可泛化表示**（**下一个要跑的**）：ID 本身不可泛化，但其属性可以——频次分桶 +
   时序安全的统计编码 + `tag`/`upload_type` 交叉，替换裸 ID，对照 `iddrop`。判据：越过 0.001 地板。
2. **E3 特征交互侧表达**：在轻量模型上加 FM/attention 式高阶交叉，参数量匹配对照。
3. **E4 天花板定位**：33 个低基数字段能达到的 AUC 上界（更强 head / GBDT 对照），
   判断 0.78 是"模型不足"还是"信息不足"。
4. ~~若 E1 = FAIL 则做 ID 侧 `embed_dim` 下探扫描~~ —— **不适用**：E1 已 PASS，
   `unlock_id_capacity_track=false`，ID 侧容量路线不再开启。

所有新实验必须：
- 先跑零成本诊断，再决定是否开 GPU；
- 同 seed 同卡配对差 + 2–3 seed；
- 预注册哨兵与判定阈值，事后不得更换。

### 3.3 交接给后续执行者

- 一份纸版初级结论与路线图见 [`HANDOFF.md`](HANDOFF.md)。
- 已可靠结论（"已知道什么"）见 [`ANALYSIS.md`](ANALYSIS.md) §2。
- 实时证据索引见 [`DRIVERS.md`](DRIVERS.md) §2/§3。
- 三处修复 + GpuBatches 高性能数据路径已固化到代码，直接复用。
