# 后续安排与授权边界

> 当前默认决策：**接受 G3 `INCONCLUSIVE`，冻结本轮路线，不启动新实验。**
> 状态证据见 [`DRIVERS.md`](DRIVERS.md)，研究解释见 [`ANALYSIS.md`](ANALYSIS.md)。

## 1. 当前停止点

共享残差路线的最终状态为：

- G1 Function-Preserving Upcycling：`done / PASS`；
- G2 Learning-Rate Semantics：`done / PASS`；
- G3 Specialist-Only Screen：`done / INCONCLUSIVE`；
- `winning_specialist_learning_rate=null`；
- `unlock_shared_path_necessity_gate=false`。

这意味着当前没有可沿用的实验授权。shared LR、router LR、完整持续学习矩阵、alignment-aware 消融和 sparse scale-out 全部继续 `blocked`。

## 2. 优先级

### P0：固化当前结论

1. 以三个根级入口作为单一事实源，历史材料只承担证据角色。
2. 对外叙事聚焦三个可复核结果：训练目标错配修复、静态 MoE 竞争性失败、持续学习 screen 未决。
3. 保留负结果与 provenance，不把 `INCONCLUSIVE` 改写成 `FAIL`，也不把小均值差改写成收益。

### P1：默认终止当前路线

若没有新的明确授权，接受当前终点：

- 不再扫描已有 3 档 specialist LR；
- 不扩大 MoE K/rank/路由配置；
- 不测试 shared/router LR；
- 不启动真实稀疏扩展；
- 不以单 seed 或跨 seed 均值重新解释现有结果。

### P2：可选的 G3 证据增强关卡

只有用户明确决定继续时，才另立新驱动和新 authorization。该关卡的唯一允许变化是**扩充预注册 seeds**；其余协议保持不变：

- 模型、dense checkpoints 来源和 upcycling 方式不变；
- tasks 与两个 order 不变；
- 每 scenario 24 步、2 epochs、early stopping=false 不变；
- task-head LR=`5e-4` 不变；
- specialist LR=`2e-4/5e-4/1e-3` 不变；
- batch size、AdamW、weight decay、路由与冻结语义不变；
- 同 seed 全部 arms/orders 固定同卡；
- 使用独立不可覆盖目录，不覆盖当前 24 trial 或机器判定。

新驱动必须在启动前冻结：seed 列表、样本数、设备映射、trial 总数、source/driver/auth hash、主终点、噪声阈值、PASS/FAIL/INCONCLUSIVE/BLOCKED 规则。

## 3. 证据增强判定规则

沿用现有 G3 主终点和方向规则，不事后改阈值：

1. 对每个 specialist LR，相对同 seed、同 order 的 head-only baseline 计算 BWT paired difference。
2. 两个 order 各自的所有预注册 seed 必须全部 `>0`。
3. 每个 order 的 mean BWT improvement 必须超过该 order head-only baseline 的预注册噪声阈值。
4. 任一 order 的 Learning Accuracy paired differences 不得全部 `<0`。
5. 至少一个 LR 同时满足两个 order，才判 `PASS`。
6. 若所有 BWT paired differences 均 `≤0`，判 `FAIL`。
7. 其余方向混合或未越过阈值，判 `INCONCLUSIVE`。
8. 缺 trial、失败、设备混杂、hash 或公平性异常，判 `BLOCKED`。

## 4. 明确停止条件

任一条件成立即停止 specialist-only 路线，不再追加 LR 或结构扫描：

- 扩充 seeds 后仍出现跨方向变号；
- 没有任何 LR 同时通过两个 order；
- mean improvement 仍未越过预注册噪声阈值；
- 结果为 `FAIL`、`INCONCLUSIVE` 或 `BLOCKED`；
- 为获得 PASS 需要改变步数、LR 集合、任务顺序或判定阈值。

若怀疑“24 步预算过小”，那是不同研究问题，必须另立预算敏感性驱动；不能与 seed 扩充混在同一关卡，也不能据此解锁 shared path。

## 5. PASS 后仍需单独授权的关卡

即使证据增强 G3 明确 `PASS`，也只允许**另立** Shared-Path Necessity Gate，不自动运行。该新关卡至少需要：

1. specialist-only winner 作为固定锚点；
2. shared path 的唯一被测变量、独立 parameter-group LR 和冻结对照；
3. 同 seed、同 order、同设备配对；
4. 明确 BWT/LA/forgetting 主次终点；
5. 预注册停止条件与不可覆盖 provenance。

Shared-Path Gate 通过后，blockwise LR、router LR、完整 continual matrix 仍需逐级另立授权。Sparse scale-out 还受 Stage B 竞争性 `FAIL` 约束，不能由持续学习局部结果自动恢复。

## 6. 禁止项

- 禁止直接重跑或覆盖现有 G3 汇总、manifest、decision JSON。
- 禁止把旧 manifest 中的原始驱动路径/hash 改写为新归档路径。
- 禁止将 `INCONCLUSIVE` 表述为“证明无效”或“接近 PASS”。
- 禁止在更多 seed 与更多训练步数之间事后择优。
- 禁止用不同设备承载同一 seed 的配对 arms。
- 禁止以路由结构指标替代 pooled-AUC/BWT 主终点。
- 禁止从当前证据外推到其他数据集、所有 MoE 或生产推理效率。

## 7. 下一次决策所需输入

若要继续，只需先作一个显式选择：

- **结束**：接受当前终点，转入论文/报告整理；或
- **提口径**：授权仅扩 seed 的 G3 证据增强关卡，并在新驱动中冻结完整协议。

在该选择出现前，实验状态保持 `blocked`，不运行 GPU 任务。
