# 后续安排与授权边界

> 当前决策：**接受共享残差 G3 `INCONCLUSIVE`，冻结持续学习路线；转向 MoE 下游留出域参数高效迁移。**
> 新路线当前为 `planned / auth=not-started`，只允许实现、审计和 smoke，不启动正式 GPU 矩阵。状态见 [`DRIVERS.md`](DRIVERS.md)，分析见 [`ANALYSIS.md`](ANALYSIS.md)。

## 1. 已冻结的旧路线

共享残差最终状态：

- G1 Function-Preserving Upcycling：`done / PASS`；
- G2 Learning-Rate Semantics：`done / PASS`；
- G3 Specialist-Only Screen：`done / INCONCLUSIVE`；
- `winning_specialist_learning_rate=null`；
- `unlock_shared_path_necessity_gate=false`。

因此 shared-path necessity、shared/router LR、完整持续学习矩阵、alignment-aware 消融和 sparse scale-out 均继续 `blocked`。不再扫描 specialist LR，不用单 seed 或跨 seed 均值重解释旧结果。

## 2. 新判断

旧 `main_moe.py` / `main_moe_v2.py` 没有完成真正的三级下游：

- 已完成的是 FeatureUsage 探索；
- `ModuleUsage(..., "Vanilla")` 不加载 backbone，只是随机初始化占位；
- 没有 true MoE ModuleUsage；
- 没有执行 ModelUsage；
- 静态 `CRs` 由 train+validation 聚合，不是严格的逐样本留出域迁移。

所以既有下游结果不足以建立或否定真正的 MoE transfer 优势。合适的优势定义应改为：**目标域完全留出、4096 总标注固定、下游可训练参数近似匹配时的参数高效迁移能力**；不声称冻结 backbone 总容量匹配。

完整预注册协议见[下游迁移驱动](archive/drivers/20260812-2303-MoE下游留出域参数高效迁移驱动.md)。

## 3. P0：实现与 G0 审计

下一逻辑单元只做以下工作：

1. 新建隔离的 downstream transfer trial/matrix runner；
2. source 固定为 `[0,1,3,4,8]`，held-out targets 固定为 `[2,5,6]`；
3. 实现逐样本 frozen-backbone forward，不复用用户静态 `CRs`；
4. 实现四 arm：`dense-head`、`dense-adapter-r2`、`moe-head`、`moe-router`；
5. 固定每 `(target,seed)` 4096 条自然分布标注行，按稳定 hash 划分 3072 fit + 1024 validation，并保存索引 hash；
6. 校验 source/target 互斥、样本守恒、冻结参数 bitwise 不变；
7. 校验 dense adapter 初始函数保持；
8. 校验 `dense-adapter-r2=4681`、`moe-router=4693`，参数差小于 1%；
9. 冻结 source/driver/runner hash，产物目录禁止覆盖；
10. 只做 CPU 或小批量 smoke，不产生正式测试结论。

顺序固定为：技术 G0 PASS → 生成含 G0 report hash 的独立 authorization → G0-final 复核 auth/hash/设备映射。任一步失败，状态转为 `blocked`，不得启动预训练。

## 4. P1：seed 42 feasibility gate

G0 PASS 且独立 authorization 完成后，才运行 seed 42：

- source models：same-FLOPs dense 与 `DCNv2MoE_LowRank(K=4,r=45)`；
- target：2/5/6；
- 每 target 四个 arm；
- 同 seed 全部配对模型固定同卡；
- test 不参与早停或筛选。

只有三个 target 的 validation 同时满足：

1. `moe-router - dense-adapter-r2 > 0`；
2. `moe-router - moe-head > 0`；
3. 无样本、冻结、设备或 hash 异常；

才允许继续 seeds 123/456。失败立即停止，不扫描 LR/K/r/target。

## 5. P2：正式优势 gate

主比较：同 seed、同 target 的

`Δ_primary = AUC(moe-router) - AUC(dense-adapter-r2)`。

同时计算：

`Δ_adaptation = (moe-router-moe-head) - (dense-adapter-r2-dense-head)`。

判定：

- **PASS**：9/9 `Δ_primary >0`；每 target 的 mean `Δ_primary >=0.001`；macro mean `Δ_primary >=0.0015`；9-pair mean `Δ_adaptation >=0.0005`，且每 target 的 mean `Δ_adaptation >0`。
- **FAIL**：9/9 `Δ_primary <=0`，或三个 target 的 mean `Δ_primary` 全部 `<=0`。
- **INCONCLUSIVE**：其余方向混合、未过实际显著性阈值或适配双重差分归因不成立。
- **BLOCKED**：缺 trial、失败、泄漏、设备混杂、参数/冻结/hash/样本审计异常。

PASS 也只能支持“该固定协议、固定测试集上跨训练 seed 方向一致的留出域参数高效迁移优势”，不能声称总体统计显著，也不能恢复静态 AUC、持续学习、same-latency 或 sparse scale-out 主张。

## 6. 禁止项

- 禁止继续把旧 112 对称为三级下游证据；准确口径是 104 个 FeatureUsage 对照 + 8 个随机初始化占位。
- 禁止为获得正结果更换 target、标签预算、K、rank、LR 或判定阈值。
- 禁止使用任何 target 行训练 source backbone；仓库原 target validation 也不得用于下游早停或 arm 选择。
- 禁止按 arm 重采样 4096 条下游标注数据或改变 3072/1024 划分。
- 禁止让 MoE 获得额外 projection、target embedding 或专家，而 dense 无参数匹配对照。
- 禁止用 router entropy、MI、门控可视化替代 test AUC 主终点。
- 禁止用 macro 均值掩盖某 target 的系统性退化。
- 禁止从当前协议外推到其他数据集或生产推理效率。

## 7. 当前可执行动作

无需再扩展旧 G3。当前仅执行新下游路线的 runner、G0 verifier、manifest/auth 模板和 smoke；在 G0 PASS 前，实验状态保持 `planned / auth=not-started`，不运行正式 GPU 任务。
