# Site B 复工预注册：b0repro + b1topk + E12 诊断（E14-B）

- 创建时间：2026-08-18 22:10
- 状态：**`launched=running`**（2026-08-18 22:25 site B 实机启动；判定规则仍以本文 §2 冻结版为准）
- 启动记录（运维事实，不改变任何判定规则）：
  - preflight 六项全绿（`cache/preflight_site_B.json`：C1 size/C2 vocab/C3 counts/C4 scenarios/C5 numerics/C4b params）。
    首次启动因 site B 环境缺 `torcheval` 而 C4/C5 FAIL（启动器按设计中止、未派发任何 run），
    补装 `torcheval==0.0.7`（torch 2.7.1 / cuda 12.8 / 3×H20）后二次启动通过。
  - `--budget-hours` 由启动器默认 8 提为 **12**：cuda:0 串行 7 个 run（seed 42/789 全臂同卡），
    8h 守卫可能截断 b1topk 尾部 run、破坏 4-seed 配对完整性；12h 为纯墙钟守卫，
    不改变任何科学参数（epochs/patience/lr/bs 仍为 E10 协议 20/10/1e-3/10000）。
  - 启动命令：`setsid nohup bash scripts/matrix/run_siteB.sh b0repro,b1topk --budget-hours 12`，
    驱动日志 `logs/macro_auc_27k_siteB/matrix_b0b1.log`。
- 上游预注册：[开放实验计划 E12/E13/E14](./20260817-1215-开放实验计划-下一个关键MoE改动.md)、
  [两端协作分离设计与实验分工](./20260817-1400-两端协作分离设计与实验分工.md)
- 与 site A 正在跑的 E17（[预注册](./20260818-1815-结论复审与E17单场景AdaTask-win-case预注册.md)）
  **完全正交**：不同实验、不同 namespace、互不抢占设备。
- 本文回答："site B 三卡现在跑什么" —— 冻结为 **b0repro → E12 诊断 → b1topk**，
  并按 E12 判定决定是否追加 b2（E13 或 E12b）。

---

## 0. 术语与口径（继承，不重复定义）

macro AUC / 噪声地板 / 配对差 / 判定枚举定义见
[开放实验计划 §口径基线](./20260817-1215-开放实验计划-下一个关键MoE改动.md) 与
[E8910 结论](./20260816-1950-E8910结论-硬路由topk2最终形态.md)。
本文新增：

- **site 内自足**：每个判定所需的全部臂（含 dense 对照）都在 site B 重跑，
  噪声地板由 site B 自己的 dense 臂估计；**禁止**拿 site B 的 run 减 site A 的 run。
- **E14-B**：E14 的 site B 版（stage 名 `b1topk`）。Δ 定义同 E14，但 dense 对照与
  地板全部来自 site B。
- **可比性校验（sanity，非判定）**：b0repro dense 4 seed 的 macro 绝对值与
  site A s1 dense 的读数只做**趋势对照**（方向、量级），不产生任何配对差结论。

## 1. 任务与顺序（冻结）

| 步骤 | 内容 | run 数 | 成本 | 产出 |
|---|---|---|---|---|
| P0 | `python scripts/verify/preflight_site.py --site B`（机器重起后必须全绿） | 0 | ~10 min | preflight json |
| P1 | stage `b0repro`：dense×{42,123,456,789} + moe_tk2_s42 | 5 | ~5 GPU·h | site B 基线 + E12 数据源 |
| P2 | **E12 诊断**（P1 的 moe run 落盘后即可，CPU 秒级）：`scripts/diagnose/diagnose_expert_usage.py` 对 b0repro_moe_tk2_s42（若 b1 的 tk2 已出可一并） | 0 | ~1 min | `expert_usage.json` + 分化/坍缩判定 |
| P3 | stage `b1topk`：moe_tk2×{42,123,456,789} + moe_tk1×{42,123,456,789} | 8 | ~8 GPU·h | E14-B 数据 |
| P4 | 按 P2 判定追加：**differentiated → b2router（E13，三臂 A/B/C）**；**collapsed → b2lb（E12b）**；partial → b2router 优先。追加前必须先更新本文档再启动（决策树照 1215 文档 §优先级） | 8–12 | ~8–12 GPU·h | — |

seed→卡映射（`run_macro_auc_matrix.py` 的 `_SEED_DEVICE_BY_SITE["B"]`，同 seed 全臂同卡）：
42/789→cuda:0，123→cuda:1，456→cuda:2。

环境（强制）：
```
LFM_SITE=B
LFM_MACRO_OUT=cache/macro_auc_27k_siteB      # 日志目录自动跟随
python scripts/matrix/run_macro_auc_matrix.py --stages b0repro,b1topk
```

产物：`cache/macro_auc_27k_siteB/run_b0repro_*.json`、`run_b1_moe_tk*_s*.json`。
所有 run json 的 `provenance.site=B`；run json 含 `router_weights`（main_macro_auc.py
已在 best-state 后 dump，E12 数据源由此而来）。

## 2. 判定规则（冻结，启动前不可改）

### 2.1 E14-B：top_k 单调性（主判定）

- 配对差：`Δ_1(s) = macro_s(tk1) − macro_s(dense_b0)`，`Δ_2(s) = macro_s(tk2) − macro_s(dense_b0)`，
  其中 tk2/dense 均取 site B 自己的 run（tk2 用 `b1_moe_tk2_s*`；dense 用 `b0repro_dense_s*`）。
- 地板：`floor_B = 2 × sample_SD(b0repro dense 4 seed 的 test macro)`（site B 自算，登记进判定 json；
  与 site A 的 0.00135 只做趋势对照，不得混用）。
- 判定（照 E14 预注册表）：
  - `Δ_1 > Δ_2` 且 Δ_1 4/4 正、mean > floor_B → 单调"越特化越好"；
  - `Δ_1 < Δ_2` 且 Δ_1 mean > floor_B 且 4/4 正 → top_k=2 为最优协作度；
  - `Δ_1` mean ≤ floor_B → 过度特化有害（协作必要）；
  - 变号 → INCONCLUSIVE，允许按原口径扩 seed（101/202）。
- 副产 sanity：`Δ_2` 应与 site A E10 的 +0.0050 同向同量级（趋势对照，不判 PASS/FAIL，
  显著背离时只登记"site 间复现分歧"，不得改判 E10）。

### 2.2 E12 诊断判据（照 1215 文档冻结表，脚本已实现）

| 观测（跨层最保守聚合） | 判定 | 动作 |
|---|---|---|
| coverage=5 且 load_ratio≤2 | differentiated | P4 上 b2router（E13） |
| coverage≤3 或 load_ratio>3 | collapsed | P4 上 b2lb（E12b） |
| 其余 | partial | b2router 优先，E12b 补做 |

多 run 聚合：verdict 全一致 → 该 verdict；任一 collapsed → `mixed_has_collapsed`（按 collapsed 处理）。

### 2.3 禁止项

- 禁止跨 site 配对差、跨 site 合并地板/均值；
- 禁止因 E12 结果改判 E10/E11；
- 禁止 b1/b2 之外的临时配置；追加任何配置先更新本文档；
- E14-B 结论只在 site B namespace 内表述，如与 site A E10 趋势背离，
  只登记分歧并通知 site A 复核，不得单方面改写既有 PASS。

## 3. 预算与时间线

- P0–P3 总计 13 run ≈ 13 GPU·h；三卡（5/3/3 分配）墙钟 ≈ **5–6 h**。
- P2 在 P1 的 moe run 完成后即时插入（不阻塞 b1topk 的 dense→moe 顺序——
  矩阵按 stage 顺序调度，b0repro 全部完成后才进 b1topk，故 P2 在两 stage 间隙执行）。
- P4 视 P2 判定 + 剩余预算启动；若 E17 Stage 1（site A）出 gate 需要扩 303/404，
  **site A 卡自足**，不与 site B 抢卡。

## 4. 交接说明

- 本文与矩阵 stage 定义（`b0repro`/`b1topk`）由 site A 写入 main；site B 直接
  `git pull origin main` 后运行，产物只写自己的 namespace。
- 判定汇总脚本如需新增（`scripts/summarize/summarize_b_stages.py`），由 site A
  在 b1topk 数据就绪后实现（避免 site B 改 site A 独占文件）。
- E12 诊断脚本 `scripts/diagnose/diagnose_expert_usage.py` 已在 main，零 GPU 成本。
