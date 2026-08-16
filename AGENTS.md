# AGENTS.md — LFM4Ads 仓库 AI 协作规则

供任何 AI agent 在本项目（LFM4Ads / KuaiRand-1K 推荐实验仓库）中工作时遵循。

---

## 1. 文档记录规则（强制）

- 所有探索/实验记录统一放在 **`docs/`** 目录。
- 文件命名格式：**`YYYYMMDD-HHMM-主题.md`**（例：`20250808-0000-LFM4Ads-初步探索与预训练复现.md`）。
  - 日期时间用记录当下的本地时间（`date '+%Y%m%d-%H%M'`）。
  - 主题用简短中文短语，词间用 `-` 连接。
- 每篇文档建议包含：目标 / 环境 / 结果（含指标表格）/ 代码改动清单 / 踩坑与修复 / 后续 backlog。
- 结果表格务必带**测量口径说明**（seed、batch_size、lr、设备等是否一致）。

## 2. 实验与对比口径

- 跨变体对比必须**控制变量**：除被测项外，seed / batch_size / lr 完全一致。
- 默认口径：`device=cuda:0 / seed=42 / DCNv2 5层 360维 / embed_dim=10`。
- **并行规则（用户 2026-08-12 放宽）**：允许跨设备并行，但必须保证**同一配对 seed 的全部路由模式跑在同一张卡上**（同 seed 的 pooled-AUC 配对差不引入设备混杂因子）；不同 seed 组可分布到不同设备并发。矩阵 runner 通过 `seed_device_map` 固定分配，未满足该约束的并行一律禁止。
- 精度指标：用 AUC（`torcheval.metrics.BinaryAUROC`），单次 trial 即写入 `results/<实验>/` 下的结果 CSV。
- 论文口径基于 100 次 trial 取平均；探索阶段可用 3-5 seed 快速验证。
- 跨 scenario 对比仅看**相对排序**（不同 scenario 的数据分布不同，绝对 AUC 不可跨 scenario 比较）。

## 3. 项目结构约定

```
LFM4Ads/
├── model.py        # 模型定义（DCNv2 / MoE / FeatureUsage 等）
├── train.py        # 训练/评估循环
├── dataset.py      # 数据预处理（加载 → 切分）
├── fields.py       # 特征字段定义（user 27 + video 9）
├── plot.py         # 画图脚本
├── *_protocol.py   # 3 个实验协议（train.py 模块级依赖其一，勿移出根目录）
├── experiments/    # 实验入口脚本（main_*.py / run_*.py，含 main_macro_auc.py）
├── scripts/        # 工具脚本按职责分子目录（见下）
├── results/        # 所有 result CSV 按实验分组 + INDEX.md 大清单
├── cache/          # checkpoints/*.pt 权重 + archives/<实验>/ 历史产物 +
│                   # macro_auc(1K)/macro_auc_27k(27K) 证据
├── logs/           # archives/<实验>/ 历史日志 + macro_auc*/ 活跃日志
└── docs/           # 所有文档和结论文档（近期）+ docs/archive/（历史）
```

**scripts/ 布局**（2026-08-16 整理后）：

```
scripts/
├── matrix/     # 实验矩阵调度（run_*_matrix.py、run_*.sh、swg_config*.py）
├── summarize/  # 结果汇总与判定（summarize_*.py，阈值硬编码防漂移）
├── verify/     # 证据校验（verify_*.py、audit_experiment_bundle.py）
├── diagnose/   # 诊断/分析（diagnose_*、analyze_*、measure_*、weighted_auc 等）
├── gpu_keeper*.py / start|stop_gpu_keeper.sh / gpu-keeper.service  ← 常驻服务，勿动
├── build_27k_dataset.py / launch_pack.py / *_authorization.json   ← 构建与授权配置
```

> 旧文档中的 `scripts/xxx.py` 路径对应 `scripts/<类>/xxx.py`（历史文档未回溯修改）。

- **结果大清单**：`results/INDEX.md` 是「结果文件 → 实验 → 判定 → 结论」的集中索引，新增结果后须同步登记。
- **核心库不动**：`model.py/train.py/dataset.py/fields.py/plot.py` 留在根目录；experiments/ 下脚本通过顶部 `sys.path` 注入指向仓库根。
- **protocol 不动**：`task_conditioned_mixture_routing_protocol.py` 被 `train.py` 模块级 import，移出根目录会断整个训练链路。

## 4. 模型与实验层次

LFM4Ads 是 DCNv2 架构（360 维，5 层 Cross Network + 1 层 DNN head），包含三级下游使用方式：

- **Feature Usage（特征级）**：冻结预训练 backbone，取出中间 Cross Representation（CR）作为特征增强，按融合方式分 gate（门控加权）和 concat（拼接）
- **Module Usage（模块级）**：加载预训练权重的前 k 层 Cross Network，其余随机初始化重新训练
- **Model Usage（模型级）**：用预训练 backbone 的输出作为 item representation（IR），与新训练的 user representation（UR）做内积

实验入口：`python experiments/main.py <device>`。默认先预训练，再跑三级下游评估，AUC 追加写入 `results/main_pretrain/`。

## 5. 编码与改动约束

- 优先小范围精确编辑，避免大文件重写。
- 改动后针对性检查，先 smoke（跑几步验证前向/反向不崩）再挂完整训练。
- **自动 git commit 并 push**：每次完成一个逻辑单元（代码改动 / 实验 run / 文档更新）后，自动 `git add` 相关文件并提交（`git commit`），随后 `git push` 到 origin/main。无需每次等待用户授权。注意：push 前确认改动范围合理、lint 无错；不要 commit 大体积产物（如 `*.pt` 权重、cache 中间 json 例外按 .gitignore 处理）。
- `.codebuddy/` 为项目数据目录，勿删。
- `docs/` 下的文档和页面属于仓库资产，应纳入版本管理。
