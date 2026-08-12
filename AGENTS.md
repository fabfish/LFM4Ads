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
- 精度指标：用 AUC（`torcheval.metrics.BinaryAUROC`），单次 trial 即写入 `result.csv`。
- 论文口径基于 100 次 trial 取平均；探索阶段可用 3-5 seed 快速验证。
- 跨 scenario 对比仅看**相对排序**（不同 scenario 的数据分布不同，绝对 AUC 不可跨 scenario 比较）。

## 3. 项目结构约定

```
LFM4Ads/
├── main.py        # 入口，调度预训练或下游评估
├── model.py       # 模型定义（DCNv2 / FeatureUsage / ModuleUsage / ModelUsage）
├── train.py       # 训练/评估循环
├── fields.py      # 特征字段定义（user 27 + video 9）
├── dataset.py     # 数据预处理（加载 → 切分 → 预训练/下游）
├── plot.py        # 画图脚本
├── result.csv     # 所有 trial 的 AUC 输出（不入库）
├── Feature/       # 下游 Feature 级图片输出（不入库）
├── Module/        # 下游 Module 级图片输出（不入库）
├── Model/         # 下游 Model 级图片输出（不入库）
└── docs/          # 所有文档和交互页面
```

## 4. 模型与实验层次

LFM4Ads 是 DCNv2 架构（360 维，5 层 Cross Network + 1 层 DNN head），包含三级下游使用方式：

- **Feature Usage（特征级）**：冻结预训练 backbone，取出中间 Cross Representation（CR）作为特征增强，按融合方式分 gate（门控加权）和 concat（拼接）
- **Module Usage（模块级）**：加载预训练权重的前 k 层 Cross Network，其余随机初始化重新训练
- **Model Usage（模型级）**：用预训练 backbone 的输出作为 item representation（IR），与新训练的 user representation（UR）做内积

实验入口：`python main.py <device>`。默认先预训练，再跑三级下游评估，AUC 追加写入 `result.csv`。

## 5. 编码与改动约束

- 优先小范围精确编辑，避免大文件重写。
- 改动后针对性检查，先 smoke（跑几步验证前向/反向不崩）再挂完整训练。
- **自动 git commit 并 push**：每次完成一个逻辑单元（代码改动 / 实验 run / 文档更新）后，自动 `git add` 相关文件并提交（`git commit`），随后 `git push` 到 origin/main。无需每次等待用户授权。注意：push 前确认改动范围合理、lint 无错；不要 commit 大体积产物（如 `*.pt` 权重、cache 中间 json 例外按 .gitignore 处理）。
- `.codebuddy/` 为项目数据目录，勿删。
- `docs/` 下的文档和页面属于仓库资产，应纳入版本管理。
