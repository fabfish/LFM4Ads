# AGENTS.md — LFM4Ads 仓库 AI 协作规则

供任何 AI agent 在本项目（LFM4Ads / KuaiRand-1K 推荐实验仓库）中工作时遵循。

> **若你在新接入的机器（site B）上工作**：先读
> [`docs/20260817-1440-交接协作文档-siteB接手指南.md`](./docs/20260817-1440-交接协作文档-siteB接手指南.md)，
> 它是自包含的接手指南（术语 / 环境 / 纪律 / 任务队列 / 回报模板）。本文件是仓库总规则，两者一起生效。

---

## 1. 文档记录规则（强制）

- 所有探索/实验记录统一放在 **`docs/`** 目录。
- 文件命名格式：**`YYYYMMDD-HHMM-主题.md`**（例：`20250808-0000-LFM4Ads-初步探索与预训练复现.md`）。
  - 日期时间用记录当下的本地时间（`date '+%Y%m%d-%H%M'`）。
  - 主题用简短中文短语，词间用 `-` 连接。
- 每篇文档建议包含：目标 / 环境 / 结果（含指标表格）/ 代码改动清单 / 踩坑与修复 / 后续 backlog。
- 结果表格务必带**测量口径说明**（seed、batch_size、lr、设备等是否一致）。

### 1.1 文档自包含与科学严谨规范（强制，用户 2026-08-17 要求）

**判定标准：把文档交给一个没参与过任何实验的读者，他必须能仅凭本文档 + 文中链接读懂全部结论。任何术语、数值、因果断言无法在文内溯源，该文档即视为不合格。**

1. **术语必须先定义再使用**。每篇含结论的文档必须有 **`§0 术语与度量定义`** 章节，用**公式或可执行口径**（而非自然语言比喻）定义文中出现的每个非通用概念。本项目必须定义的高频术语至少包括：
   `pooled AUC`、`macro AUC`、`端点(endpoint)`、`噪声地板(noise floor)`、`配对差(paired delta)`、
   `软路由/硬路由 top_k`、`参数守恒`、`哨兵(sentinel)`、`判定枚举(PASS/INCONCLUSIVE/FAIL)`。
   已在别处定义过的，给出**指向该文档具体章节的链接**，不得只写术语名。
   - **抽象术语必须配「在代码/流程的哪一步起作用」的说明 + 一个真实数据的例子**。例如「端点」不能只写"用于判定的指标"，而要指出它作用于两个具体决策点（① 训练中按 valid 指标选 epoch / early-stop，② 训练后按 test 指标判 PASS），并用落盘 run 的 epoch 曲线举例说明换端点会选出不同权重、后果是多少 AUC。
2. **禁止用比喻代替定量**。像"白拿"、"一大半"、"掩盖了机制"这类说法，**必须**同时给出：定义式 → 代入的数值 → 结果。若确实只有量级估算，写明"估算"、列出假设、标注误差方向。
3. **每个因果断言必须附四元组**：`唯一变量` + `对照的两组 run` + `证据文件路径` + `事先冻结的判定规则`。缺任一项则该断言只能写为"观察到的相关性"，不得写成"因为…所以…"。
4. **数值必须带来源标签**：数据集（1K / 27K）、是**实测**还是**估算/外推**、seed 数、是否配对。跨数据集的数值不得放进同一张表内直接比较。
5. **端点不混表**：macro 与 pooled 是两个不同端点，必须分表或分列并在表头写明端点名；不得把两者的 Δ 放在同一列。
   - **端点对称性必须显式声明**：任何 Δ（配对差）表格须注明"**两臂使用同一端点 + 同一模型选择准则（选 epoch / early-stop）**"，并可指向 run json 的 `provenance.model_selection` 字段核对。**严禁**出现 baseline 按一个端点、实验臂按另一个端点的比较。
   - **端点必须在开跑前的预注册文档中冻结**，禁止看到结果后改端点（post-hoc endpoint selection）；结论文档须回链该预注册文档的冻结章节。
   - 若某结论只在某一端点成立，必须**同时给出另一端点的读数**并标注该读数的口径限制（例如"epoch 仍按主端点选出，故非独立的第二端点实验"）。
6. **结论强度只用登记的判定枚举**（`PASS` / `INCONCLUSIVE` / `FAIL`）+ 地板倍数 + 同号 seed 数。禁止使用"显著提升""大幅改善""效果明显"等无定义的强度词。
7. **必要性/充分性声明必须带适用范围**：写"X 是必要条件"时须限定效应量级或配置范围（例：对 +0.002 量级效应必要，对 +0.005 量级非必要），不得无条件外推。
8. **反直觉结论必须给出机制解释 + 该机制的可检验预测**（下一个实验如何能证伪它）。

## 2. 实验与对比口径

- 跨变体对比必须**控制变量**：除被测项外，seed / batch_size / lr 完全一致。
- 默认口径：`device=cuda:0 / seed=42 / DCNv2 5层 360维 / embed_dim=10`。
- **并行规则（用户 2026-08-12 放宽）**：允许跨设备并行，但必须保证**同一配对 seed 的全部路由模式跑在同一张卡上**（同 seed 的 pooled-AUC 配对差不引入设备混杂因子）；不同 seed 组可分布到不同设备并发。矩阵 runner 通过 `seed_device_map` 固定分配，未满足该约束的并行一律禁止。
- 精度指标：用 AUC（`torcheval.metrics.BinaryAUROC`），单次 trial 即写入 `results/<实验>/` 下的结果 CSV。
- 论文口径基于 100 次 trial 取平均；探索阶段可用 3-5 seed 快速验证。
- 跨 scenario 对比仅看**相对排序**（不同 scenario 的数据分布不同，绝对 AUC 不可跨 scenario 比较）。

### 2.1 两端协作隔离（2026-08-17 起，强制）

本仓库由两个 site 并行贡献实验（site A = 2 卡机，site B = 3 卡机）。详见
[两端协作分离设计与实验分工](./docs/20260817-1400-两端协作分离设计与实验分工.md)。硬约束：

- **配对差绝不跨 site**：禁止拿 site B 的 moe 减 site A 的 dense；禁止跨 site 共用噪声地板；禁止跨 site 合并均值。分配粒度是**整个实验**（含其 dense 对照臂），每个 site 的每个实验必须 **site 内自足**。
- **产物命名空间隔离**：site B 一律 `LFM_MACRO_OUT=cache/macro_auc_27k_siteB`（日志目录自动跟随），run tag 带 site 前缀；每个 run json 必须有 `provenance.site`（由 `LFM_SITE` 注入）。
- **跑任何矩阵前必须 preflight 全绿**：`python scripts/verify/preflight_site.py --site <X>`（校验数据字节、切分样本量、场景集合、参数量、数值环境）。
- **seed→device 映射按 site 选择**（`run_macro_auc_matrix.py` 的 `_SEED_DEVICE_BY_SITE`）；无论几张卡，**同一 seed 的全部臂必须同卡**。
- **文件归属独占**：`model.py` / `experiments/**` / `scripts/matrix/run_macro_auc_matrix.py` / `results/INDEX.md` / `docs/NEXT.md` 由 site A 独占修改；site B 只写自己 namespace 下的产物与带 `-siteB` 后缀的新文档。site B 在 `site-b/*` 分支工作，由 A 合并。

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
- **主动 git commit 并 push（强制，用户 2026-08-17 重申）**：agent **每次**完成一个逻辑单元（代码改动 / 实验 run / 文档更新 / 任一单次交互产出）后，**必须主动** `git add` 相关文件 → `git commit` → `git push origin main`，**不等用户提醒、不等用户授权**。这是一个动作闭环，不是一个可选项。
  - 提交粒度：一个逻辑单元一次 commit，不要攒多个无关改动混在一起。
  - push 前确认改动范围合理、`read_lints` 无新增错误。
  - 不要 commit 大体积产物（`*.pt` 权重、`dataset*.feather`、`cache` 中间产物）——它们按 `.gitignore` 排除；`docs/`、`results/INDEX.md`、代码、小 json（如 `fields_27k.json` / `sample_counts_27k.json` / 判定 json）属于仓库资产，**应纳入版本管理**。
  - 注意：仓库目录与数据目录在**同一套 CephFS 挂载**上（见 §6），`dataset_27k.feather` 虽在仓库目录内但被 `.gitignore` 排除，**不得** `git add -f` 它。
- `.codebuddy/` 为项目数据目录，勿删。
- `docs/` 下的文档和页面属于仓库资产，应纳入版本管理。

## 6. 运行环境与磁盘挂载（2026-08-17 起，两端共用）

两端（site A / site B）**在同一套 CephFS 挂载**上工作，数据文件共享、无需各自重建。以下为 site A 实测的挂载拓扑：

| 挂载点 | 文件系统 | 容量 | 已用 | 可用 | 用途 |
|---|---|---|---|---|---|
| `/root/Documents/Lfm4ads`（git 仓库） | `ceph-fuse`（CephFS） | 2.0T | 454G | 1.6T（23%） | 仓库 + `dataset_27k.feather` + `cache/`（80G 总占用） |
| `/apdcephfs/private-xavieryu` | `ceph-fuse`（CephFS，**同上一个池**） | 2.0T | 454G | 1.6T（23%） | KuaiRand-27K 原始 CSV（`.../database/KuaiRand-27K/data/`） |
| `/dockerdata` | `xfs`（本地盘） | 12T | 7.7T | 4.0T（66%） | 本地临时盘 |
| `/apdcephfs_fsgm/share_303710656` | `dop-fuse` | 271T | 249T | 23T（92%） | 共享大池 |
| `/jizhicfs` | `dop-fuse` | 325T | 270T | 56T（83%） | 共享大池 |

**要点**：
- 仓库目录与数据目录在**同一个 CephFS 池**（`df` 显示同为 2.0T / 454G / 23%），因此两端挂载同一路径即可**直接共享**：原始 CSV、已构建的 `dataset_27k.feather`（在仓库根目录，6.0GB）、`cache/` 产物。
- 但这**不改变两端隔离纪律**（§2.1）：`LFM_MACRO_OUT` 仍须按 site 分目录，配对差仍禁止跨 site 相减；共享挂载只是让"数据自建"这一步变得可选。
- git remote：`git@github.com:fabfish/LFM4Ads.git`（SSH）。`dataset_27k.feather`（6.0GB）与 `cache/`（68G）**不入 git**，靠共享挂载在两端间可见。
- 空间余量：CephFS 剩 1.6T，足以容纳多轮实验产物；若接近满（可用 < 200G）须先清理 `cache/archives` 或旧日志再跑新矩阵。
