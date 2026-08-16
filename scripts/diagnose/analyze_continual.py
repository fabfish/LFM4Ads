"""持续学习遗忘分析（补齐 train.compute_forgetting 未给出的标准指标）。

`train.py:compute_forgetting`（D1-D 修复后，2026-08-09）的 baseline 已改为
**pre_continual**（训练前），即 code 口径 = a[i][s] - pre[s]。docstring 符号已修正为
"负值 = 遗忘"（原 R4 符号写反问题见 D1-C）。本脚本给出三套口径：

  1. code 口径   : a[i][s] - pre[s]          （与 continual_results.json 中的 *_forgetting 字段一致，负=遗忘）
  2. pre 口径    : a[T-1][s] - pre[s]         （相对预训练水平的净漂移）
  3. 标准 BWT    : mean_{i<T-1} ( a[T-1][i] - a[i][i] )   （学完全部任务后，回看旧任务掉了多少）
     配套 LA（Learning Accuracy）: mean_i a[i][i]

来源: cache/archives/continual/continual_results.json
用法: python scripts/diagnose/analyze_continual.py
输出: stdout + cache/archives/continual/continual_analysis.json
"""

from __future__ import annotations

import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
MODELS = ["vanilla", "moe"]

data = json.load(open(os.path.join(ROOT, "cache", "continual_results.json")))
counts = None
prov = os.path.join(ROOT, "cache", "provenance_report.json")
if os.path.exists(prov):
    per_tab = json.load(open(prov))["C9"]["detail"]["per_tab"]
    counts = {int(k): v["test"] for k, v in per_tab.items() if int(k) in SCENARIOS}

report = {}
for m in MODELS:
    traj = data[f"{m}_trajectory"]
    order = [t["train_scenario"] for t in traj]
    a = [t["auc"] for t in traj]              # a[i][str(s)]
    T = len(traj)
    pre = data["pre_continual"][m]
    final = data["final_auc"][m]

    # --- 1. code 口径（重算，校验 json 字段） ---
    code_vals = [f["forgetting"] for f in data[f"{m}_forgetting"]]

    # --- 2. pre 口径 ---
    pre_delta = {s: final[str(s)] - pre[str(s)] for s in SCENARIOS}

    # --- 3. 标准 BWT / LA ---
    la = [a[i][str(order[i])] for i in range(T)]                  # 刚学完任务 i 时在任务 i 上的表现
    bwt = [a[T - 1][str(order[i])] - a[i][str(order[i])] for i in range(T - 1)]

    # 头部流量保持度
    head = [0, 1]
    head_pre = sum(pre[str(s)] * counts[s] for s in head) / sum(counts[s] for s in head) \
        if counts else None
    head_fin = sum(final[str(s)] * counts[s] for s in head) / sum(counts[s] for s in head) \
        if counts else None

    report[m] = {
        "task_order": order,
        "code_forgetting": {
            "n": len(code_vals),
            "mean": round(statistics.mean(code_vals), 6),
            "min": round(min(code_vals), 6),
            "max": round(max(code_vals), 6),
            "n_negative": sum(1 for v in code_vals if v < 0),
        },
        "pre_delta": {s: round(v, 6) for s, v in pre_delta.items()},
        "pre_delta_mean": round(statistics.mean(pre_delta.values()), 6),
        "pre_delta_weighted": round(
            sum(pre_delta[s] * counts[s] for s in SCENARIOS) / sum(counts.values()), 6)
        if counts else None,
        "learning_accuracy": {order[i]: round(la[i], 6) for i in range(T)},
        "LA_mean": round(statistics.mean(la), 6),
        "BWT_per_task": {order[i]: round(bwt[i], 6) for i in range(T - 1)},
        "BWT_mean": round(statistics.mean(bwt), 6),
        "head_weighted": {"pre": round(head_pre, 6), "final": round(head_fin, 6),
                          "delta": round(head_fin - head_pre, 6)} if counts else None,
    }

print("任务顺序:", report["vanilla"]["task_order"])
print(f"\n{'指标':<38} {'Vanilla':>11} {'MoE':>11} {'MoE 相对改善':>14}")


def line(label, key, fmt="{:+.4f}", better="higher"):
    v, mm = key(report["vanilla"]), key(report["moe"])
    if v == 0:
        imp = "n/a"
    else:
        imp = f"{(1 - mm / v) * 100:+.1f}%" if better == "closer0" \
            else f"{mm - v:+.4f}"
    print(f"{label:<36} {fmt.format(v):>11} {fmt.format(mm):>11} {imp:>14}")


line("平均 forgetting（code 口径，负=遗忘）",
     lambda r: r["code_forgetting"]["mean"], better="closer0")
line("最差单条 forgetting",
     lambda r: r["code_forgetting"]["min"], better="closer0")
line("真实遗忘条目数 / 28",
     lambda r: r["code_forgetting"]["n_negative"], fmt="{:d}")
line("pre→final 未加权均值漂移",
     lambda r: r["pre_delta_mean"], better="closer0")
line("pre→final 流量加权漂移",
     lambda r: r["pre_delta_weighted"], better="closer0")
line("LA（刚学完时的任务内 AUC）均值",
     lambda r: r["LA_mean"], fmt="{:.4f}")
line("BWT（标准反向迁移，负=遗忘）",
     lambda r: r["BWT_mean"], better="closer0")
line("头部场景 0+1 加权 pre→final",
     lambda r: r["head_weighted"]["delta"], better="closer0")

print(f"\n逐场景 pre→final 漂移")
print(f"{'S':>3} {'Vanilla':>10} {'MoE':>10}")
for s in SCENARIOS:
    print(f"{s:>3} {report['vanilla']['pre_delta'][s]:>+10.4f} "
          f"{report['moe']['pre_delta'][s]:>+10.4f}")

print(f"\n逐任务 BWT（学完全部 8 个任务后，回看该任务）")
print(f"{'trainS':>7} {'Vanilla':>10} {'MoE':>10}")
for s in report["vanilla"]["task_order"][:-1]:
    print(f"{s:>7} {report['vanilla']['BWT_per_task'][s]:>+10.4f} "
          f"{report['moe']['BWT_per_task'][s]:>+10.4f}")

with open(os.path.join(ROOT, "cache", "continual_analysis.json"), "w") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2)
print("\n已写入 cache/archives/continual/continual_analysis.json")
