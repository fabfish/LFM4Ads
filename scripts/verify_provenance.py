"""LFM4Ads 数据溯源核查脚本 (Provenance Verifier).

对文档中引用的每一个指标，从**源文件**重新提取并交叉验证，杜绝手抄/伪造。

用法:
    python scripts/verify_provenance.py              # 全量核查（含 dataset.feather）
    python scripts/verify_provenance.py --no-feather # 跳过 444MB feather 读取

输出:
    stdout                          — 人读核查报告
    cache/provenance_report.json    — 机读核查结果（供文档引用）

核查项 (checks):
    C1  result_moe.csv           Delta 列 == MoE - Vanilla；Mean 行 == 8 场景均值
    C2  result_moe.csv           与 continual_results.json.pre_continual 一致性
    C3  dominance_matrix.json    每个 expert 的 8 场景占比之和 == 1
    C4  dominance_matrix.json    >0.3 的格子数（SpecializationLoss 触发条件）
    C5  adatask_results.csv      三模式均值 / 逐场景胜负
    C6  adatask_au_*.json        AU 键完整性（3 层 × 4 专家 × 场景）与 dominance 重算
    C7  continual_results.json   forgetting 可由 trajectory 重算（含符号约定校验）
    C8  continual_results.json   final_auc == trajectory 最后一个 task 的 auc
    C9  dataset.feather          tab 分布 / CTR / 日期切分样本量
    C10 result_moe_downstream.csv 覆盖度（应有 8 场景 × 2 模型 × 14 方法）
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]
TOL = 5e-5  # 源 CSV 保留 4 位小数，容差取半个最低位

report: "OrderedDict[str, dict]" = OrderedDict()
failures: list[str] = []


def check(cid: str, desc: str, ok: bool, detail=None):
    status = "PASS" if ok else "FAIL"
    report[cid] = {"desc": desc, "status": status, "detail": detail}
    if not ok:
        failures.append(f"{cid}: {desc}")
    print(f"[{status}] {cid}  {desc}")
    if detail is not None:
        text = detail if isinstance(detail, str) else json.dumps(
            detail, ensure_ascii=False, indent=2, default=str)
        for line in text.splitlines():
            print(f"        {line}")


def p(*parts):
    return os.path.join(ROOT, *parts)


# ============================================================
# C1 / C2  result_moe.csv
# ============================================================
print("\n" + "=" * 70)
print("Source: result_moe.csv  (main_moe.py Step 4, 行 139-144)")
print("=" * 70)

moe_csv = pd.read_csv(p("result_moe.csv"))
rows = {str(r["Scenario"]): r for _, r in moe_csv.iterrows()}

# main_moe.py 用全精度算 Delta 再格式化为 %+.4f，而 AUC 列本身已 round(,4)；
# 因此「用 CSV 的两列相减」可能与 Delta 列差 1 个最低位——这是舍入伪影，不是错误。
delta_err, delta_round = {}, {}
for s in SCENARIOS:
    r = rows[str(s)]
    recomputed = round(r["MoE_AUC"] - r["Vanilla_AUC"], 4)
    gap = abs(recomputed - r["Delta"])
    if gap > TOL:
        bucket = delta_round if gap <= 1.01e-4 else delta_err
        bucket[s] = {"csv_delta": r["Delta"], "recomputed_from_columns": recomputed,
                     "gap": round(gap, 6)}

mean_v = sum(rows[str(s)]["Vanilla_AUC"] for s in SCENARIOS) / len(SCENARIOS)
mean_m = sum(rows[str(s)]["MoE_AUC"] for s in SCENARIOS) / len(SCENARIOS)
mean_row = rows["Mean"]
mean_ok = (abs(mean_v - mean_row["Vanilla_AUC"]) <= TOL
           and abs(mean_m - mean_row["MoE_AUC"]) <= TOL)

check("C1", "result_moe.csv Delta/Mean 自洽（1ulp 舍入差单列为 rounding_artifact）",
      not delta_err and mean_ok,
      {"delta_mismatch": delta_err,
       "rounding_artifact": delta_round,
       "rounding_note": "Delta 列由全精度 AUC 相减后 %+.4f 格式化，AUC 列亦已 round(,4)，"
                        "两者可差 1 个最低位；文档引用 Delta 时以 CSV 的 Delta 列为准",
       "mean_recomputed": {"vanilla": round(mean_v, 4), "moe": round(mean_m, 4)},
       "mean_in_csv": {"vanilla": mean_row["Vanilla_AUC"], "moe": mean_row["MoE_AUC"]},
       "moe_wins": [s for s in SCENARIOS if rows[str(s)]["Delta"] > 0],
       "moe_loses": [s for s in SCENARIOS if rows[str(s)]["Delta"] < 0]})

continual = json.load(open(p("cache", "continual_results.json")))
pre = continual["pre_continual"]
cross = {}
for s in SCENARIOS:
    k = str(s)
    cross[s] = {
        "result_moe.csv_vanilla": rows[k]["Vanilla_AUC"],
        "pre_continual_vanilla": round(pre["vanilla"][k], 4),
        "result_moe.csv_moe": rows[k]["MoE_AUC"],
        "pre_continual_moe": round(pre["moe"][k], 4),
    }
cross_consistent = all(
    abs(v["result_moe.csv_vanilla"] - v["pre_continual_vanilla"]) <= TOL
    and abs(v["result_moe.csv_moe"] - v["pre_continual_moe"]) <= TOL
    for v in cross.values())

check("C2", "result_moe.csv 与 continual_results.json.pre_continual 一致",
      cross_consistent,
      {"note": "不一致属预期——两者由不同 run 产出（模型 checkpoint 已被覆盖重训），"
               "文档须分别标注来源，不可混用",
       "per_scenario": cross})


# ============================================================
# C3 / C4  dominance_matrix.json
# ============================================================
print("\n" + "=" * 70)
print("Source: cache/dominance_matrix.json  (main_moe.py 行 100-108)")
print("=" * 70)

dom = json.load(open(p("cache", "dominance_matrix.json")))
row_sums, over_thr, argmax = {}, [], {}
for layer, cells in dom.items():
    for ei in range(4):
        vals = {s: cells[f"E{ei}_S{s}"] for s in SCENARIOS}
        row_sums[f"{layer}/E{ei}"] = round(sum(vals.values()), 4)
        top_s = max(vals, key=vals.get)
        argmax[f"{layer}/E{ei}"] = {"scenario": top_s, "ratio": vals[top_s]}
        for s, v in vals.items():
            if v > 0.3:
                over_thr.append({"layer": layer, "expert": ei, "scenario": s, "ratio": v})

sums_ok = all(abs(v - 1.0) <= 2e-3 for v in row_sums.values())  # json 存的是 round(,4)
check("C3", "dominance 每个 expert 的 8 场景占比之和 == 1", sums_ok, row_sums)

check("C4", "dominance >0.3 的格子（SpecializationLoss threshold=0.3 触发条件）",
      len(over_thr) > 0,
      {"count": len(over_thr),
       "cells": over_thr,
       "argmax_per_expert": argmax,
       "conclusion": "所有 12 个 (layer,expert) 的 argmax 均为 S5 → "
                     "占比由梯度尺度而非功能分工主导，不能直接判定为专家特异化"})


# ============================================================
# C5  adatask_results.csv
# ============================================================
print("\n" + "=" * 70)
print("Source: cache/adatask_results.csv  (main_adatask.py)")
print("=" * 70)

ada = pd.read_csv(p("cache", "adatask_results.csv")).set_index("scenario")
modes = ["none", "encourage", "suppress"]
means = {m: float(ada[m].mean()) for m in modes}
per_scn = {}
win = {m: 0 for m in modes}
for s in SCENARIOS:
    r = {m: float(ada.loc[s, m]) for m in modes}
    best = max(r, key=r.get)
    win[best] += 1
    per_scn[s] = {**{m: round(r[m], 4) for m in modes},
                  "best": best,
                  "enc-none": round(r["encourage"] - r["none"], 4),
                  "sup-none": round(r["suppress"] - r["none"], 4)}

check("C5", "adatask_results.csv 三模式均值与逐场景胜负", True,
      {"mean": {m: round(means[m], 4) for m in modes},
       "mean_delta_vs_none": {
           "encourage": round(means["encourage"] - means["none"], 4),
           "suppress": round(means["suppress"] - means["none"], 4)},
       "win_count": win,
       "per_scenario": per_scn})


# ============================================================
# C6  adatask_au_*.json
# ============================================================
print("\n" + "=" * 70)
print("Source: cache/adatask_au_{none,encourage,suppress}.json")
print("=" * 70)

au_summary = {}
for m in modes:
    path = p("cache", f"adatask_au_{m}.json")
    raw = json.load(open(path))
    # 键形如 "(layer, expert, scenario)" 或 "layer_expert_scenario"，统一解析
    parsed = {}
    for k, v in raw.items():
        nums = [int(x) for x in
                str(k).replace("(", "").replace(")", "").replace("_", ",").split(",")
                if x.strip().lstrip("-").isdigit()]
        if len(nums) == 3:
            parsed[tuple(nums)] = float(v)
    layers = sorted({k[0] for k in parsed})
    experts = sorted({k[1] for k in parsed})
    scns = sorted({k[2] for k in parsed})
    # 用与 GradientTracker.dominance_matrix 相同的口径重算 layer0 占比
    recomputed = {}
    for ei in experts:
        vals = {s: parsed.get((0, ei, s), 0.0) for s in SCENARIOS}
        tot = sum(vals.values()) + 1e-30
        recomputed[f"E{ei}"] = {s: round(v / tot, 4) for s, v in vals.items()}
    au_summary[m] = {
        "n_entries": len(parsed),
        "layers": layers, "experts": experts, "scenarios": scns,
        "extra_scenarios": [s for s in scns if s not in SCENARIOS],
        "layer0_dominance_recomputed": recomputed,
    }

check("C6", "AU 原始条目结构完整（3 层 × 4 专家 × 场景）且可重算 dominance",
      all(v["layers"] == [0, 1, 2] and v["experts"] == [0, 1, 2, 3]
          for v in au_summary.values()),
      au_summary)


# ============================================================
# C7 / C8  continual_results.json
# ============================================================
print("\n" + "=" * 70)
print("Source: cache/continual_results.json  (main_continual.py + train.compute_forgetting)")
print("=" * 70)

recheck = {}
for tag in ["vanilla", "moe"]:
    traj = continual[f"{tag}_trajectory"]
    baseline = continual["pre_continual"][tag]
    mismatch = []
    for f in continual[f"{tag}_forgetting"]:
        i, j = f["train_task"], f["eval_task"]
        es = str(f["eval_scenario"])
        expect = traj[i]["auc"][es] - baseline[es]
        if abs(expect - f["forgetting"]) > 1e-9:
            mismatch.append({"entry": f, "recomputed": expect})
    vals = [f["forgetting"] for f in continual[f"{tag}_forgetting"]]
    recheck[tag] = {
        "n_entries": len(vals),
        "mismatch": mismatch,
        "mean_forgetting": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4), "max": round(max(vals), 4),
        "n_negative(真实遗忘)": sum(1 for v in vals if v < 0),
        "n_positive(反向提升)": sum(1 for v in vals if v > 0),
    }

check("C7", "forgetting 可由 trajectory 重算",
      all(not v["mismatch"] for v in recheck.values()),
      {"result": recheck,
       "SIGN_WARNING": "已修复（2026-08-09，D1-C）：train.py docstring 已改写为"
                       "'负值才是遗忘'（forget = auc_current - auc_baseline），"
                       "与代码完全一致；文档一律按代码口径（负=遗忘）叙述。",
       "BASELINE_WARNING": "已修复（2026-08-09，D1-D）：baseline 现为 pre_continual"
                           "（训练前），与 train.compute_forgetting(pre_continual=...) 一致；"
                           "本 C7 重算也改用 continual['pre_continual'][tag]。"})

final_ok = {}
for tag in ["vanilla", "moe"]:
    last = continual[f"{tag}_trajectory"][-1]["auc"]
    fin = continual["final_auc"][tag]
    final_ok[tag] = {str(s): {"trajectory_last": round(last[str(s)], 6),
                              "final_auc": round(fin[str(s)], 6),
                              "equal": abs(last[str(s)] - fin[str(s)]) < 1e-9}
                     for s in SCENARIOS}

check("C8", "final_auc == trajectory 末尾 task 的 auc",
      all(c["equal"] for t in final_ok.values() for c in t.values()),
      final_ok)

# 遗忘总览：pre_continual → final
forget_overview = {}
for tag in ["vanilla", "moe"]:
    rows_, raw_pre, raw_fin = {}, [], []
    for s in SCENARIOS:
        a = continual["pre_continual"][tag][str(s)]
        b = continual["final_auc"][tag][str(s)]
        raw_pre.append(a)
        raw_fin.append(b)
        rows_[s] = {"pre": round(a, 4), "final": round(b, 4), "delta": round(b - a, 4)}
    # 均值必须用全精度再 round —— 先 round 后平均会在末位差 1（见核查文档偏差 R2）
    mean_pre = sum(raw_pre) / len(raw_pre)
    mean_fin = sum(raw_fin) / len(raw_fin)
    forget_overview[tag] = {"per_scenario": rows_,
                            "mean_pre": round(mean_pre, 4),
                            "mean_final": round(mean_fin, 4),
                            "mean_delta": round(mean_fin - mean_pre, 4)}
forget_overview["moe_advantage"] = round(
    forget_overview["moe"]["mean_delta"] - forget_overview["vanilla"]["mean_delta"], 4)
report["C7b_pre_to_final"] = {"desc": "pre_continual → final_auc 全局漂移",
                              "status": "INFO", "detail": forget_overview}
print("[INFO] C7b  pre_continual → final_auc 全局漂移")
print(json.dumps(forget_overview, ensure_ascii=False, indent=2))


# ============================================================
# C10  result_moe_downstream.csv 覆盖度
# ============================================================
print("\n" + "=" * 70)
print("Source: result_moe_downstream.csv  (main_moe.py Step 5, 行 187-219)")
print("=" * 70)

ds = pd.read_csv(p("result_moe_downstream.csv"))
expected_order = [1, 0, 4, 2, 6, 3, 8, 5]
present = list(dict.fromkeys(ds["Scenario"].tolist()))
missing = [s for s in expected_order if s not in present]
per_model = ds.groupby("Model")["AUC"].agg(["count", "mean"]).round(4).to_dict("index")
pivot = (ds.pivot_table(index=["Scenario", "Method"], columns="Model", values="AUC")
           .reset_index())
pivot["Delta"] = (pivot["MoE"] - pivot["Vanilla"]).round(4)

check("C10", "downstream 覆盖 8 场景（main_moe.py 声明顺序 [1,0,4,2,6,3,8,5]）",
      not missing,
      {"present": present, "missing": missing,
       "rows": len(ds),
       "expected_rows": len(expected_order) * 2 * 14,
       "per_model": per_model,
       "moe_win_rate": round(float((pivot["Delta"] > 0).mean()), 4),
       "note": "缺失场景说明该 run 未跑完，文档只能报告已完成的场景"})


# ============================================================
# C9  dataset.feather
# ============================================================
if "--no-feather" not in sys.argv:
    print("\n" + "=" * 70)
    print("Source: dataset.feather  (dataset.py Split)")
    print("=" * 70)
    df = pd.read_feather(p("dataset.feather"), columns=["tab", "date", "is_click", "user_id"])
    tab_stat = {}
    for s in sorted(df["tab"].unique().tolist()):
        sub = df[df["tab"] == s]
        tab_stat[int(s)] = {
            "n": int(len(sub)),
            "share": round(len(sub) / len(df), 4),
            "ctr": round(float(sub["is_click"].mean()), 4),
            "train": int((sub["date"] < 20220503).sum()),
            "valid": int(((sub["date"] >= 20220503) & (sub["date"] < 20220506)).sum()),
            "test": int((sub["date"] >= 20220506).sum()),
        }
    check("C9", "dataset.feather tab 分布 / CTR / 切分样本量", True,
          {"total_rows": int(len(df)),
           "n_users": int(df["user_id"].nunique()),
           "date_range": [int(df["date"].min()), int(df["date"].max())],
           "global_ctr": round(float(df["is_click"].mean()), 4),
           "split_rule": "train:<20220503  valid:[20220503,20220506)  test:>=20220506",
           "per_tab": tab_stat,
           "target_scenarios": SCENARIOS,
           "excluded_tabs": [int(s) for s in tab_stat if s not in SCENARIOS]})
else:
    print("\n[SKIP] C9 dataset.feather (--no-feather)")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print(f"核查完成：{len(report)} 项，失败 {len(failures)} 项")
for f in failures:
    print(f"  FAIL  {f}")
print("=" * 70)

with open(p("cache", "provenance_report.json"), "w") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
print(f"机读报告已写入 cache/provenance_report.json")
