#!/usr/bin/env python
"""Stage A 关卡汇总与四态判定（run-code: swg-*）。

读取 cache/manifests/sample_weighting/*.json 与 equ-swg 不变量标记，计算：
  - 稠密 sample 三 seed 均值、与 full(0.7807)/equal(0.7666) 的配对差、恢复率 R
  - 关卡四态判定（PASS / FAIL / INCONCLUSIVE / DEBUG）
  - 若已铺开，分组报告 MoE 相对 sample-dense / full-dense 的落后/匹配/超过

用法：
  python scripts/summarize_sample_weighting_gate.py
  python scripts/summarize_sample_weighting_gate.py --write-conclusion   # 终态时起草结论 md
"""

import datetime
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_DIR = os.path.join(REPO, "cache", "manifests", "sample_weighting")

# 整批稠密（非 per-scenario）精确三 seed 均值，上界参照。
# 注意：此前 docs 误记四位五入值 0.7807，导致恢复率被压到 0.997；
# 精确值 0.780677 下恢复率 R≈0.999（见 decide_gate 的 residual 形式）。
FULL = 0.780677
EQUAL = 0.7666  # 按场景等权稠密 3-seed 均值（下界参照）
DIFF_TOL = 0.0036  # 与 full 均值差容差（等于 dense per-scenario 噪声地板）


def load_succeeded():
    out = {}
    for path in glob.glob(os.path.join(MANIFEST_DIR, "swg-*.json")):
        try:
            with open(path) as f:
                m = json.load(f)
        except Exception:
            continue
        if m.get("status") == "succeeded":
            out[m["run_code"]] = m
    return out


def load_equ():
    p = os.path.join(MANIFEST_DIR, "equ_swg_status.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def group_aucs(manifests, prefix):
    aucs = []
    for s in (42, 123, 456):
        code = f"{prefix}-{s}"
        if code in manifests:
            aucs.append(manifests[code]["results"]["pooled_auc"])
    return aucs


def decide_gate(sample_aucs, equ):
    equ_ok = (equ is not None and equ.get("status") == "pass")
    if not equ_ok:
        return "DEBUG", "equ-swg invariant NOT pass (see equ_swg_status.json)"
    if len(sample_aucs) < 3:
        return "DEBUG", f"seed incomplete: have {len(sample_aucs)}/3 succeeded"
    mean = sum(sample_aucs) / 3
    diff_full = abs(mean - FULL)
    # 恢复率（residual 形式）：R = 1 - mean(|sample - full|) / (full - equal)
    # 与增益形式 R_gain=(mean-EQUAL)/(FULL-EQUAL) 在 mean≈FULL 时一致；
    # 用精确 FULL=0.780677 时 R≈0.999（此前用 0.7807 误得 0.997）。
    R = 1 - sum(abs(a - FULL) for a in sample_aucs) / (len(sample_aucs) * (FULL - EQUAL))
    R_gain = (mean - EQUAL) / (FULL - EQUAL)
    if diff_full <= DIFF_TOL and R >= 0.75:
        return "PASS", (f"mean={mean:.4f} |Δfull|={diff_full:.4f} "
                        f"R={R:.3f} (≥0.75) R_gain={R_gain:.3f}")
    if R <= 0.25 or (mean - EQUAL) <= DIFF_TOL:
        return "FAIL", f"mean={mean:.4f} R={R:.3f} (≤0.25) gain_vs_equal={mean-EQUAL:.4f}"
    if 0.25 < R < 0.75:
        return "INCONCLUSIVE", f"mean={mean:.4f} R={R:.3f} in (0.25,0.75)"
    return "DEBUG", f"unexpected mean={mean:.4f} diff_full={diff_full:.4f} R={R:.3f}"


def main():
    manifests = load_succeeded()
    equ = load_equ()

    dens = group_aucs(manifests, "swg-dens-sp")
    frout = group_aucs(manifests, "swg-frout-sp")
    nrout = group_aucs(manifests, "swg-nrout-sp")
    pshr = group_aucs(manifests, "swg-pshr-sp")

    print("=" * 64)
    print("Stage A 样本加权关卡汇总")
    print("=" * 64)
    print(f"参照: full-batch dense={FULL}  equal per-scenario dense={EQUAL}")
    print(f"equ-swg 不变量: {equ['status'] if equ else 'MISSING'}")
    print("-" * 64)
    print(f"稠密 sample 三 seed: {dens}")
    if dens:
        mean = sum(dens) / len(dens)
        print(f"  均值={mean:.4f}  |Δfull|={abs(mean-FULL):.4f}  "
              f"R={(mean-EQUAL)/(FULL-EQUAL):.3f}")
    print(f"全路由冻结 sample 三 seed: {frout}")
    print(f"全路由正常 sample 三 seed: {nrout}")
    print(f"部分路由加共享 sample 三 seed: {pshr}")

    state, reason = decide_gate(dens, equ)
    print("-" * 64)
    print(f"关卡判定: {state}")
    print(f"依据: {reason}")

    if frout and nrout and pshr and dens:
        dmean = sum(dens) / 3
        def rel(name, aucs):
            m = sum(aucs) / 3
            vs_d = m - dmean
            vs_full = m - FULL
            tag = "超过" if vs_full > DIFF_TOL else ("匹配" if abs(vs_full) <= DIFF_TOL else "落后")
            print(f"  {name}: mean={m:.4f}  vs sample-dense={vs_d:+.4f}  vs full={vs_full:+.4f} -> {tag}")
        print("铺开分组（相对口径）:")
        rel("frout(冻结)", frout)
        rel("nrout(正常)", nrout)
        rel("pshr(共享)", pshr)
        if len(nrout) == 3 and len(frout) == 3:
            nm = sum(nrout)/3; fm = sum(frout)/3
            verdict = "正常路由稳定优于冻结" if (nm - fm) > 0.0036 else "正常路由未稳定优于冻结"
            print(f"  normal vs frozen: Δ={nm-fm:+.4f} -> {verdict}")

    print("=" * 64)

    if "--write-conclusion" in sys.argv and state in ("PASS", "FAIL"):
        write_conclusion(state, reason, dens, frout, nrout, pshr, equ)

    return 0 if state in ("PASS", "FAIL", "INCONCLUSIVE") else 1


def write_conclusion(state, reason, dens, frout, nrout, pshr, equ):
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(REPO, "docs", f"{now}-按场景训练代价消除结论.md")
    lines = [
        f"# 按场景训练代价消除结论（{now}）",
        "",
        f"- 关卡判定：**{state}**",
        f"- 依据：{reason}",
        f"- equ-swg 不变量：{equ['status'] if equ else 'MISSING'}",
        "",
        "## 测量口径",
        "- device=cuda:0 独占；seed=42/123/456；lr=1e-3；beta2=0.999；shuffle=True；"
        "batch=10000；scenario-loss-weighting=sample",
        "",
        "## 逐 seed 结果（pooled AUC）",
        f"- 稠密 sample：{dens}",
        f"- 全路由冻结 sample：{frout}",
        f"- 全路由正常 sample：{nrout}",
        f"- 部分路由加共享 sample：{pshr}",
        "",
        "## 后续动作",
        ("- 关卡 PASS：解锁第九节 9 次 MoE 铺开（见驱动文档）。" if state == "PASS"
         else "- 关卡未 PASS：停止旧 MoE 铺开，记录负结果（见驱动文档 §九）。"),
        "",
        "> 本结论由 summarize 脚本起草，人工复核后回填驱动文档与 DRIVERS.md。",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[conclusion] draft -> {path}")


if __name__ == "__main__":
    sys.exit(main())
