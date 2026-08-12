#!/usr/bin/env python
"""Stage A 样本加权关卡汇总与四态判定。

读取 cache/manifests/sample_weighting/*.json 与数值不变量标记，计算：
  - 稠密 sample 三 seed 均值、与同 seed full-batch dense 的配对差
  - 增益恢复率 R_gain 与配对残差恢复率 R_resid（禁止混称）
  - 关卡四态判定（PASS / FAIL / INCONCLUSIVE / DEBUG）
  - 分组报告 MoE 相对同 seed sample-dense 的方向一致性

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

# 整批稠密（非 per-scenario）三 seed 精确锚点。必须同 seed 配对，不能把
# 单一均值重复用于 residual 计算。数值来自 moe_pretrain_summary_vanilla_from_scratch_seed*.json。
FULL_BY_SEED = {
    42: 0.7806205759391338,
    123: 0.7815498621268214,
    456: 0.7798601329239648,
}
FULL_MEAN = sum(FULL_BY_SEED.values()) / len(FULL_BY_SEED)
EQUAL_MEAN = 0.7666  # 按场景等权稠密 3-seed 文档均值（下界参照）
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
    aucs = {}
    for seed in (42, 123, 456):
        code = f"{prefix}-{seed}"
        if code in manifests:
            aucs[seed] = manifests[code]["results"]["pooled_auc"]
    return aucs


def recovery_rates(sample_by_seed):
    """返回语义不同、必须分开报告的两种恢复率。"""
    seeds = sorted(sample_by_seed)
    sample_mean = sum(sample_by_seed[s] for s in seeds) / len(seeds)
    full_mean = sum(FULL_BY_SEED[s] for s in seeds) / len(seeds)
    denominator = full_mean - EQUAL_MEAN
    gain = (sample_mean - EQUAL_MEAN) / denominator
    paired_residual = sum(
        abs(sample_by_seed[s] - FULL_BY_SEED[s]) for s in seeds
    ) / len(seeds)
    residual = 1 - paired_residual / denominator
    return gain, residual


def decide_gate(sample_by_seed, equ):
    equ_ok = equ is not None and equ.get("status") == "pass"
    if not equ_ok:
        return "DEBUG", "sample-weighting invariant NOT pass (see equ_swg_status.json)"
    if len(sample_by_seed) < 3:
        return "DEBUG", f"seed incomplete: have {len(sample_by_seed)}/3 succeeded"
    mean = sum(sample_by_seed.values()) / len(sample_by_seed)
    diff_full = abs(mean - FULL_MEAN)
    r_gain, r_resid = recovery_rates(sample_by_seed)
    if diff_full <= DIFF_TOL and r_resid >= 0.75:
        return "PASS", (f"mean={mean:.6f} |Δfull_mean|={diff_full:.6f} "
                        f"R_gain={r_gain:.3f} R_resid_paired={r_resid:.3f}")
    if r_resid <= 0.25 or (mean - EQUAL_MEAN) <= DIFF_TOL:
        return "FAIL", (f"mean={mean:.6f} R_gain={r_gain:.3f} "
                        f"R_resid_paired={r_resid:.3f}")
    if 0.25 < r_resid < 0.75:
        return "INCONCLUSIVE", (f"mean={mean:.6f} "
                                f"R_resid_paired={r_resid:.3f} in (0.25,0.75)")
    return "DEBUG", (f"unexpected mean={mean:.6f} diff_full={diff_full:.6f} "
                     f"R_resid_paired={r_resid:.3f}")


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
    print(f"参照: full-batch dense mean={FULL_MEAN:.9f}  "
          f"equal per-scenario dense mean={EQUAL_MEAN:.4f}")
    print(f"sample-weighting 不变量: {equ['status'] if equ else 'MISSING'}")
    print("-" * 64)
    print(f"稠密 sample 三 seed: {dens}")
    if dens:
        mean = sum(dens.values()) / len(dens)
        r_gain, r_resid = recovery_rates(dens)
        print(f"  均值={mean:.9f}  |Δfull_mean|={abs(mean-FULL_MEAN):.9f}  "
              f"R_gain={r_gain:.3f}  R_resid_paired={r_resid:.3f}")
    print(f"全路由冻结 sample 三 seed: {frout}")
    print(f"全路由正常 sample 三 seed: {nrout}")
    print(f"部分路由加共享 sample 三 seed: {pshr}")

    state, reason = decide_gate(dens, equ)
    print("-" * 64)
    print(f"关卡判定: {state}")
    print(f"依据: {reason}")

    if frout and nrout and pshr and dens:
        def rel(name, aucs):
            seeds = sorted(set(aucs) & set(dens))
            deltas = [aucs[seed] - dens[seed] for seed in seeds]
            directions = ["+" if delta > 0 else "-" if delta < 0 else "0"
                          for delta in deltas]
            stable = len(set(directions)) == 1 and "0" not in directions
            verdict = "方向一致" if stable else "跨 seed 变号，未稳定匹配"
            print(f"  {name}: paired_deltas={[round(v, 7) for v in deltas]} "
                  f"directions={directions} -> {verdict}")
        print("铺开分组（同 seed 配对口径）:")
        rel("frozen fully-routed", frout)
        rel("trainable fully-routed", nrout)
        rel("partial-routed with shared expert", pshr)
        if len(nrout) == 3 and len(frout) == 3:
            deltas = [nrout[s] - frout[s] for s in sorted(nrout)]
            directions = ["+" if delta > 0 else "-" if delta < 0 else "0"
                          for delta in deltas]
            verdict = ("方向一致" if len(set(directions)) == 1 and "0" not in directions
                       else "跨 seed 变号，无稳定方向性差异")
            print(f"  trainable vs frozen: paired_deltas="
                  f"{[round(v, 7) for v in deltas]} -> {verdict}")

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
