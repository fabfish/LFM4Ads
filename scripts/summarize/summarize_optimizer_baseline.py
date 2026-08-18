"""F1 — optimizer state-sharing baseline: summarise + direction.

Reads the 8 run jsons (4 arms x 2 seeds) produced by
``run_optimizer_baseline.py`` and reports each optimizer's delta vs
``shared_adamw`` (same seed) on test macro AUC.

Exploration stage (2 seeds + subsampled train): reports DIRECTION + MAGNITUDE
only, no formal verdict (formal needs 4 seeds + full data). Reference floor is
E10's 0.00135 (full-data macro noise floor) — printed for scale, NOT used as
a verdict threshold here.

Usage: python scripts/summarize/summarize_optimizer_baseline.py
"""

import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "cache", "task_role_optimizer_27k_siteA")
ARMS = ("shared_adamw", "task_state_uniform", "dual_optim_plus", "role_isolated")
ARM_LABEL = {
    "shared_adamw": "共享 AdamW（对照）",
    "task_state_uniform": "DualOptim（每场景独立状态）",
    "dual_optim_plus": "DualOptim+（共享 base+残差）",
    "role_isolated": "SkewAdam（角色分层 lr）",
}
SEEDS = (42, 123)
REFERENCE_FLOOR = 0.00135  # E10 full-data macro floor (scale reference only)


def load(arm, seed):
    p = os.path.join(OUT_DIR, f"run_f1_{arm}_s{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    runs = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    missing = [k for k, v in runs.items() if v is None]
    if missing:
        print(f"[wait] {len(missing)} run(s) not finished: {missing}")
        return

    control = {s: runs[("shared_adamw", s)]["test"]["macro"] for s in SEEDS}
    print("# F1 优化器状态共享度 baseline（探索：2 seed × 3.9% 数据 × 5 epoch）\n")
    print(f"对照 shared_adamw test macro: "
          + " ".join(f"s{s}={control[s]:.6f}" for s in SEEDS))
    print(f"参考噪声地板（E10 全数据 macro）= {REFERENCE_FLOOR}\n")

    print("| 臂 | 方法 | Δ vs shared(seed42) | Δ vs shared(seed123) | 均值 | 同号 |")
    print("|---|---|---|---|---|---|")
    report = {}
    for arm in ARMS:
        if arm == "shared_adamw":
            continue
        deltas = [runs[(arm, s)]["test"]["macro"] - control[s] for s in SEEDS]
        mean = st.mean(deltas)
        same_sign = (all(d > 0 for d in deltas) or all(d < 0 for d in deltas))
        sign = "+" if mean > 0 else "-"
        report[arm] = {"deltas": deltas, "mean": mean, "same_sign": same_sign}
        print(f"| {arm} | {ARM_LABEL[arm]} | {deltas[0]:+.6f} | {deltas[1]:+.6f} "
              f"| {sign}{abs(mean):.6f} | {'是' if same_sign else '否'} |")

    print("\n## 方向结论（探索，非正式判定）\n")
    for arm, r in report.items():
        d = r["deltas"]
        mag = "高于参考地板" if abs(r["mean"]) > REFERENCE_FLOOR else "低于参考地板"
        sign = "正" if r["mean"] > 0 else "负"
        print(f"- **{ARM_LABEL[arm]}**：Δ 均值 {r['mean']:+.6f}，"
              f"2 seed {'同号' if r['same_sign'] else '异号'}，"
              f"{sign}向且量级{mag}")
    print("\n（注意：3.9% 数据 + 5 epoch 的探索口径，量级与 E10 全数据不可直接比较；"
          "正式判定需 4 seed 全数据）")

    out = {"control": control, "report": report,
           "reference_floor": REFERENCE_FLOOR, "seeds": list(SEEDS)}
    dst = os.path.join(OUT_DIR, "f1_report.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
