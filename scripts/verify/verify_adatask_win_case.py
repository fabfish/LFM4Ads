"""E17 启动前审计（预注册 20260818-1815 §4，九项全 PASS 才允许长跑）。

用法：
  LFM_DATASET=dataset_27k.feather LFM_VOCAB_JSON=cache/fields_27k.json \
  LFM_SAMPLE_COUNTS_JSON=cache/sample_counts_27k.json \
  python scripts/verify/verify_adatask_win_case.py --device cuda:0

产物：cache/audit/adatask_win_s0_27k_siteA/prelaunch_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from adatask_win_case import (  # noqa: E402
    NUM_GROUPS,
    TwoGroupOptimizer,
    collect_group_gradients,
    task_axis_factors,
)
from dataset import GpuBatches, Split  # noqa: E402
from main_macro_auc import EXPECTED_COUNTS, build, evaluate_all  # noqa: E402
from main_task_role_optimizer import router_statistics  # noqa: E402
from task_role_optimizer import SPARSE_EXPERT, ParameterRoleRegistry  # noqa: E402
from task_role_optimizer_protocol import EXPECTED_MODEL_PARAMS  # noqa: E402

AUDIT_DIR = REPO_ROOT / "cache" / "audit" / "adatask_win_s0_27k_siteA"
AUDIT_SEED = 101
TARGET_TAB = 0
REL_TOL_GROUP_GRAD = 5e-6
MAX_ABS_F1_SENTINEL = 1e-7
EFFICIENCY_BATCHES = 100
SINGLE_RUN_BUDGET_HOURS = 12.0


def _check(report: dict, name: str, ok: bool, detail) -> bool:
    report["checks"][name] = {
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
    }
    print(f"[{name}] {'PASS' if ok else 'FAIL'} — {detail}", flush=True)
    return ok


def _tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_seeded(device):
    torch.manual_seed(AUDIT_SEED)
    torch.cuda.manual_seed_all(AUDIT_SEED)
    return build("moe", 5, True, device, top_k=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device
    report: dict = {"checks": {}, "device": device, "seed": AUDIT_SEED}
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- S1: 样本计数 ----------
    train_set, valid_set, test_set = Split("all")
    counts = {"train": len(train_set), "valid": len(valid_set),
              "test": len(test_set)}
    s1 = _check(report, "s1_sample_counts", counts == dict(EXPECTED_COUNTS),
                {"counts": counts, "expected": dict(EXPECTED_COUNTS)})

    # ---------- S7: 参数量 / 同 seed 初始化一致 / 数据生成器一致 ----------
    model_a, info_a = build_seeded(device)
    model_b, info_b = build_seeded(device)
    same_init = all(
        _tensor_digest(pa) == _tensor_digest(pb)
        for (na, pa), (nb, pb) in zip(
            model_a.named_parameters(), model_b.named_parameters()))
    gen_a = GpuBatches(train_set, 10_000, device, shuffle=True,
                       seed=AUDIT_SEED)
    batch_a = {k: v.clone() for k, v in next(iter(gen_a)).items()}
    del gen_a
    torch.cuda.empty_cache()
    gen_b = GpuBatches(train_set, 10_000, device, shuffle=True,
                       seed=AUDIT_SEED)
    batch_b = next(iter(gen_b))
    same_batches = all(
        torch.equal(batch_a[k], batch_b[k]) for k in batch_a)
    del gen_b
    torch.cuda.empty_cache()
    s7 = _check(
        report, "s7_params_and_init",
        info_a["total_params"] == EXPECTED_MODEL_PARAMS
        and info_a == info_b and same_init and same_batches,
        {"total_params": info_a["total_params"],
         "expected_params": EXPECTED_MODEL_PARAMS,
         "same_init": same_init, "same_first_batch": same_batches})

    registry = ParameterRoleRegistry.from_model(model_a)
    expert_specs = [s for s in registry.specs if s.role == SPARSE_EXPERT]
    report["expert_specs"] = len(expert_specs)

    # ---------- S8: 路由语义 ----------
    routing = router_statistics(model_a)["report"]
    s8 = _check(
        report, "s8_routing_semantics",
        int(model_a.top_k) == 2
        and len({(s.layer_index, s.expert_index) for s in expert_specs})
        == 5 * len(model_a.cross_layers),
        {"top_k": int(model_a.top_k),
         "n_expert_param_tensors": len(expert_specs),
         "n_layers": len(model_a.cross_layers),
         "init_expert_coverage": routing.get("expert_coverage"),
         "init_expert_loads": routing.get("expert_loads")})

    # ---------- S2: 组梯度等价（jacrev 平均 vs 直接 backward） ----------
    groups = collect_group_gradients(
        model_a, batch_a, registry, target_tab=TARGET_TAB)
    both_present = len(groups) == NUM_GROUPS and all(
        g.count > 0 for g in groups)
    params = [p for _, p in model_a.named_parameters()]
    inputs = {k: v for k, v in batch_a.items()
              if k not in ("logit", "_gate", "_cr")}
    out = dict(inputs)
    model_a(out)
    per_row = F.binary_cross_entropy_with_logits(
        out["logit"], out["is_click"].to(out["logit"].dtype),
        reduction="none")
    tabs = batch_a["tab"].long()
    group_ids = torch.where(tabs == TARGET_TAB, 0, 1).to(device)
    sums = torch.zeros(NUM_GROUPS, device=device, dtype=per_row.dtype)
    counts_g = torch.zeros_like(sums)
    sums.scatter_add_(0, group_ids, per_row)
    counts_g.scatter_add_(0, group_ids, torch.ones_like(per_row))
    losses = sums / counts_g.clamp_min(1)
    direct = torch.autograd.grad(losses.mean(), params)
    max_rel = 0.0
    for (name, _), d in zip(model_a.named_parameters(), direct):
        merged = (groups[0].gradients[name]
                  + groups[1].gradients[name]) / 2
        denom = float(d.abs().max()) if d.numel() else 0.0
        diff = float((merged - d).abs().max()) if d.numel() else 0.0
        max_rel = max(max_rel, diff / max(denom, 1e-12))
    s2 = _check(report, "s2_group_grad_equivalence",
                both_present and max_rel <= REL_TOL_GROUP_GRAD,
                {"max_rel_err": max_rel, "tol": REL_TOL_GROUP_GRAD,
                 "group_counts": [g.count for g in groups],
                 "group_losses": [g.loss for g in groups]})

    groups2 = collect_group_gradients(
        model_a, batch_a, registry, target_tab=TARGET_TAB)
    deterministic = all(
        torch.equal(groups[i].gradients[n], groups2[i].gradients[n])
        for i in range(NUM_GROUPS) for n in groups[0].gradients)
    s2b = _check(report, "s2b_collector_determinism", deterministic,
                 {"bitwise_equal": deterministic})

    # ---------- S3/S4/S6: 同对象快照法（T 步 → 还原 → A 步） ----------
    # 预注册语义哨兵检验的是"算法不把因子写进状态"，必须在同一模型对象上
    # 逐步对比；跨对象前向存在合法的 fp32 逐位差异，不能用于本哨兵。
    model_x, _ = build_seeded(device)
    reg_x = ParameterRoleRegistry.from_model(model_x)
    opt_x = TwoGroupOptimizer(model_x, mode="T")
    src = GpuBatches(train_set, 10_000, device, shuffle=True,
                     seed=AUDIT_SEED)
    batch_x = next(iter(src))
    del src
    torch.cuda.empty_cache()
    groups_x = collect_group_gradients(
        model_x, batch_x, reg_x, target_tab=TARGET_TAB)
    groups_x2 = collect_group_gradients(
        model_x, batch_x, reg_x, target_tab=TARGET_TAB)
    same_collect = all(
        torch.equal(groups_x[0].gradients[n], groups_x2[0].gradients[n])
        for n in groups_x[0].gradients)

    def _snapshot():
        return (
            {n: p.detach().clone()
             for n, p in model_x.named_parameters()},
            {n: t.clone() for n, t in opt_x._m.items()},
            {n: t.clone() for n, t in opt_x._v.items()},
            {n: t.clone() for n, t in opt_x._steps.items()},
        )

    def _restore(snap):
        params, ms, vs, steps = snap
        with torch.no_grad():
            for n, p in model_x.named_parameters():
                p.copy_(params[n])
            for n in opt_x._m:
                opt_x._m[n].copy_(ms[n])
                opt_x._v[n].copy_(vs[n])
                opt_x._steps[n].copy_(steps[n])

    snap0 = _snapshot()

    # T 步
    opt_x.mode = "T"
    opt_x.step(groups_x)
    snap_T = _snapshot()

    # A 步（从同一起点）
    _restore(snap0)
    opt_x.mode = "A"
    opt_x.step(groups_x2)
    snap_A = _snapshot()

    # A(f≡1) 步（从同一起点）
    _restore(snap0)
    opt_x.mode = "A"
    opt_x.force_identity_factors = True
    opt_x.step(groups_x2)
    snap_I = _snapshot()
    opt_x.force_identity_factors = False

    def _state_diff(sa, sb):
        return max(
            max(float((sa[1][n] - sb[1][n]).abs().max()),
                float((sa[2][n] - sb[2][n]).abs().max()))
            for n in sa[1])

    def _states_bitwise(sa, sb):
        return all(
            torch.equal(sa[1][n], sb[1][n])
            and torch.equal(sa[2][n], sb[2][n])
            and torch.equal(sa[3][n], sb[3][n]) for n in sa[1])

    f1_param_diff = max(
        float((a - b).abs().max())
        for (n, a), (_, b) in zip(
            snap_T[0].items(), snap_I[0].items()))
    s3 = _check(report, "s3_f1_update_equivalence",
                f1_param_diff <= MAX_ABS_F1_SENTINEL
                and _state_diff(snap_T, snap_I) <= MAX_ABS_F1_SENTINEL
                and _states_bitwise(snap_T, snap_I) and same_collect,
                {"param_max_abs": f1_param_diff,
                 "state_max_abs": _state_diff(snap_T, snap_I),
                 "states_bitwise": _states_bitwise(snap_T, snap_I),
                 "collector_repeatable": same_collect,
                 "tol": MAX_ABS_F1_SENTINEL})

    au_nonzero = any(
        float(opt_x._au[n].abs().sum()) > 0 for n in opt_x._au)
    s4 = _check(report, "s4_post_preconditioner_states_untouched",
                _states_bitwise(snap_T, snap_A) and au_nonzero,
                {"states_bitwise_equal": _states_bitwise(snap_T, snap_A),
                 "au_updated": au_nonzero})

    role_ok = True
    n_diff_experts = 0
    for (name, pT), (_, pA) in zip(
            snap_T[0].items(), snap_A[0].items()):
        spec = next(s for s in reg_x.specs if s.name == name)
        if spec.role == SPARSE_EXPERT:
            if not torch.equal(pT, pA):
                n_diff_experts += 1
        elif not torch.equal(pT, pA):
            role_ok = False
            break
    s6 = _check(report, "s6_role_isolation",
                role_ok and n_diff_experts > 0,
                {"non_expert_bitwise_equal": role_ok,
                 "n_expert_param_tensors_changed": n_diff_experts})

    # ---------- S5: task-axis 因子语义 ----------
    f1_ = task_axis_factors(torch.tensor([4.0, 1.0]))
    f2_ = task_axis_factors(torch.tensor([1.0, 4.0]))
    s5 = _check(
        report, "s5_task_axis_factors",
        bool(f1_[0] < f1_[1]) and bool(f2_[0] > f2_[1])
        and float(f1_.min()) >= 1.0 / 3.0
        and float(f1_.max()) <= 3.0,
        {"f_au4_vs1": [float(x) for x in f1_],
         "f_au1_vs4": [float(x) for x in f2_]})

    # ---------- S9: 效率 smoke ----------
    model_e, _ = build_seeded(device)
    reg_e = ParameterRoleRegistry.from_model(model_e)
    opt_e = TwoGroupOptimizer(model_e, mode="T")
    src_e = GpuBatches(train_set, 10_000, device, shuffle=True,
                       seed=AUDIT_SEED)
    it_e = iter(src_e)
    for _ in range(5):
        b = next(it_e)
        opt_e.step(collect_group_gradients(
            model_e, b, reg_e, TARGET_TAB))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(EFFICIENCY_BATCHES):
        b = next(it_e)
        opt_e.step(collect_group_gradients(
            model_e, b, reg_e, TARGET_TAB))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    sec_per_batch = (time.time() - t0) / EFFICIENCY_BATCHES
    batches_per_epoch = (len(train_set) + 9_999) // 10_000
    t_valid = time.time()
    evaluate_all(model_e, GpuBatches(
        valid_set, 10_000, device, shuffle=False))
    valid_sec = time.time() - t_valid
    per_epoch = sec_per_batch * batches_per_epoch + valid_sec
    run_hours = per_epoch * 20 / 3600.0
    s9 = _check(report, "s9_efficiency_budget",
                run_hours <= SINGLE_RUN_BUDGET_HOURS,
                {"sec_per_train_batch": round(sec_per_batch, 3),
                 "batches_per_epoch": batches_per_epoch,
                 "valid_eval_seconds": round(valid_sec, 1),
                 "est_epoch_seconds": round(per_epoch, 1),
                 "est_20epoch_hours": round(run_hours, 2),
                 "budget_hours": SINGLE_RUN_BUDGET_HOURS})

    # ---------- 汇总落盘 ----------
    all_pass = all([s1, s2, s2b, s3, s4, s5, s6, s7, s8, s9])
    report["all_pass"] = bool(all_pass)
    report["gate"] = "pass" if all_pass else "blocked"
    try:
        report["git_rev"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True).strip()
    except subprocess.CalledProcessError:
        report["git_rev"] = None
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = AUDIT_DIR / "prelaunch_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"all_pass={all_pass} → {out}", flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
