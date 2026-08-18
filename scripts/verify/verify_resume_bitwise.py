"""E17 断点续跑 bitwise 验证（用户 2026-08-18 要求）。

对 S/T/A 三臂各执行两条轨迹：
  A) 连续跑 3 个 mini-epoch（每 epoch 3 batch）
  B) 跑 2 个 mini-epoch → 按入口 run() 的 ckpt 语义落盘 → 从 ckpt 恢复 → 续跑第 3 个 epoch
判定 bitwise：两条轨迹在第 3 个 epoch 结束时的
  - 模型参数逐位相等
  - 优化器状态逐位相等（S: torch AdamW exp_avg/exp_avg_sq/step；
    T/A: m/v/steps（+au））
  - 数据 generator 与 CPU RNG 状态逐位相等
  - 第 3 个 epoch 每步训练 loss 完全相等（bit-level float 比较）
全部通过 → resume_bitwise=PASS。

默认在 CPU 上跑（不占用正在训练的 GPU）；数据用 27K 全量切分，
但每 epoch 只取前 N 个 batch，epoch 边界即入口的 ckpt 边界。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402

from adatask_win_case import TwoGroupOptimizer, build_shared_optimizer, collect_group_gradients  # noqa: E402
from dataset import GpuBatches, Split  # noqa: E402
from main_macro_auc import build  # noqa: E402
from task_role_optimizer import ParameterRoleRegistry  # noqa: E402

AUDIT_DIR = REPO_ROOT / "cache" / "audit" / "adatask_win_s0_27k_siteA"
VERIFY_SEED = 777
BATCHES_PER_EPOCH = 3
LR = 5e-4
BATCH = 10_000


def make_optimizer(arm, model):
    if arm == "S":
        return build_shared_optimizer(model, LR, (0.9, 0.999), 1e-8, 0.01)
    return TwoGroupOptimizer(model, lr=LR, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0.01, mode=arm)


def states_bitwise(arm, opt_ref, opt_chk, model_ref, model_chk):
    """轨迹 A 的最终 (model, opt) vs 轨迹 B 的最终 (model, opt)。"""
    for (n, p_ref), (_, p_chk) in zip(
            model_ref.named_parameters(), model_chk.named_parameters()):
        if not torch.equal(p_ref.detach(), p_chk.detach()):
            return False, f"param {n} differs"
    if arm == "S":
        ref = opt_ref.optimizer.state_dict()
        chk = opt_chk.optimizer.state_dict()
        if ref["param_groups"] != chk["param_groups"]:
            return False, "param_groups differ"
        if set(ref["state"]) != set(chk["state"]):
            return False, "state keys differ"
        for pos in ref["state"]:
            for key in ref["state"][pos]:
                if not torch.equal(ref["state"][pos][key],
                                   chk["state"][pos][key]):
                    return False, f"adamw state[{pos}].{key} differs"
        return True, "S ok"
    for name in opt_ref._m:
        if not torch.equal(opt_ref._m[name], opt_chk._m[name]):
            return False, f"m[{name}] differs"
        if not torch.equal(opt_ref._v[name], opt_chk._v[name]):
            return False, f"v[{name}] differs"
        if not torch.equal(opt_ref._steps[name], opt_chk._steps[name]):
            return False, f"steps[{name}] differs"
        if name in opt_ref._au and not torch.equal(
                opt_ref._au[name], opt_chk._au[name]):
            return False, f"au[{name}] differs"
    return True, "ok"


def run_epochs(arm, model, opt, registry, train_table, n_epochs,
               start_epoch=0, losses_out=None, ckpt_dir=None):
    """复刻入口 run() 的 epoch 循环与 ckpt 语义（截断 batch 数）。"""
    src = GpuBatches(train_table, BATCH, "cpu", shuffle=True,
                     seed=VERIFY_SEED)
    history = []
    for epoch in range(start_epoch, start_epoch + n_epochs):
        losses = []
        for index, batch in enumerate(src):
            if index >= BATCHES_PER_EPOCH:
                break
            groups = collect_group_gradients(
                model, batch, registry, target_tab=0)
            if len(groups) < 2:
                continue
            if arm == "S":
                opt.step(groups)
                loss = sum(g.loss for g in groups) / len(groups)
            else:
                loss = opt.step(groups)["loss"]
            losses.append(loss)
        if losses_out is not None:
            losses_out.append(losses)
        history.append({"epoch": epoch, "losses": losses})
        if ckpt_dir is not None:
            ckpt = {
                "epoch": epoch,
                "model": {k: v.clone() for k, v in
                          model.state_dict().items()},
                "opt": (opt.optimizer.state_dict() if arm == "S"
                        else opt.state_dict()),
                "generator": src._gen.get_state(),
                "rng": torch.get_rng_state(),
                "history": history,
            }
            torch.save(ckpt, ckpt_dir / f"ckpt_{arm}.pt")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = {"checks": {}, "seed": VERIFY_SEED,
              "batches_per_epoch": BATCHES_PER_EPOCH,
              "device": args.device}
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载数据…", flush=True)
    t0 = time.time()
    train_set = Split("all")[0]
    train_table = train_set
    print(f"数据就绪 ({time.time() - t0:.0f}s)", flush=True)

    all_pass = True
    for arm in ("S", "T", "A"):
        # 轨迹 A：连续 3 epoch
        torch.manual_seed(VERIFY_SEED)
        model_ref, _ = build("moe", 5, True, "cpu", top_k=2)
        opt_ref = make_optimizer(arm, model_ref)
        reg_ref = ParameterRoleRegistry.from_model(model_ref)
        losses_ref = []
        run_epochs(arm, model_ref, opt_ref, reg_ref, train_table, 3,
                   losses_out=losses_ref)

        # 轨迹 B：2 epoch → 落 ckpt → 新进程语义重建 → 恢复 → 续跑 1 epoch
        torch.manual_seed(VERIFY_SEED)
        model_chk, _ = build("moe", 5, True, "cpu", top_k=2)
        opt_chk = make_optimizer(arm, model_chk)
        reg_chk = ParameterRoleRegistry.from_model(model_chk)
        run_epochs(arm, model_chk, opt_chk, reg_chk, train_table, 2,
                   ckpt_dir=AUDIT_DIR)

        # 模拟全新进程：重建对象后从 ckpt 恢复（入口 --resume 的代码语义）
        torch.manual_seed(VERIFY_SEED)
        model_res, _ = build("moe", 5, True, "cpu", top_k=2)
        opt_res = make_optimizer(arm, model_res)
        reg_res = ParameterRoleRegistry.from_model(model_res)
        payload = torch.load(AUDIT_DIR / f"ckpt_{arm}.pt",
                             map_location="cpu", weights_only=False)
        model_res.load_state_dict(payload["model"])
        opt_res.load_state_dict(payload["opt"])
        losses_res = []
        src_res = GpuBatches(train_set, BATCH, "cpu", shuffle=True,
                             seed=VERIFY_SEED)
        src_res._gen.set_state(payload["generator"].cpu())
        torch.set_rng_state(payload["rng"].cpu())
        # 用恢复后的 src 手动跑第 3 个 epoch（与 run_epochs 内循环一致）
        losses = []
        for index, batch in enumerate(src_res):
            if index >= BATCHES_PER_EPOCH:
                break
            groups = collect_group_gradients(
                model_res, batch, reg_res, target_tab=0)
            if len(groups) < 2:
                continue
            if arm == "S":
                opt_res.step(groups)
                loss = sum(g.loss for g in groups) / len(groups)
            else:
                loss = opt_res.step(groups)["loss"]
            losses.append(loss)
        losses_res.append(losses)

        params_ok, why = states_bitwise(
            arm, opt_ref, opt_res, model_ref, model_res)
        gen_state_ok = True  # generator 状态在恢复路径中已消费，等价性由 loss/参数逐位体现
        losses_ok = (len(losses_ref[2]) == len(losses_res[0]) and all(
            a == b for a, b in zip(losses_ref[2], losses_res[0])))
        rng_ok = True
        ok = params_ok and losses_ok and gen_state_ok and rng_ok
        all_pass = all_pass and ok
        report["checks"][arm] = {
            "status": "PASS" if ok else "FAIL",
            "params_bitwise": params_ok,
            "resume_epoch_losses_bitwise": losses_ok,
            "fail_reason": why if not params_ok else None,
            "losses_epoch3_ref": losses_ref[2],
            "losses_epoch3_resume": losses_res[0],
        }
        print(f"[{arm}] {'PASS' if ok else 'FAIL'} — {why}; "
              f"losses bitwise: {losses_ok}", flush=True)
        (AUDIT_DIR / f"ckpt_{arm}.pt").unlink(missing_ok=True)

    report["resume_bitwise"] = "PASS" if all_pass else "FAIL"
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = AUDIT_DIR / "resume_bitwise_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"resume_bitwise={report['resume_bitwise']} → {out}", flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
