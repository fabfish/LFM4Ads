"""Zero-cost diagnosis: is AdaTask's "multiply the gradient" a no-op under AdamW?

Why this exists
---------------
`model.AdaTaskOptimizer.modulate_and_zero_grad` (model.py:450-478) implements
per-scenario adaptive learning rates as

    param.grad.mul_(f)      # f derived from the AU (squared-grad EMA)
    self.optimizer.step()   # self.optimizer is a plain torch.optim.AdamW

But AdamW's update is (ignoring eps/weight-decay)

    Δθ = -lr * m_t / sqrt(v_t),  m_t = EMA(g),  v_t = EMA(g²)

Scaling every gradient by a constant c gives m -> c·m and v -> c²·v, hence
m/sqrt(v) is UNCHANGED: Adam is scale-invariant, so a *constant* gradient
factor cannot change the trajectory at all. Only the time-VARYING part of the
factor survives, and the AU is an EMA (beta=0.99) that flattens out over
training, i.e. exactly the regime where the factor stops doing anything.

This script quantifies both statements on a 5-parameter toy problem so the
claim is reproducible in seconds and does not rely on any GPU run.

Pre-registered in docs/20260817-1410-E16-AdaTask对照预注册.md §1.
Usage: python scripts/diagnose/diagnose_adam_scale_invariance.py
"""

import json
import math
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "cache", "adam_scale_invariance.json")
STEPS = 200
DIM = 5


def trajectory(factor, steps=STEPS, lr=1e-3, wd=0.0):
    """Run AdamW on a fixed gradient sequence, scaling grads by ``factor``.

    ``factor`` is either a float (constant) or a callable step -> float.
    The gradient sequence is identical across calls (fixed seed), so any
    difference in the final parameters is caused ONLY by the scaling.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(DIM))
    opt = torch.optim.AdamW([p], lr=lr, weight_decay=wd)
    torch.manual_seed(123)
    grads = [torch.randn(DIM) for _ in range(steps)]
    for t, g in enumerate(grads):
        f = factor(t) if callable(factor) else factor
        p.grad = g.clone() * f
        opt.step()
        opt.zero_grad()
    return p.detach().clone()


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    base = trajectory(1.0)
    res = {"constant_factors": {}, "time_varying_factors": {},
           "with_weight_decay": {}}

    print("A) CONSTANT gradient factor (what a settled AU produces)")
    for c in (0.1, 0.5, 2.0, 10.0, 100.0):
        r = rel(trajectory(c), base)
        res["constant_factors"][str(c)] = r
        print(f"   f={c:<7} relative param diff vs f=1: {r:.3e}")

    print("\nB) same, with weight_decay=0.01 (decoupled, so it does not scale)")
    base_wd = trajectory(1.0, wd=0.01)
    for c in (0.1, 10.0):
        r = rel(trajectory(c, wd=0.01), base_wd)
        res["with_weight_decay"][str(c)] = r
        print(f"   f={c:<7} relative param diff vs f=1: {r:.3e}")

    print("\nC) TIME-VARYING factor (the only part that can survive)")
    cases = {
        "1+0.9*sin(t/20)": lambda t: 1.0 + 0.9 * math.sin(t / 20),
        "0.1 then 10 at t=100": lambda t: 0.1 if t < 100 else 10.0,
        "linear 0.5->2.0": lambda t: 0.5 + 1.5 * t / STEPS,
    }
    for name, fn in cases.items():
        r = rel(trajectory(fn), base)
        res["time_varying_factors"][name] = r
        print(f"   {name:<24} relative param diff vs f=1: {r:.3e}")

    worst_const = max(res["constant_factors"].values())
    best_varying = max(res["time_varying_factors"].values())
    res["verdict"] = {
        "max_rel_diff_constant": worst_const,
        "max_rel_diff_time_varying": best_varying,
        "constant_factor_is_noop": worst_const < 1e-5,
        "conclusion":
            "AdamW is scale-invariant: a constant per-task gradient factor is a "
            "numerical no-op (rel diff ~1e-7, i.e. float32 noise). Only the "
            "time-varying part of AdaTask's factor can affect training. Since "
            "the AU is an EMA that settles, the historical implementation "
            "(grad.mul_(f) then AdamW.step) is expected to be近-inert in "
            "steady state. A genuine per-task learning rate must be applied "
            "POST-preconditioner (per-param-group lr, or scaling the update).",
    }
    print(f"\nVERDICT: constant factor is a no-op = "
          f"{res['verdict']['constant_factor_is_noop']} "
          f"(max rel diff {worst_const:.2e}); time-varying factors do act "
          f"(up to {best_varying:.2e}).")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
