"""调试：同一模型对象上 T 步与 A 步的状态/参数差异定位。"""

import sys
from pathlib import Path

REPO_ROOT = Path("/root/Documents/Lfm4ads")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402

from adatask_win_case import TwoGroupOptimizer, collect_group_gradients  # noqa: E402
from dataset import GpuBatches, Split  # noqa: E402
from main_macro_auc import build  # noqa: E402
from task_role_optimizer import SPARSE_EXPERT, ParameterRoleRegistry  # noqa: E402

device = "cuda:1"
train_set = Split("all")[0]
torch.manual_seed(101)
torch.cuda.manual_seed_all(101)
model, _ = build("moe", 5, True, device, top_k=2)
reg = ParameterRoleRegistry.from_model(model)
opt = TwoGroupOptimizer(model, mode="T")

src = GpuBatches(train_set, 10_000, device, shuffle=True, seed=101)
batch = next(iter(src))
groups = collect_group_gradients(model, batch, reg, target_tab=0)
print("active g0:", {k: sorted(v) for k, v in groups[0].active_experts.items()})
print("active g1:", {k: sorted(v) for k, v in groups[1].active_experts.items()})

params_snap = {n: p.detach().clone() for n, p in model.named_parameters()}
m_snap = {n: t.clone() for n, t in opt._m.items()}
v_snap = {n: t.clone() for n, t in opt._v.items()}
steps_snap = {n: t.clone() for n, t in opt._steps.items()}

opt.step(groups)
params_T = {n: p.detach().clone() for n, p in model.named_parameters()}
states_T = {k: {n: t.clone() for n, t in d.items()}
            for k, d in (("m", opt._m), ("v", opt._v), ("steps", opt._steps))}

# 还原
with torch.no_grad():
    for n, p in model.named_parameters():
        p.copy_(params_snap[n])
    for n in opt._m:
        opt._m[n].copy_(m_snap[n])
        opt._v[n].copy_(v_snap[n])
        opt._steps[n].copy_(steps_snap[n])

opt.mode = "A"
groups2 = collect_group_gradients(model, batch, reg, target_tab=0)
grad_equal = all(
    torch.equal(groups[0].gradients[n], groups2[0].gradients[n])
    for n in groups[0].gradients)
print("same-batch grads bitwise equal (g0):", grad_equal)
opt.step(groups2)
params_A = {n: p.detach().clone() for n, p in model.named_parameters()}

n_expert_diff = 0
n_other_diff = 0
expert_names = {s.name for s in reg.specs if s.role == SPARSE_EXPERT}
au_examples = []
for name in params_T:
    if name in expert_names:
        if not torch.equal(params_T[name], params_A[name]):
            n_expert_diff += 1
            if len(au_examples) < 3:
                au_examples.append(
                    (name, [float(x) for x in opt._au[name]],
                     float((params_T[name] - params_A[name]).abs().max())))
    elif not torch.equal(params_T[name], params_A[name]):
        n_other_diff += 1
states_eq = all(
    torch.equal(opt._m[n], states_T["m"][n])
    and torch.equal(opt._v[n], states_T["v"][n])
    and torch.equal(opt._steps[n], states_T["steps"][n])
    for n in opt._m)
print("expert params differ:", n_expert_diff,
      " other params differ:", n_other_diff)
print("states bitwise equal after A:", states_eq)
print("AU examples (name, [au0,au1], |dT-dA|max):", au_examples)
