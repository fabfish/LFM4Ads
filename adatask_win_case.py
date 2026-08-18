"""E17 one-vs-rest 优化器臂：s0 vs 其余场景（docs/20260818-1815 预注册）。

三个臂的唯一差异是"梯度 → 参数更新"的映射：

  S  SharedAdamWBatchOptimizer   AdamW((g_s0+g_rest)/2)          ← 复用 task_role_optimizer
  T  TwoGroupOptimizer(mode="T") (Adam_s0(g_s0)+Adam_rest(g_rest))/2，统一超参
  A  TwoGroupOptimizer(mode="A") 与 T 状态语义逐位一致；仅对 sparse_expert 的
     post-preconditioner 方向乘 task-axis AU 因子（f_g 定义见 task_axis_factors）

审计哨兵语义由本模块的纯函数保证：
  - ``task_axis_factors``：唯一因子生成入口（mean AU 跨两组、同参数块比较）
  - ``TwoGroupOptimizer``：AU 只影响方向，绝不写入 m/v/steps（哨兵 4）
  - 非专家参数在 A/T 下走完全相同的代码路径（哨兵 6）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call, jacrev

from task_role_optimizer import (
    SPARSE_EXPERT,
    ParameterRoleRegistry,
    SharedAdamWBatchOptimizer,
)

NUM_GROUPS = 2
TARGET_GROUP = 0  # s0
REST_GROUP = 1
ADATASK_ALPHA = 0.5
ADATASK_BETA = 0.99
ADATASK_EPS = 1e-12
ADATASK_CLIP = (1.0 / 3.0, 3.0)


@dataclass
class GroupGradient:
    """one-vs-rest 分组梯度；``task_id`` 兼容 SharedAdamWBatchOptimizer。"""

    group_id: int
    task_id: int
    gradients: dict[str, torch.Tensor]
    active_experts: dict[int, frozenset[int]]
    loss: float
    count: int


def group_ids_for_tabs(tabs: torch.Tensor, target_tab: int) -> torch.Tensor:
    """tab==target → 组 0，其余 → 组 1。"""
    return torch.where(tabs == int(target_tab), 0, 1).to(tabs.device)


def task_axis_factors(
    au: torch.Tensor,
    alpha: float = ADATASK_ALPHA,
    clip: tuple[float, float] = ADATASK_CLIP,
    eps: float = ADATASK_EPS,
) -> torch.Tensor:
    """f_g = clip[(mean_g' AU_g' / (AU_g + eps))^alpha, clip[0], clip[1]]。

    AU 大（梯度长期更大）的任务方向被缩小 → suppress 式 case。mean 只在本次
    传入的（对该专家有方向的）组上计算。
    """
    mean_au = au.mean()
    factors = ((mean_au / (au + eps)) ** float(alpha)).clamp(
        min=float(clip[0]), max=float(clip[1]))
    return factors


def collect_group_gradients(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    registry: ParameterRoleRegistry,
    target_tab: int = 0,
) -> list[GroupGradient]:
    """一次 jacrev（2 维损失向量 = 2 个 vjp）提取 {s0, rest} 两组梯度。

    组损失 = 组内样本逐行 BCE 的均值；与 collect_task_gradients 完全同构，
    仅把 15 维场景向量换成 2 维分组向量。
    """
    tabs = batch["tab"].long()
    group_ids = group_ids_for_tabs(tabs, target_tab)
    generated_fields = {"logit", "_gate", "_cr"}
    model_inputs = {
        name: value for name, value in batch.items()
        if name not in generated_fields
    }
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def group_loss_vector(current_parameters):
        functional_batch = dict(model_inputs)
        functional_call(
            model, (current_parameters, buffers), (functional_batch,))
        if "logit" not in functional_batch or "_gate" not in functional_batch:
            raise ValueError("model must write batch['logit'] and batch['_gate']")
        labels = functional_batch["is_click"].to(
            functional_batch["logit"].dtype)
        per_row = F.binary_cross_entropy_with_logits(
            functional_batch["logit"], labels, reduction="none")
        sums = torch.zeros(
            NUM_GROUPS, device=per_row.device, dtype=per_row.dtype)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, group_ids, per_row)
        counts.scatter_add_(0, group_ids, torch.ones_like(per_row))
        losses = sums / counts.clamp_min(1)
        auxiliary = (
            losses.detach(),
            tuple(gate.detach() for gate in functional_batch["_gate"]),
            counts.detach(),
        )
        return losses, auxiliary

    jacobians, (losses, gates, counts) = jacrev(
        group_loss_vector, has_aux=True)(parameters)

    items: list[GroupGradient] = []
    for group_id in range(NUM_GROUPS):
        count = int(counts[group_id])
        if count == 0:
            continue
        mask = group_ids == group_id
        active: dict[int, frozenset[int]] = {}
        for layer_index, gate in enumerate(gates):
            selected = torch.nonzero(
                gate[mask].detach().abs().sum(dim=0) > 0,
                as_tuple=False,
            ).flatten()
            active[layer_index] = frozenset(
                int(index) for index in selected.tolist())
        gradients = {
            spec.name: jacobians[spec.name][group_id].detach()
            for spec in registry.specs
        }
        items.append(GroupGradient(
            group_id=group_id,
            task_id=group_id,
            gradients=gradients,
            active_experts=active,
            loss=float(losses[group_id]),
            count=count,
        ))
    return items


def build_shared_optimizer(model, lr, betas, eps, weight_decay):
    """S 臂：AdamW(两组梯度均值)。"""
    registry = ParameterRoleRegistry.from_model(model)
    return SharedAdamWBatchOptimizer(
        registry=registry,
        learning_rate=float(lr),
        betas=tuple(betas),
        eps=float(eps),
        weight_decay=float(weight_decay),
    )


class TwoGroupOptimizer:
    """T/A 臂：两组独立 Adam 状态，方向平均；A 仅对专家方向乘 task-axis 因子。

    冻结语义（与预注册 §3.2 一致）：
      - 每参数两组状态 {m,v}，组 g 未激活（专家未被该组路由）时 m/v/step 冻结
      - 方向只累加本次激活组的 d_g = m̂_g/(sqrt(v̂_g)+eps)，再除以 NUM_GROUPS(=2)
      - decoupled weight decay 每步作用于全部参数（与 AdamW/S 臂一致）
      - A 模式：AU_{p,g} ← beta*AU + (1-beta)*mean(g^2)；f 由 task_axis_factors
        计算；因子只乘方向，绝不改变 m/v/steps
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 5e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        mode: str = "T",
        alpha: float = ADATASK_ALPHA,
        au_beta: float = ADATASK_BETA,
        f_clip: tuple[float, float] = ADATASK_CLIP,
    ):
        if mode not in ("T", "A"):
            raise ValueError(f"unknown mode {mode!r}")
        self.registry = ParameterRoleRegistry.from_model(model)
        self.mode = mode
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.alpha = float(alpha)
        self.au_beta = float(au_beta)
        self.f_clip = (float(f_clip[0]), float(f_clip[1]))
        self.force_identity_factors = False  # 审计哨兵 3 用
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._steps: dict[str, torch.Tensor] = {}
        self._au: dict[str, torch.Tensor] = {}
        for spec in self.registry.specs:
            param = spec.parameter
            self._m[spec.name] = torch.zeros(
                (NUM_GROUPS,) + tuple(param.shape),
                device=param.device, dtype=param.dtype)
            self._v[spec.name] = torch.zeros_like(self._m[spec.name])
            self._steps[spec.name] = torch.zeros(
                (NUM_GROUPS,), device=param.device, dtype=torch.long)
            if spec.role == SPARSE_EXPERT:
                self._au[spec.name] = torch.zeros(
                    (NUM_GROUPS,), device=param.device, dtype=param.dtype)
        self._reset_epoch_stats()

    # ------------------------------------------------------------------
    def _reset_epoch_stats(self) -> None:
        self._epoch_factors_applied = 0
        self._epoch_factors_clipped = 0
        self._epoch_factor_sum = torch.zeros(
            NUM_GROUPS, dtype=torch.float64)
        self._epoch_factor_count = torch.zeros(NUM_GROUPS, dtype=torch.float64)

    def epoch_adatask_stats(self) -> dict[str, float] | None:
        """本 epoch 的 AdaTask 因子统计；T 模式返回 None。"""
        if self.mode != "A":
            return None
        applied = int(self._epoch_factors_applied)
        if applied == 0:
            return {"applied": 0, "clip_rate": 0.0,
                    "mean_f_target": 0.0, "mean_f_rest": 0.0}
        clip_rate = float(self._epoch_factors_clipped) / applied
        counts = self._epoch_factor_count.clamp_min(1.0)
        return {
            "applied": applied,
            "clip_rate": clip_rate,
            "mean_f_target": float(self._epoch_factor_sum[0] / counts[0]),
            "mean_f_rest": float(self._epoch_factor_sum[1] / counts[1]),
        }

    def reset_epoch_stats(self) -> None:
        self._reset_epoch_stats()

    # ------------------------------------------------------------------
    def _expert_active(self, spec, item: GroupGradient) -> bool:
        selected = item.active_experts.get(spec.layer_index or 0, frozenset())
        return spec.expert_index in selected

    @torch.no_grad()
    def step(self, groups: Sequence[GroupGradient]) -> dict[str, float]:
        if not groups:
            raise ValueError("no group gradients")
        beta1, beta2 = self.betas
        decay_scale = 1.0 - self.lr * self.weight_decay
        stats = {"loss": sum(g.loss for g in groups) / len(groups)}

        for spec in self.registry.specs:
            param = spec.parameter
            active_ids: list[int] = []
            grad_rows: list[torch.Tensor] = []
            for item in groups:
                if (spec.role == SPARSE_EXPERT
                        and not self._expert_active(spec, item)):
                    continue
                grad = item.gradients.get(spec.name)
                if grad is None:
                    continue
                active_ids.append(item.group_id)
                grad_rows.append(grad)
            if not active_ids:
                # 两组都未激活该专家：只做与 AdamW 一致的每步衰减
                if self.weight_decay:
                    param.mul_(decay_scale)
                continue

            idx = torch.tensor(
                active_ids, device=param.device, dtype=torch.long)
            g_stack = torch.stack(grad_rows)
            m = self._m[spec.name]
            v = self._v[spec.name]
            steps = self._steps[spec.name]
            m[idx] = m[idx] * beta1 + g_stack * (1.0 - beta1)
            v[idx] = v[idx] * beta2 + g_stack.square() * (1.0 - beta2)
            steps[idx] += 1

            step_now = steps[idx].to(param.dtype)
            bc_shape = (-1,) + (1,) * param.ndim
            m_hat = m[idx] / (
                1.0 - beta1 ** step_now.view(bc_shape))
            v_hat = v[idx] / (
                1.0 - beta2 ** step_now.view(bc_shape))
            direction = m_hat / (v_hat.sqrt() + self.eps)

            factors = None
            if (self.mode == "A" and spec.role == SPARSE_EXPERT
                    and not self.force_identity_factors):
                au = self._au[spec.name]
                au[idx] = (
                    au[idx] * self.au_beta
                    + g_stack.square().mean(
                        dim=tuple(range(1, g_stack.ndim)))
                    * (1.0 - self.au_beta))
                factors = task_axis_factors(
                    au[idx].clone(), alpha=self.alpha, clip=self.f_clip)
                self._epoch_factors_applied += int(factors.numel())
                self._epoch_factors_clipped += int(
                    ((factors <= self.f_clip[0] + 1e-9)
                     | (factors >= self.f_clip[1] - 1e-9)).sum())
                for row, gid in enumerate(active_ids):
                    self._epoch_factor_sum[gid] += float(factors[row])
                    self._epoch_factor_count[gid] += 1

            if factors is not None:
                direction = direction * factors.view(bc_shape).to(
                    direction.dtype)

            update = direction.sum(dim=0) / NUM_GROUPS
            new_param = param * decay_scale - self.lr * update
            param.copy_(new_param)

        stats["mean_loss"] = stats["loss"]
        return stats

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "mode": self.mode,
            "num_groups": NUM_GROUPS,
            "m": dict(self._m),
            "v": dict(self._v),
            "steps": dict(self._steps),
            "au": dict(self._au),
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        if payload.get("format_version") != 1:
            raise ValueError("unsupported two-group optimizer state")
        if payload.get("mode") != self.mode:
            raise ValueError("checkpoint mode mismatch")
        for key, store in (("m", self._m), ("v", self._v),
                           ("steps", self._steps), ("au", self._au)):
            restored = payload.get(key, {})
            for name, value in restored.items():
                if name in store:
                    store[name].copy_(value.to(store[name].device))
