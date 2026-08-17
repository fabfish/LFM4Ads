"""按参数角色隔离子任务优化器。

同一个混合场景批次先产生每个场景自己的梯度和预条件方向，最后每个参数
只写入一次。共享主干、稀疏专家、路由器、任务输出头和未来始终启用的
共享专家使用不同的状态与活动规则。
"""

from __future__ import annotations

import copy
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


SHARED_EMBEDDING = "shared_embedding"
SHARED_BACKBONE = "shared_backbone"
SPARSE_EXPERT = "sparse_expert"
ROUTER = "router"
TASK_HEAD = "task_head"
ALWAYS_ON_SHARED_EXPERT = "always_on_shared_expert"
PARAMETER_ROLES = (
    SHARED_EMBEDDING,
    SHARED_BACKBONE,
    SPARSE_EXPERT,
    ROUTER,
    TASK_HEAD,
    ALWAYS_ON_SHARED_EXPERT,
)

_EXPERT_RE = re.compile(r"^cross_layers\.(\d+)\.experts\.(\d+)\.")
_ROUTER_RE = re.compile(r"^cross_layers\.(\d+)\.router\.embed\.weight$")
_SHARED_EXPERT_RE = re.compile(r"^cross_layers\.(\d+)\.shared\.")
_EMBEDDING_RE = re.compile(r"^sparse\.tables\.([^.]+)\.weight$")


@dataclass(frozen=True)
class RoleHyperParameters:
    learning_rate: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    use_first_moment: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= self.betas[0] < 1 or not 0 <= self.betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    parameter: nn.Parameter
    role: str
    layer_index: int | None = None
    expert_index: int | None = None
    embedding_field: str | None = None
    row_sparse: bool = False


@dataclass
class TaskGradient:
    task_id: int
    gradients: dict[str, torch.Tensor | None]
    active_experts: dict[int, frozenset[int]]
    active_rows: dict[str, torch.Tensor]
    loss: float | None = None


def default_role_hyperparameters(
    expert_learning_rate: float = 5e-4,
    router_learning_rate_ratio: float = 0.05,
    shared_learning_rate_ratio: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> dict[str, RoleHyperParameters]:
    shared_lr = expert_learning_rate * shared_learning_rate_ratio
    return {
        SHARED_EMBEDDING: RoleHyperParameters(
            shared_lr, betas, eps, weight_decay, True),
        SHARED_BACKBONE: RoleHyperParameters(
            shared_lr, betas, eps, weight_decay, True),
        SPARSE_EXPERT: RoleHyperParameters(
            expert_learning_rate, betas, eps, weight_decay, True),
        ROUTER: RoleHyperParameters(
            expert_learning_rate * router_learning_rate_ratio,
            betas, eps, 0.0, False),
        TASK_HEAD: RoleHyperParameters(
            expert_learning_rate, betas, eps, 0.0, True),
        ALWAYS_ON_SHARED_EXPERT: RoleHyperParameters(
            shared_lr, betas, eps, weight_decay, True),
    }


class ParameterRoleRegistry:
    """建立互斥且完备的参数角色映射。"""

    def __init__(self, specs: Sequence[ParameterSpec]):
        self.specs = tuple(specs)
        self.by_name = {spec.name: spec for spec in self.specs}
        if len(self.by_name) != len(self.specs):
            raise ValueError("duplicate parameter names in role registry")

    @classmethod
    def from_model(cls, model: nn.Module) -> "ParameterRoleRegistry":
        specs: list[ParameterSpec] = []
        unknown: list[str] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            match = _EMBEDDING_RE.match(name)
            if match:
                specs.append(ParameterSpec(
                    name, parameter, SHARED_EMBEDDING,
                    embedding_field=match.group(1), row_sparse=True))
                continue
            match = _EXPERT_RE.match(name)
            if match:
                specs.append(ParameterSpec(
                    name, parameter, SPARSE_EXPERT,
                    layer_index=int(match.group(1)),
                    expert_index=int(match.group(2))))
                continue
            match = _ROUTER_RE.match(name)
            if match:
                specs.append(ParameterSpec(
                    name, parameter, ROUTER,
                    layer_index=int(match.group(1)), row_sparse=True))
                continue
            match = _SHARED_EXPERT_RE.match(name)
            if match:
                specs.append(ParameterSpec(
                    name, parameter, ALWAYS_ON_SHARED_EXPERT,
                    layer_index=int(match.group(1))))
                continue
            if name.startswith("dnn."):
                specs.append(ParameterSpec(name, parameter, SHARED_BACKBONE))
                continue
            if name.startswith("head."):
                specs.append(ParameterSpec(
                    name, parameter, TASK_HEAD, row_sparse=True))
                continue
            unknown.append(name)
        if unknown:
            raise ValueError(
                "parameters without a frozen role: " + ", ".join(unknown))
        registry = cls(specs)
        registry.assert_complete(model)
        return registry

    def assert_complete(self, model: nn.Module) -> None:
        expected = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        actual = set(self.by_name)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"role registry mismatch: missing={missing}, extra={extra}")
        parameter_ids = [id(spec.parameter) for spec in self.specs]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("a trainable parameter belongs to more than one role")

    def role_counts(self) -> dict[str, int]:
        counts = {role: 0 for role in PARAMETER_ROLES}
        for spec in self.specs:
            counts[spec.role] += spec.parameter.numel()
        return counts

    def metadata(self) -> list[dict[str, object]]:
        return [{
            "name": spec.name,
            "role": spec.role,
            "shape": list(spec.parameter.shape),
            "layer_index": spec.layer_index,
            "expert_index": spec.expert_index,
            "embedding_field": spec.embedding_field,
            "row_sparse": spec.row_sparse,
        } for spec in self.specs]


class SharedAdamWBatchOptimizer:
    """把场景平均梯度交给原生 AdamW，作为逐位等价内部对照。"""

    def __init__(
        self,
        registry: ParameterRoleRegistry,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.registry = registry
        self.optimizer = torch.optim.AdamW(
            [spec.parameter for spec in registry.specs],
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def step(self, task_gradients: Sequence[TaskGradient]) -> None:
        if not task_gradients:
            raise ValueError("task_gradients must not be empty")
        count = float(len(task_gradients))
        self.optimizer.zero_grad(set_to_none=True)
        for spec in self.registry.specs:
            present = [item.gradients.get(spec.name) for item in task_gradients]
            present = [gradient for gradient in present if gradient is not None]
            if not present:
                continue
            total = torch.zeros_like(spec.parameter)
            for gradient in present:
                total.add_(gradient)
            spec.parameter.grad = total.div_(count)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, object]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.optimizer.load_state_dict(state)


class TaskRoleOptimizer:
    """场景状态隔离、角色差异化、每批单次参数写入的优化器。"""

    FORMAT_VERSION = 1

    def __init__(
        self,
        registry: ParameterRoleRegistry,
        role_hyperparameters: Mapping[str, RoleHyperParameters],
    ):
        self.registry = registry
        missing = set(PARAMETER_ROLES) - set(role_hyperparameters)
        extra = set(role_hyperparameters) - set(PARAMETER_ROLES)
        if missing or extra:
            raise ValueError(f"role hyperparameters mismatch: missing={missing}, extra={extra}")
        self.role_hyperparameters = dict(role_hyperparameters)
        self._states: dict[str, dict[int, dict[str, torch.Tensor]]] = {
            spec.name: {} for spec in registry.specs
        }
        self.last_step_metrics: dict[str, object] = {}

    def _new_state(self, spec: ParameterSpec, task_id: int) -> dict[str, torch.Tensor]:
        del task_id
        parameter = spec.parameter
        if spec.row_sparse:
            step = torch.zeros(
                parameter.shape[0], device=parameter.device, dtype=torch.long)
        else:
            step = torch.zeros((), device=parameter.device, dtype=torch.long)
        state = {
            "step": step,
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        if self.role_hyperparameters[spec.role].use_first_moment:
            state["exp_avg"] = torch.zeros_like(parameter)
        return state

    def _state_for(self, spec: ParameterSpec, task_id: int) -> dict[str, torch.Tensor]:
        states = self._states[spec.name]
        if task_id not in states:
            states[task_id] = self._new_state(spec, task_id)
        return states[task_id]

    @staticmethod
    def _reshape_bias_correction(value: torch.Tensor, ndim: int) -> torch.Tensor:
        return value.reshape((value.shape[0],) + (1,) * (ndim - 1))

    def _dense_direction(
        self,
        spec: ParameterSpec,
        task_id: int,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        hyper = self.role_hyperparameters[spec.role]
        beta1, beta2 = hyper.betas
        state = self._state_for(spec, task_id)
        state["step"].add_(1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(
            gradient, gradient, value=1.0 - beta2)
        step = int(state["step"].item())
        second = state["exp_avg_sq"] / (1.0 - beta2 ** step)
        if hyper.use_first_moment:
            state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            numerator = state["exp_avg"] / (1.0 - beta1 ** step)
        else:
            numerator = gradient
        return numerator / (second.sqrt() + hyper.eps)

    def _row_direction(
        self,
        spec: ParameterSpec,
        task_id: int,
        gradient: torch.Tensor,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hyper = self.role_hyperparameters[spec.role]
        beta1, beta2 = hyper.betas
        state = self._state_for(spec, task_id)
        rows = torch.unique(rows.to(device=spec.parameter.device, dtype=torch.long))
        grad_rows = gradient.index_select(0, rows)
        steps = state["step"].index_select(0, rows).add_(1)
        state["step"].index_copy_(0, rows, steps)

        second = state["exp_avg_sq"].index_select(0, rows)
        second.mul_(beta2).addcmul_(grad_rows, grad_rows, value=1.0 - beta2)
        state["exp_avg_sq"].index_copy_(0, rows, second)
        correction2 = self._reshape_bias_correction(
            1.0 - beta2 ** steps.to(grad_rows.dtype), grad_rows.ndim)
        second_hat = second / correction2

        if hyper.use_first_moment:
            first = state["exp_avg"].index_select(0, rows)
            first.mul_(beta1).add_(grad_rows, alpha=1.0 - beta1)
            state["exp_avg"].index_copy_(0, rows, first)
            correction1 = self._reshape_bias_correction(
                1.0 - beta1 ** steps.to(grad_rows.dtype), grad_rows.ndim)
            numerator = first / correction1
        else:
            numerator = grad_rows
        return rows, numerator / (second_hat.sqrt() + hyper.eps)

    @staticmethod
    def _is_active(spec: ParameterSpec, item: TaskGradient) -> bool:
        if spec.role != SPARSE_EXPERT:
            return True
        selected = item.active_experts.get(spec.layer_index or 0, frozenset())
        return spec.expert_index in selected

    @staticmethod
    def _rows_for(spec: ParameterSpec, item: TaskGradient) -> torch.Tensor | None:
        if not spec.row_sparse:
            return None
        return item.active_rows.get(spec.name)

    @torch.no_grad()
    def step(self, task_gradients: Sequence[TaskGradient]) -> dict[str, object]:
        if not task_gradients:
            raise ValueError("task_gradients must not be empty")
        task_ids = [item.task_id for item in task_gradients]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"duplicate task ids in one batch: {task_ids}")
        task_count = float(len(task_gradients))
        role_update_sq = {role: 0.0 for role in PARAMETER_ROLES}
        parameter_writes: dict[str, int] = {}
        active_rows_metric: dict[str, int] = {}

        for spec in self.registry.specs:
            hyper = self.role_hyperparameters[spec.role]
            accumulated = torch.zeros_like(spec.parameter)
            full_active = False
            row_sets: list[torch.Tensor] = []
            contributed = False
            for item in task_gradients:
                gradient = item.gradients.get(spec.name)
                if gradient is None or not self._is_active(spec, item):
                    continue
                if gradient.shape != spec.parameter.shape:
                    raise ValueError(
                        f"gradient shape mismatch for {spec.name}: "
                        f"{tuple(gradient.shape)} != {tuple(spec.parameter.shape)}")
                rows = self._rows_for(spec, item)
                if spec.row_sparse:
                    if rows is None or rows.numel() == 0:
                        continue
                    rows, direction = self._row_direction(
                        spec, item.task_id, gradient, rows)
                    accumulated.index_add_(0, rows, direction / task_count)
                    row_sets.append(rows)
                else:
                    direction = self._dense_direction(
                        spec, item.task_id, gradient)
                    accumulated.add_(direction, alpha=1.0 / task_count)
                    full_active = True
                contributed = True
            if not contributed:
                continue

            if full_active:
                old_value = spec.parameter.detach().clone()
                new_value = spec.parameter - accumulated * hyper.learning_rate
                if hyper.weight_decay and hyper.learning_rate:
                    new_value.add_(
                        spec.parameter,
                        alpha=-hyper.learning_rate * hyper.weight_decay,
                    )
                applied = old_value - new_value
                spec.parameter.copy_(new_value)
            else:
                active_rows = torch.unique(torch.cat(row_sets))
                active_rows_metric[spec.name] = int(active_rows.numel())
                old_value = spec.parameter.index_select(0, active_rows)
                direction = accumulated.index_select(0, active_rows)
                new_value = old_value - direction * hyper.learning_rate
                if hyper.weight_decay and hyper.learning_rate:
                    new_value.add_(
                        old_value,
                        alpha=-hyper.learning_rate * hyper.weight_decay,
                    )
                applied = old_value - new_value
                spec.parameter.index_copy_(0, active_rows, new_value)
            parameter_writes[spec.name] = 1
            role_update_sq[spec.role] += float(applied.float().pow(2).sum())

        self.last_step_metrics = {
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "parameter_writes": parameter_writes,
            "active_rows": active_rows_metric,
            "role_update_norm": {
                role: value ** 0.5 for role, value in role_update_sq.items()
            },
        }
        return copy.deepcopy(self.last_step_metrics)

    def state_dict(self) -> dict[str, object]:
        states: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        for name, by_task in self._states.items():
            states[name] = {
                task_id: {key: value.detach().clone().cpu()
                          for key, value in state.items()}
                for task_id, state in by_task.items()
            }
        return {
            "format_version": self.FORMAT_VERSION,
            "registry": self.registry.metadata(),
            "role_hyperparameters": {
                role: {
                    "learning_rate": hyper.learning_rate,
                    "betas": hyper.betas,
                    "eps": hyper.eps,
                    "weight_decay": hyper.weight_decay,
                    "use_first_moment": hyper.use_first_moment,
                }
                for role, hyper in self.role_hyperparameters.items()
            },
            "states": states,
            "last_step_metrics": copy.deepcopy(self.last_step_metrics),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(
                f"unsupported optimizer format {payload.get('format_version')}")
        if payload.get("registry") != self.registry.metadata():
            raise ValueError("optimizer checkpoint parameter roles do not match model")
        saved_hyper = payload.get("role_hyperparameters")
        current_hyper = self.state_dict()["role_hyperparameters"]
        if saved_hyper != current_hyper:
            raise ValueError("optimizer checkpoint hyperparameters do not match")
        restored = payload.get("states")
        if not isinstance(restored, Mapping):
            raise ValueError("optimizer checkpoint has no states mapping")
        self._states = {spec.name: {} for spec in self.registry.specs}
        for spec in self.registry.specs:
            by_task = restored.get(spec.name, {})
            if not isinstance(by_task, Mapping):
                raise ValueError(f"invalid state mapping for {spec.name}")
            for raw_task_id, raw_state in by_task.items():
                task_id = int(raw_task_id)
                if not isinstance(raw_state, Mapping):
                    raise ValueError(f"invalid task state for {spec.name}/{task_id}")
                state: dict[str, torch.Tensor] = {}
                for key, value in raw_state.items():
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(
                            f"non-tensor state {spec.name}/{task_id}/{key}")
                    dtype = torch.long if key == "step" else spec.parameter.dtype
                    state[str(key)] = value.to(
                        device=spec.parameter.device, dtype=dtype).clone()
                self._states[spec.name][task_id] = state
        self.last_step_metrics = copy.deepcopy(
            payload.get("last_step_metrics", {}))

    def task_state(self, parameter_name: str, task_id: int) -> dict[str, torch.Tensor] | None:
        state = self._states[parameter_name].get(int(task_id))
        if state is None:
            return None
        return {key: value.detach().clone() for key, value in state.items()}


def _active_experts_for_task(
    gates: Sequence[torch.Tensor],
    task_mask: torch.Tensor,
) -> dict[int, frozenset[int]]:
    active: dict[int, frozenset[int]] = {}
    for layer_index, gate in enumerate(gates):
        selected = torch.nonzero(
            gate[task_mask].detach().abs().sum(dim=0) > 0,
            as_tuple=False,
        ).flatten()
        active[layer_index] = frozenset(int(index) for index in selected.tolist())
    return active


def _active_rows_for_task(
    registry: ParameterRoleRegistry,
    batch: Mapping[str, torch.Tensor],
    task_id: int,
    task_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {}
    for spec in registry.specs:
        if not spec.row_sparse:
            continue
        if spec.role == SHARED_EMBEDDING:
            if spec.embedding_field not in batch:
                raise KeyError(
                    f"batch has no embedding field {spec.embedding_field!r}")
            rows[spec.name] = torch.unique(batch[spec.embedding_field][task_mask].long())
        elif spec.role in (ROUTER, TASK_HEAD):
            rows[spec.name] = torch.tensor(
                [task_id], device=spec.parameter.device, dtype=torch.long)
    return rows


def collect_task_gradients(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    registry: ParameterRoleRegistry,
) -> list[TaskGradient]:
    """一次前向后提取各场景梯度，不修改参数。"""
    model(batch)
    if "logit" not in batch or "_gate" not in batch:
        raise ValueError("model must write batch['logit'] and batch['_gate']")
    labels = batch["is_click"].to(batch["logit"].dtype)
    per_row = F.binary_cross_entropy_with_logits(
        batch["logit"], labels, reduction="none")
    tabs = batch["tab"].long()
    task_ids = [int(value) for value in torch.unique(tabs, sorted=True).tolist()]
    parameters = [spec.parameter for spec in registry.specs]
    results: list[TaskGradient] = []
    for index, task_id in enumerate(task_ids):
        mask = tabs == task_id
        loss = per_row[mask].mean()
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index + 1 < len(task_ids),
            allow_unused=True,
        )
        results.append(TaskGradient(
            task_id=task_id,
            gradients={
                spec.name: None if gradient is None else gradient.detach()
                for spec, gradient in zip(registry.specs, gradients)
            },
            active_experts=_active_experts_for_task(batch["_gate"], mask),
            active_rows=_active_rows_for_task(
                registry, batch, task_id, mask),
            loss=float(loss.detach()),
        ))
    return results


def capture_random_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: object, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
