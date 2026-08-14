import torch
import torch.nn.functional as F
from torch import nn

import fields


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = nn.ModuleDict()
        for field, size in (fields.user | fields.video).items():
            self.tables[field] = nn.Embedding(size, 10)

    def forward(self, batch):
        embeddings = [table(batch[field]) for field, table in self.tables.items()]
        return torch.cat(embeddings, -1)


class DCNv2(nn.Module):
    def __init__(self, dim=360):
        super().__init__()
        self.sparse = Sparse()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(5)])
        self.layers += [nn.Linear(dim, 15)]

    def embed(self, batch):
        return self.sparse(batch)

    def forward(self, batch):
        CRs = [self.embed(batch)]
        for i, layer in enumerate(self.layers):
            if i < 3:
                CRs += [layer(CRs[-1]) * CRs[0] + CRs[-1]]
            else:
                CRs += [layer(CRs[-1].relu())]
        batch["logit"] = CRs[-1][range(len(CRs[-1])), batch["tab"]]
        if hasattr(self, "CRs"):
            for id, CRs in zip(batch["user_id"], torch.stack(CRs[:-1], 1)):
                self.CRs[id] *= 0.99
                self.CRs[id] += CRs


# ============================================================
#  Zero-Parameter MoE: CrossExpertLayer + DCNv2MoE
#  Splits Linear(dim,dim) → K × Linear(dim, dim//K)  (exact param count)
#  Router: Embedding(8,K) — 32 params, practically zero
# ============================================================

class ScenarioRouter(nn.Module):
    """Scenario-aware router: Embedding(15, K) — near-zero params.

    At init (weight=0, softmax uniform), all experts receive equal weight.
    """
    def __init__(self, K=4, num_scenarios=15):
        super().__init__()
        self.embed = nn.Embedding(num_scenarios, K)
        nn.init.zeros_(self.embed.weight)

    def forward(self, tab):
        return self.embed(tab).softmax(-1) * self.embed.weight.shape[1]


class DataRouter(nn.Module):
    """Data-driven router: Linear(dim, K) — selects experts from input features.

    uniform init(weight=0, bias=0) → softmax 恒为 1/K。
    gate_scaling 用于统一「恒等初始化」语义：gate_scaling=K 时，uniform init
    输出全 1，与 ScenarioRouter(softmax*K) 一致。默认 1.0 —— 不影响既有训练。
    """
    def __init__(self, dim=360, K=4, gate_scaling=1.0):
        super().__init__()
        self.linear = nn.Linear(dim, K)
        self.gate_scaling = gate_scaling
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).softmax(-1) * self.gate_scaling  # [B, K]


class CrossExpertLayer(nn.Module):
    """Zero-parameter MoE Cross layer.

    Splits a Linear(dim, dim) into K Linear(dim, dim//K) experts.
    Total params = K*(dim*(dim//K) + dim//K) = dim² + dim (preserved).

    Two routing modes:
      - 'scenario': ScenarioRouter — Embedding(15, K), ~0 params
      - 'data':     DataRouter    — Linear(dim, K), selects from feature

    At init (router=0, softmax uniform), output ≡ vanilla Cross layer.
    Formula: x_{l+1} = concat([g_i · W_i(x_l)]) ⊙ x_0 + x_l
    """

    def __init__(self, dim=360, K=4, routing='scenario'):
        super().__init__()
        assert dim % K == 0, f"dim {dim} must be divisible by K={K}"
        self.dim, self.K, self.routing = dim, K, routing
        expert_dim = dim // K
        self.experts = nn.ModuleList([nn.Linear(dim, expert_dim) for _ in range(K)])

        if routing == 'scenario':
            self.router = ScenarioRouter(K)
        elif routing == 'data':
            self.router = DataRouter(dim, K)
        else:
            raise ValueError(f"Unknown routing mode: {routing}")

    def load_pretrained(self, pretrained_linear: nn.Linear):
        """Split pretrained Linear(dim,dim) weight into K experts along output dim.
        W ∈ R^{dim×dim} → W_0:0..c, W_1:c..2c, ..., W_{K-1}:(K-1)c..Kc
        where c = dim // K.
        """
        w, b = pretrained_linear.weight.data, pretrained_linear.bias.data
        chunk = self.dim // self.K
        for i, expert in enumerate(self.experts):
            expert.weight.data.copy_(w[i * chunk:(i + 1) * chunk])
            expert.bias.data.copy_(b[i * chunk:(i + 1) * chunk])

    def forward(self, x, x0, tab=None):
        # x: [B, dim], x0: [B, dim], tab: [B] optional scenario ids
        if self.routing == 'scenario':
            assert tab is not None, "Scenario routing requires tab input"
            gate = self.router(tab)  # [B, K], avg=1 → init=uniform
        else:
            gate = self.router(x)    # [B, K], data-driven

        expert_outs = [e(x) for e in self.experts]  # K × [B, dim//K]
        weighted = torch.cat(
            [gate[:, i:i + 1] * expert_outs[i] for i in range(self.K)], dim=-1
        )  # [B, dim]
        return weighted * x0 + x, gate


class DCNv2MoE(nn.Module):
    """DCNv2 with zero-parameter MoE: 3 MoE Cross layers + 2 DNN + head."""

    def __init__(self, dim=360, K=4, routing='scenario'):
        super().__init__()
        self.dim, self.K, self.routing = dim, K, routing
        self.sparse = Sparse()
        self.cross_layers = nn.ModuleList(
            [CrossExpertLayer(dim, K, routing=routing) for _ in range(3)]
        )
        self.dnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.head = nn.Linear(dim, 15)

    def embed(self, batch):
        return self.sparse(batch)

    def load_pretrained(self, pretrained: DCNv2):
        """Initialize from pretrained DCNv2: split Cross weights + copy Sparse/DNN/Head."""
        self.sparse.load_state_dict(pretrained.sparse.state_dict())
        for i, cross_layer in enumerate(self.cross_layers):
            cross_layer.load_pretrained(pretrained.layers[i])
        self.dnn[0].load_state_dict(pretrained.layers[3].state_dict())
        self.dnn[1].load_state_dict(pretrained.layers[4].state_dict())
        self.head.load_state_dict(pretrained.layers[5].state_dict())

    def forward(self, batch):
        x0 = self.embed(batch)
        x = x0
        gates = []
        CRs = [x0.detach()]
        for layer in self.cross_layers:
            tab = batch["tab"] if self.routing == 'scenario' else None
            x, gate = layer(x, x0, tab)
            gates.append(gate)
            CRs.append(x.detach())
        for layer in self.dnn:
            x = layer(x.relu())
            CRs.append(x.detach())
        batch["logit"] = self.head(x)[range(len(x)), batch["tab"]]
        batch["_gate"] = gates
        if hasattr(self, "CRs"):
            for uid, c in zip(batch["user_id"], torch.stack(CRs, 1)):
                self.CRs[uid] *= 0.99
                self.CRs[uid] += c


# ============================================================
#  GradientTracker: per-expert per-scenario AU tracking (AdaTask-style)
# ============================================================

class GradientTracker:
    """Accumulates per-(layer, expert, scenario) squared gradient EMA (AU).

    Uses backward hooks on expert parameters — zero forward overhead.
    AU_{s,l,e} ← β · AU_{s,l,e} + (1-β) · ||∇_{θ_{l,e}} L_s||²
    """

    def __init__(self, model: DCNv2MoE, beta=0.99):
        self.model = model
        self.beta = beta
        self.AU = {}  # key: (layer_idx, expert_idx, scenario:int) → float (AU value)
        self._hooks = []
        self._current_scenario = None

    def set_scenario(self, scenario: int):
        """Set the current scenario for this backward pass."""
        self._current_scenario = scenario

    def _make_hook(self, layer_idx: int, expert_idx: int):
        def hook(grad: torch.Tensor):
            if self._current_scenario is None:
                return
            key = (layer_idx, expert_idx, int(self._current_scenario))
            gsq = grad.detach().norm().item() ** 2
            prev = self.AU.get(key, 0.0)
            self.AU[key] = self.beta * prev + (1 - self.beta) * gsq
        return hook

    def register(self):
        """Register backward hooks on all CrossExpertLayer expert parameters."""
        for li, layer in enumerate(self.model.cross_layers):
            for ei, expert in enumerate(layer.experts):
                for name, param in expert.named_parameters():
                    h = param.register_hook(self._make_hook(li, ei))
                    self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    TARGET_SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]

    def dominance_matrix(self, layer_idx: int = 0) -> dict:
        """Return dominance ratios for a given MoE layer on target scenarios.

        Returns: dict mapping (expert_idx, scenario) → ratio ∈ [0, 1]
        where ratio = AU[scenario] / Σ_{s∈TARGET} AU[s] for that expert.
        Non-target scenarios are excluded from the normalization.
        """
        ratios = {}
        for ei in range(self.model.K):
            au_per_scenario = {s: self.AU.get((layer_idx, ei, s), 0.0)
                               for s in self.TARGET_SCENARIOS}
            total = sum(au_per_scenario.values()) + 1e-30
            for s, au in au_per_scenario.items():
                ratios[(ei, s)] = au / total
        return ratios

    def summary(self) -> str:
        """Human-readable dominance summary for all MoE layers."""
        lines = []
        for li in range(3):
            ratios = self.dominance_matrix(li)
            lines.append(f"--- Cross Layer {li} Dominance ---")
            for ei in range(self.model.K):
                row = [f"{ratios.get((ei, s), 0):.3f}" for s in self.TARGET_SCENARIOS]
                lines.append(f"  E{ei}: " + " ".join(row))
        return "\n".join(lines)


# ============================================================
#  AdaTaskOptimizer: AU-based LR modulation for expert specialization control
# ============================================================

class AdaTaskOptimizer:
    """子任务自适应学习率调制器（交叉多维版）。

    在按场景切分的子批量上，用梯度平方累计（AU）驱动「逐目标」的梯度缩放，
    用于观察「路由网络 / 路由专家 / 共享专家」被「促进 / 抑制」时对上游任务的影响。

    三个调制目标是**相乘的独立维度**，每个可独立设为：
      - 'none'      不调制（仅做 AU 跟踪，等同正常训练该目标）
      - 'encourage' 促进（factor = (au / mean_au)^alpha）
      - 'suppress'  抑制（factor = (mean_au / au)^alpha）

    参考均值（mean_au）取法：
      - 路由网络：该层路由器跨全部场景的 AU 均值。
      - 路由专家：同层同场景下 K 个路由专家 AU 的均值。
      - 共享专家：同层同场景下 K 个路由专家 AU 的均值（即「共享专家相对专家平均」的对比）。

    AU 跟踪通过反向钩子完成；三类目标的钩子同时注册，调制在每步同时施加。

    向后兼容：旧接口 mode=encourage/suppress/none（仅作用于专家）与
    target=.../direction=... 仍可用，会被映射到对应的 *_mode。
    """

    TARGET_SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]

    def __init__(self, model, lr=1e-3,
                 router_mode="none", expert_mode="none", shared_mode="none",
                 alpha=0.5, beta=0.99, weight_decay=0.01,
                 # 向后兼容旧接口
                 target="expert", direction=0, mode=None):
        # ---- 解析调制模式（优先新接口，旧接口兼容）----
        if mode is not None:
            # 旧接口：仅调制专家
            expert_mode = mode
            router_mode = "none"
            shared_mode = "none"
        else:
            if target == "router" and direction != 0:
                router_mode = "encourage" if direction > 0 else "suppress"
            elif target == "expert" and direction != 0:
                expert_mode = "encourage" if direction > 0 else "suppress"
        self.model = model
        self.router_mode = router_mode
        self.expert_mode = expert_mode
        self.shared_mode = shared_mode
        self.alpha = alpha
        self.beta = beta
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay)
        # AU 跟踪：路由专家 (层, 专家序号, 场景)；路由网络 (层, 'r', 场景)；
        #          共享专家 (层, 's', 场景)
        self.AU: dict = {}
        self._hooks = []
        self._current_scenario: int | None = None

    def set_scenario(self, scenario: int):
        self._current_scenario = scenario

    def _make_hook(self, key):
        def hook(grad: torch.Tensor):
            if self._current_scenario is None:
                return
            k = (key[0], key[1], int(self._current_scenario))
            gsq = grad.detach().norm().item() ** 2
            self.AU[k] = self.beta * self.AU.get(k, 0.0) + (1 - self.beta) * gsq
        return hook

    @staticmethod
    def _router_params(layer):
        """返回该交叉层的路由器参数（兼容 V1 与 V2）。"""
        if hasattr(layer, "w_gate"):
            return list(layer.w_gate.parameters()) + list(layer.w_noise.parameters())
        if hasattr(layer, "router"):
            return list(layer.router.parameters())
        return []

    @staticmethod
    def _shared_params(layer):
        """返回该交叉层的共享专家参数（仅 V2 有；V1 返回空）。"""
        if hasattr(layer, "shared"):
            return list(layer.shared.parameters())
        return []

    def register_hooks(self):
        """同时注册 路由专家 / 路由网络 / 共享专家 三类钩子。"""
        for li, layer in enumerate(self.model.cross_layers):
            for ei, expert in enumerate(layer.experts):
                for param in expert.parameters():
                    self._hooks.append(
                        param.register_hook(self._make_hook((li, ei))))
            for param in self._router_params(layer):
                self._hooks.append(
                    param.register_hook(self._make_hook((li, "r"))))
            for param in self._shared_params(layer):
                self._hooks.append(
                    param.register_hook(self._make_hook((li, "s"))))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _expert_factors(self, s):
        factors = {}
        for li, layer in enumerate(self.model.cross_layers):
            aus = [self.AU.get((li, ei, s), 0.0) for ei in range(self.model.K)]
            mean_au = sum(aus) / len(aus) + 1e-30
            for ei in range(self.model.K):
                au = aus[ei] + 1e-30
                if self.expert_mode == "encourage":
                    f = (au / mean_au) ** self.alpha
                elif self.expert_mode == "suppress":
                    f = (mean_au / au) ** self.alpha
                else:
                    f = 1.0
                factors[(li, ei)] = float(f)
        return factors

    def _router_factors(self, s):
        factors = {}
        for li, layer in enumerate(self.model.cross_layers):
            aus = [self.AU.get((li, "r", sc), 0.0) for sc in self.TARGET_SCENARIOS]
            mean_au = sum(aus) / len(aus) + 1e-30
            au = self.AU.get((li, "r", s), 0.0) + 1e-30
            if self.router_mode == "encourage":
                f = (au / mean_au) ** self.alpha
            elif self.router_mode == "suppress":
                f = (mean_au / au) ** self.alpha
            else:
                f = 1.0
            factors[(li, "r")] = float(f)
        return factors

    def _shared_factors(self, s):
        factors = {}
        for li, layer in enumerate(self.model.cross_layers):
            if not self._shared_params(layer):
                continue
            routed = [self.AU.get((li, ei, s), 0.0) for ei in range(self.model.K)]
            mean_routed = sum(routed) / len(routed) + 1e-30
            au = self.AU.get((li, "s", s), 0.0) + 1e-30
            if self.shared_mode == "encourage":
                f = (au / mean_routed) ** self.alpha
            elif self.shared_mode == "suppress":
                f = (mean_routed / au) ** self.alpha
            else:
                f = 1.0
            factors[(li, "s")] = float(f)
        return factors

    def modulate_and_zero_grad(self):
        """同时施加三个目标的梯度调制 + optimizer.step() + zero_grad。

        某目标 mode='none' 时该目标因子恒为 1，等价不调制。
        """
        s = self._current_scenario
        if s is None:
            self.optimizer.step()
            self.optimizer.zero_grad()
            return
        ef = self._expert_factors(s)
        rf = self._router_factors(s)
        sf = self._shared_factors(s)
        for li, layer in enumerate(self.model.cross_layers):
            for ei, expert in enumerate(layer.experts):
                f = ef.get((li, ei), 1.0)
                for param in expert.parameters():
                    if param.grad is not None:
                        param.grad.mul_(f)
            f = rf.get((li, "r"), 1.0)
            for param in self._router_params(layer):
                if param.grad is not None:
                    param.grad.mul_(f)
            f = sf.get((li, "s"), 1.0)
            for param in self._shared_params(layer):
                if param.grad is not None:
                    param.grad.mul_(f)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def dominance_matrix(self, layer_idx: int = 0) -> dict:
        """专家主导性矩阵（路由专家目标）。"""
        ratios = {}
        for ei in range(self.model.K):
            au_per_scenario = {s: self.AU.get((layer_idx, ei, s), 0.0)
                               for s in self.TARGET_SCENARIOS}
            total = sum(au_per_scenario.values()) + 1e-30
            for s, au in au_per_scenario.items():
                ratios[(ei, s)] = au / total
        return ratios

    def summary(self) -> str:
        lines = [f"[router={self.router_mode} expert={self.expert_mode} "
                 f"shared={self.shared_mode}]"]
        for li in range(3):
            ratios = self.dominance_matrix(li)
            lines.append(f"--- Cross Layer {li} Expert Dominance ---")
            for ei in range(self.model.K):
                row = [f"{ratios.get((ei, s), 0):.3f}"
                       for s in self.TARGET_SCENARIOS[:4]]
                row += ["..."]
                row += [f"{ratios.get((ei, s), 0):.3f}"
                        for s in self.TARGET_SCENARIOS[-3:]]
                lines.append(f"  E{ei}: " + " ".join(row))
        return "\n".join(lines)

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "AU": {str(k): v for k, v in self.AU.items()},
        }

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd["optimizer"])
        self.AU = {eval(k): v for k, v in sd["AU"].items()}


# ============================================================
#  SpecializationLoss: encourage expert specialization via MI maximization
# ============================================================

class SpecializationLoss:
    """Encourages scenario-expert specialization by maximizing mutual information.

    L_spec = -λ · Σ_s Σ_e 1{dominance > threshold} · log(avg_gate(e|s))

    Where avg_gate(e|s) is the empirical probability of scenario s routing
    to expert e (averaged over batches).
    Only activates when gradient dominance is already observed (ratio > threshold).
    """

    def __init__(self, threshold=0.3, lmbda=0.01, ema_beta=0.99):
        self.threshold = threshold
        self.lmbda = lmbda
        self.ema_beta = ema_beta
        # EMA of avg gate per (layer, expert, scenario): gate_ema[l][e][s]
        self.gate_ema = {}  # (layer_idx, expert_idx, scenario) → float
        self.enabled = False  # turn on after dominance detected

    def update(self, gates: list, tab: torch.Tensor):
        """Update EMA of routing probabilities per scenario.

        Args:
            gates: list of [B, K] tensors from each CrossExpertLayer
            tab: [B] scenario ids
        """
        for li, gate in enumerate(gates):  # gate: [B, K]
            for s in tab.unique():
                mask = tab == s
                if mask.sum() == 0:
                    continue
                avg_gate = gate[mask].mean(0)  # [K]
                for ei in range(gate.shape[1]):
                    key = (li, ei, int(s.item()))
                    prev = self.gate_ema.get(key, 0.0)
                    self.gate_ema[key] = (
                        self.ema_beta * prev + (1 - self.ema_beta) * avg_gate[ei].item()
                    )

    def compute(self, gates: list, tab: torch.Tensor, ratios: dict = None) -> torch.Tensor:
        """Compute specialization loss for current batch.

        Args:
            gates: list of [B, K] gate tensors
            tab: [B] scenario ids
            ratios: optional pre-computed dominance ratios dict
        Returns:
            scalar loss (0 if not enabled)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=gates[0].device)

        loss = torch.tensor(0.0, device=gates[0].device)
        for li, gate in enumerate(gates):
            for s in tab.unique():
                mask = tab == s
                si = int(s.item())
                if mask.sum() == 0:
                    continue
                avg_prob = gate[mask].mean(0)  # [K]
                for ei in range(gate.shape[1]):
                    key = (li, ei, si)
                    dom = ratios.get(key, 0.0) if ratios else self.gate_ema.get(key, 0.0)
                    if dom > self.threshold:
                        loss = loss - self.lmbda * (avg_prob[ei] + 1e-8).log()
        return loss

    def check_and_enable(self, tracker: GradientTracker, layer_idx: int = None):
        """Enable specialization loss if any expert shows clear scenario dominance.

        Fix (D1-B): scan ALL three MoE layers by default, not just layer 0,
        since dominance may appear in any cross layer.
        """
        layer_indices = [layer_idx] if layer_idx is not None else range(3)
        for li in layer_indices:
            ratios = tracker.dominance_matrix(li)
            for (ei, s), ratio in ratios.items():
                if ratio > self.threshold:
                    self.enabled = True
                    return True
        return False


# ============================================================
#  Parameter grouping & selective freezing
#
#  用于两类实验:
#   1. router+experts-only 消融 (冻结 dnn/head/sparse, 只训 MoE 本体)
#   2. 梯度主导度测量 (按组统计每参数 RMS 梯度)
#
#  组定义对 DCNv2 / DCNv2MoE(V1) / DCNv2MoE_V2 三种架构通用。
# ============================================================

#: 组名 → 判定函数（输入 named_parameters 的 name）
PARAM_GROUPS: dict = {
    # 类别特征 embedding 表 —— 占总参数 ~99.2%, 与 MoE 无关
    "Sparse":       lambda n: n.startswith("sparse."),
    # V2 shared expert（始终激活）
    "CrossShared":  lambda n: "cross_layers." in n and ".shared." in n,
    # V1 的 K 个 expert / V2 的 K 个 routed expert
    "CrossExpert":  lambda n: "cross_layers." in n and ".experts." in n,
    # 路由器: V1 ScenarioRouter/DataRouter (.router.) ; V2 w_gate/w_noise
    "Router":       lambda n: "cross_layers." in n
                              and (".router." in n or ".w_gate" in n or ".w_noise" in n),
    # vanilla DCNv2 的 cross 层 (layers.0-2)，MoE 架构下不存在
    "CrossVanilla": lambda n: n.startswith("layers.")
                              and n.split(".")[1].isdigit() and int(n.split(".")[1]) < 3,
    "DNN":          lambda n: n.startswith("dnn.")
                              or (n.startswith("layers.") and n.split(".")[1].isdigit()
                                  and 3 <= int(n.split(".")[1]) <= 4),
    "Head":         lambda n: n.startswith("head.")
                              or (n.startswith("layers.") and n.split(".")[1].isdigit()
                                  and int(n.split(".")[1]) == 5),
}

#: `--freeze` 可接受的别名 → 对应的组集合
FREEZE_ALIASES: dict = {
    "sparse":  {"Sparse"},
    "dnn":     {"DNN"},
    "head":    {"Head"},
    "cross":   {"CrossShared", "CrossExpert", "CrossVanilla"},
    "experts": {"CrossExpert"},
    "shared":  {"CrossShared"},
    "router":  {"Router"},
}


def param_group_of(name: str) -> str:
    """Return the group name for a parameter, or 'Other' if unmatched."""
    for group, pred in PARAM_GROUPS.items():
        if pred(name):
            return group
    return "Other"


def group_param_counts(model: nn.Module) -> dict:
    """{group: n_params} for every group present in the model."""
    counts: dict = {}
    for n, p in model.named_parameters():
        counts[param_group_of(n)] = counts.get(param_group_of(n), 0) + p.numel()
    return counts


def apply_freeze(model: nn.Module, freeze: str) -> dict:
    """Freeze parameter groups in-place, e.g. freeze='dnn,head,sparse'.

    对 router+experts-only 消融: `apply_freeze(model, 'dnn,head,sparse')`
    ⇒ 仅 CrossShared + CrossExpert + Router 保持 requires_grad=True。

    Args:
        model:  DCNv2 / DCNv2MoE / DCNv2MoE_V2
        freeze: 逗号分隔的别名（见 FREEZE_ALIASES），空串/None 表示不冻结
    Returns:
        {'frozen_groups', 'trainable': {group: n}, 'frozen': {group: n},
         'n_trainable', 'n_frozen'} —— 供日志核对
    """
    targets: set = set()
    for token in (freeze or "").split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in FREEZE_ALIASES:
            raise ValueError(
                f"Unknown freeze target '{token}'. "
                f"Valid: {sorted(FREEZE_ALIASES)}"
            )
        targets |= FREEZE_ALIASES[token]

    trainable, frozen = {}, {}
    for n, p in model.named_parameters():
        g = param_group_of(n)
        if g in targets:
            p.requires_grad_(False)
            frozen[g] = frozen.get(g, 0) + p.numel()
        else:
            trainable[g] = trainable.get(g, 0) + p.numel()

    return {
        "frozen_groups": sorted(targets),
        "trainable": trainable,
        "frozen": frozen,
        "n_trainable": sum(trainable.values()),
        "n_frozen": sum(frozen.values()),
    }


def freeze_summary(info: dict) -> str:
    """Human-readable summary of apply_freeze() output."""
    total = info["n_trainable"] + info["n_frozen"]
    lines = [f"Freeze summary (frozen groups: {info['frozen_groups'] or 'none'}):"]
    for tag in ("trainable", "frozen"):
        if not info[tag]:
            continue
        lines.append(f"  [{tag}]")
        for g, n in sorted(info[tag].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {g:<14s}: {n:>12,}  ({n / total * 100:6.3f}% of all)")
    lines.append(
        f"  TOTAL trainable: {info['n_trainable']:,} / {total:,} "
        f"({info['n_trainable'] / total * 100:.3f}%)"
    )
    return "\n".join(lines)


def trainable_parameters(model: nn.Module):
    """Iterator over parameters with requires_grad=True (for the optimizer)."""
    return (p for p in model.parameters() if p.requires_grad)


class FeatureUsage(DCNv2):
    def __init__(self, LFM4Ads, method):
        super().__init__(360 if "gate" in method else 370)
        self.LFM4Ads = LFM4Ads
        self.method = method
        self.linear = nn.Linear(
            360 if "CR" in method else 270,
            360 if "gate" in method else 10,
        )

    def embed(self, batch):
        E = self.sparse(batch)
        if "CR" in self.method:
            UR = self.LFM4Ads.CRs[batch["user_id"], int(self.method[-1])]
        else:
            UR = self.LFM4Ads.sparse(batch)[:, :270]
        UR = self.linear(UR)
        if "gate" in self.method:
            return E * UR.sigmoid()
        else:
            return torch.cat([E, UR], -1)


class ModuleUsage(DCNv2):
    def __init__(self, LFM4Ads, method):
        super().__init__()
        if method != "Vanilla":
            self.sparse.load_state_dict(LFM4Ads.sparse.state_dict())
            for i in range(int(method[-1])):
                self.layers[i].load_state_dict(LFM4Ads.layers[i].state_dict())


class ModelUsage(nn.Module):
    def __init__(self, LFM4Ads, method):
        super().__init__()
        self.LFM4Ads = LFM4Ads
        self.method = method
        self.sparse = Sparse()
        self.linear = nn.Linear(360 if "CR" in method else 270, 90)

    def forward(self, batch):
        if self.method == "Retriever":
            E = self.sparse(batch)
        else:
            E = self.LFM4Ads.sparse(batch)
        if "CR" in self.method:
            UR = self.LFM4Ads.CRs[batch["user_id"], int(self.method[-1])]
        else:
            UR = E[:, :270]
        UR = self.linear(UR)
        IR = E[:, -90:]
        batch["logit"] = (UR * IR).sum(-1)


# ============================================================
#  CrossExpertLayerV2 — Advanced MoE (Qwen/DeepSeek–style)
#
#  核心创新:
#   1. Shared Expert: dim → dim_shared, 始终激活 (捕获 common knowledge)
#   2. Routed Experts: K×dim→dim_routed, top-k 稀疏激活 (捕获 specialized knowledge)
#   3. Zero-parameter: dim_shared + K·dim_routed = dim (param count ≡ vanilla)
#   4. Noisy Top-K Gating: 训练时加噪声, 推理时纯 top-k
#   5. Load Balancing Loss: Switch Transformer 风格, 防止专家坍塌
#   6. Warmup → Sparsify: 从 top_k=K (soft routing) 逐渐过渡到 top_k=2
# ============================================================

class CrossExpertLayerV2(nn.Module):
    """Advanced MoE Cross layer with shared expert + noisy top-k routing.

    参数量: dim_shared + K·dim_routed = dim, 与 vanilla Cross layer 严格相等。
    计算量: dim×dim_shared + top_k×dim×dim_routed < dim×dim (inference 时更少).

    Args:
        dim:        Cross layer dimension (360)
        K:          number of routed experts (4)
        top_k:      number of active routed experts (2 at inference, K at warmup)
        routing:    'scenario' (Embedding router) or 'data' (Linear router)
        noise_scale:std of gating noise during training
        lb_alpha:   load balancing loss coefficient
    """

    def __init__(self, dim=360, K=4, top_k=None, routing='data',
                 noise_scale=0.1, lb_alpha=0.01):
        super().__init__()
        self.dim, self.K, self.routing = dim, K, routing
        self.top_k = top_k if top_k is not None else K  # K=soft routing default
        self.noise_scale = noise_scale
        self._router_noise_enabled = True  # 冻结路由器时由 freeze_routers 置 False
        self.lb_alpha = lb_alpha

        # Zero-parameter split: shared takes 1/(K+1), each routed takes 1/(K+1)
        dim_shared = dim // (K + 1)
        dim_routed = dim // (K + 1)
        assert dim_shared + K * dim_routed == dim, \
            f"dim mismatch: {dim_shared}+{K}×{dim_routed}={dim_shared+K*dim_routed}≠{dim}"

        self.dim_shared = dim_shared
        self.dim_routed = dim_routed

        # Shared expert — always active, captures common knowledge
        self.shared = nn.Linear(dim, dim_shared)

        # Routed experts — selectively activated via top-k
        self.experts = nn.ModuleList(
            [nn.Linear(dim, dim_routed) for _ in range(K)])

        # Router: small gate network
        if routing == 'scenario':
            self.w_gate = nn.Embedding(15, K)
            nn.init.zeros_(self.w_gate.weight)
            self.w_noise = nn.Embedding(15, K)
            nn.init.normal_(self.w_noise.weight, std=0.001)
        else:
            self.w_gate = nn.Linear(dim, K)
            nn.init.normal_(self.w_gate.weight, std=0.01)
            nn.init.zeros_(self.w_gate.bias)
            self.w_noise = nn.Linear(dim, K)
            nn.init.normal_(self.w_noise.weight, std=0.001)
            nn.init.zeros_(self.w_noise.bias)

        # Record last load-balance loss for training loop
        self._last_lb_loss = torch.tensor(0.0)

    def set_top_k(self, k: int):
        """Update top_k for warmup → sparsify schedule."""
        self.top_k = min(k, self.K)

    def load_pretrained(self, pretrained_linear: nn.Linear):
        """Split pretrained Linear(dim,dim) into shared + K routed experts.

        布局: shared → first dim_shared rows; expert_i → next dim_routed rows.
        这样 init 时 (top_k=K, gate uniform), 输出 ≡ vanilla.
        """
        w, b = pretrained_linear.weight.data, pretrained_linear.bias.data
        # Shared: first dim_shared output rows
        self.shared.weight.data.copy_(w[:self.dim_shared])
        self.shared.bias.data.copy_(b[:self.dim_shared])
        # Routed: remaining K × dim_routed rows
        for i, expert in enumerate(self.experts):
            start = self.dim_shared + i * self.dim_routed
            end = start + self.dim_routed
            expert.weight.data.copy_(w[start:end])
            expert.bias.data.copy_(b[start:end])

    def _gate(self, x_or_tab, training: bool):
        """Compute noisy top-k gates.

        Returns:
            gates:      [B, K] — softmax over selected experts, 0 for unselected
            topk_idx:   [B, top_k] — indices of selected experts (None if k==K)
            clean_prob: [B, K] — clean softmax (for load balance loss)
        """
        if self.routing == 'scenario':
            clean_logits = self.w_gate(x_or_tab)      # [B, K]
            noise_logits = self.w_noise(x_or_tab)      # [B, K]
        else:
            clean_logits = self.w_gate(x_or_tab)
            noise_logits = self.w_noise(x_or_tab)

        # Noisy gating during training (exploration).
        # _router_noise_enabled 由 freeze_routers 关闭，保证冻结即零噪声 + 恒均匀门控。
        if training and self.noise_scale > 0 and self._router_noise_enabled:
            noise_std = F.softplus(noise_logits) + 1e-2
            noise = torch.randn_like(clean_logits) * noise_std * self.noise_scale
            logits = clean_logits + noise
        else:
            logits = clean_logits

        # Top-k selection
        if self.top_k < self.K:
            # Sparse mode: top-k hard selection
            topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)  # [B, k]
            topk_gates = F.softmax(topk_logits, dim=-1)
            gates = torch.zeros_like(logits).scatter_(-1, topk_idx, topk_gates)
        else:
            # Warmup mode (top_k = K): scale softmax by K so uniform→1.0,
            # giving raw concatenation (= vanilla Cross layer equivalence)
            gates = F.softmax(logits, dim=-1) * self.K
            topk_idx = None

        clean_prob = F.softmax(clean_logits, dim=-1)  # for load balance
        return gates, topk_idx, clean_prob

    def _load_balance_loss(self, gates: torch.Tensor,
                           clean_prob: torch.Tensor,
                           topk_idx) -> torch.Tensor:
        """Switch Transformer–style load balancing loss.

        L_balance = K · Σ_i f_i · P_i
          f_i = fraction of tokens dispatched to expert i
          P_i = mean softmax probability for expert i

        仅在稀疏模式 (top_k < K) 计算; warmup 模式下返回 0.
        """
        if self.lb_alpha <= 0 or topk_idx is None:
            return torch.tensor(0.0, device=gates.device)

        P_i = clean_prob.mean(0)  # [K]

        # f_i: fraction of tokens where expert i is in top-k
        disp = torch.zeros(self.K, device=gates.device)
        for i in range(self.K):
            disp[i] = (topk_idx == i).any(dim=-1).float().mean()
        f_i = disp + 1e-8

        return self.lb_alpha * self.K * (f_i * P_i).sum()

    def forward(self, x, x0, tab=None):
        """Cross layer forward with MoE.

        Args:
            x:   [B, dim] current cross representation
            x0:  [B, dim] original embedding (x₀ in DCNv2 formula)
            tab: [B] scenario ids (required for 'scenario' routing)

        Returns:
            output: [B, dim] cross layer output
            gates:  [B, K] routing weights
            lb_loss: scalar load balance loss
        """
        # Shared expert — always active
        shared_out = self.shared(x)  # [B, dim_shared]

        # 全软路由 (top_k == K)：所有路由专家按 w_gate 软加权，路由器始终激活
        # （修正：原实现在此分支强制 gates=1 绕过路由器，导致 top_k=K 时路由网络死亡）
        if self.top_k >= self.K:
            router_input = tab if self.routing == 'scenario' else x
            gates, topk_idx, clean_prob = self._gate(router_input, self.training)
            expert_outs = [expert(x) for expert in self.experts]
            routed_out = torch.cat(
                [gates[:, i:i + 1] * expert_outs[i] for i in range(self.K)], dim=-1
            )  # [B, K*dim_routed]
            lb_loss = torch.tensor(0.0, device=x.device)
        else:
            # Sparse mode: noisy top-k routing + load balance
            router_input = tab if self.routing == 'scenario' else x
            gates, topk_idx, clean_prob = self._gate(router_input, self.training)

            expert_outs = [expert(x) for expert in self.experts]  # K × [B, dim_routed]
            routed_out = torch.cat(
                [gates[:, i:i + 1] * expert_outs[i] for i in range(self.K)], dim=-1
            )  # [B, K*dim_routed]

            lb_loss = self._load_balance_loss(gates, clean_prob, topk_idx)

        # Combine: [B, dim_shared + K*dim_routed] = [B, dim]
        combined = torch.cat([shared_out, routed_out], dim=-1)

        # Cross operation: x_{l+1} = x₀ ⊙ f(x_l) + x_l
        output = combined * x0 + x

        return output, gates, lb_loss


class DCNv2MoE_V2(nn.Module):
    """DCNv2 with advanced MoE: shared expert + noisy top-k routing + load balancing.

    架构:
      3 × CrossExpertLayerV2 (MoE Cross) + 2 × DNN Linear + Head
      Sparse Embedding 层与 vanilla DCNv2 完全一致.

    Warmup 机制:
      默认 top_k=K (soft routing), 训练过程中逐渐降至 target_k=2.
      set_top_k(k) 控制每个 CrossExpertLayerV2 的稀疏度.

    Args:
        dim:                Cross layer dim (360)
        K:                  number of routed experts (4)
        top_k:              initial/target active routed experts (default K = soft routing)
        routing:            'scenario' or 'data'
        noise_scale:        gate noise std
        lb_alpha:           load balance loss coefficient
    """

    def __init__(self, dim=360, K=4, top_k=None, routing='data',
                 noise_scale=0.1, lb_alpha=0.01):
        super().__init__()
        self.dim, self.K, self.routing = dim, K, routing
        # Default: top_k=K → warmup mode (vanilla-equiv at init). Training
        # script should call set_top_k(2) after warmup epoch to enable sparsity.
        top_k = top_k if top_k is not None else K
        self.sparse = Sparse()
        self.cross_layers = nn.ModuleList([
            CrossExpertLayerV2(dim, K, top_k=top_k, routing=routing,
                               noise_scale=noise_scale, lb_alpha=lb_alpha)
            for _ in range(3)
        ])
        self.dnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.head = nn.Linear(dim, 15)

    def embed(self, batch):
        return self.sparse(batch)

    def set_top_k(self, k: int):
        """Update top_k for all MoE layers (warmup control)."""
        for layer in self.cross_layers:
            layer.set_top_k(k)

    def load_pretrained(self, pretrained: DCNv2):
        """Initialize from pretrained DCNv2: split Cross + copy Sparse/DNN/Head."""
        self.sparse.load_state_dict(pretrained.sparse.state_dict())
        for i, cross_layer in enumerate(self.cross_layers):
            cross_layer.load_pretrained(pretrained.layers[i])
        self.dnn[0].load_state_dict(pretrained.layers[3].state_dict())
        self.dnn[1].load_state_dict(pretrained.layers[4].state_dict())
        self.head.load_state_dict(pretrained.layers[5].state_dict())

    def forward(self, batch):
        x0 = self.embed(batch)
        x = x0
        gates = []
        lb_total = torch.tensor(0.0, device=x.device)
        CRs = [x0.detach()]
        for layer in self.cross_layers:
            tab = batch["tab"] if self.routing == 'scenario' else None
            x, gate, lb = layer(x, x0, tab)
            gates.append(gate)
            lb_total = lb_total + lb
            CRs.append(x.detach())
        for layer in self.dnn:
            x = layer(x.relu())
            CRs.append(x.detach())
        batch["logit"] = self.head(x)[range(len(x)), batch["tab"]]
        batch["_gate"] = gates
        batch["_load_balance_loss"] = lb_total
        if hasattr(self, "CRs"):
            for uid, c in zip(batch["user_id"], torch.stack(CRs, 1)):
                self.CRs[uid] *= 0.99
                self.CRs[uid] += c

    @staticmethod
    def param_summary(model) -> str:
        """Human-readable parameter count by module group."""
        groups = {
            'Sparse': ['sparse'],
            'Shared Experts': ['cross_layers.0.shared', 'cross_layers.1.shared',
                               'cross_layers.2.shared'],
            'Routed Experts': ['cross_layers.0.experts', 'cross_layers.1.experts',
                               'cross_layers.2.experts'],
            'Routers': ['cross_layers.0.w_gate', 'cross_layers.0.w_noise',
                        'cross_layers.1.w_gate', 'cross_layers.1.w_noise',
                        'cross_layers.2.w_gate', 'cross_layers.2.w_noise'],
            'DNN': ['dnn'],
            'Head': ['head'],
        }
        lines = ['Parameter Summary:']
        total = 0
        for name, prefixes in groups.items():
            count = 0
            for n, p in model.named_parameters():
                if any(n.startswith(px) for px in prefixes):
                    count += p.numel()
            total += count
            pct = count / total * 100 if total > 0 else 0
            lines.append(f'  {name:20s}: {count:>10,}  ({pct:5.1f}%)')
        lines.append(f'  {"TOTAL":20s}: {total:>10,}')
        return '\n'.join(lines)


# ============================================================
#  Stage B: 低秩全维专家 MoE（与 dense 同 FLOPs 当 r = dim/(2K)）
# ============================================================

class LowRankExpert(nn.Module):
    """低秩全维专家：Linear(dim, r) → Linear(r, dim)，输出全维 dim。

    等价于对 Cross 变换做秩 r 分解；多个专家的门控加权组合重构全维输出。
    参数量 2*dim*r（不含偏置），K 个专家总参数 2*K*dim*r；当 r = dim/(2K) 时
    与 dense Linear(dim,dim)（dim²）同量级，构成 same-FLOPs 对照。
    """

    def __init__(self, dim=360, r=45):
        super().__init__()
        self.dim, self.r = dim, r
        self.down = nn.Linear(dim, r)
        self.up = nn.Linear(r, dim)

    def forward(self, x):
        return self.up(self.down(x))  # [B, dim]


class CrossExpertLayerLR(nn.Module):
    """低秩全维 MoE 交叉层：K 个 LowRankExpert + 可学习/可冻结 DataRouter。

    routing 固定为 'data'（DataRouter 由输入特征决定门控，与 V1 一致）。
    冻结时由外部 freeze_routers 把 router.linear 清零 → softmax 恒 1/K（uniform）。
    输出口径与 V1 一致：weighted * x0 + x。
    """

    def __init__(self, dim=360, K=4, r=45, routing="data"):
        super().__init__()
        assert routing == "data", "low-rank full-dim MoE only supports data routing"
        self.dim, self.K, self.r, self.routing = dim, K, r, routing
        self.experts = nn.ModuleList([LowRankExpert(dim, r) for _ in range(K)])
        self.router = DataRouter(dim, K)  # init 0 → softmax 恒 1/K（uniform）

    def forward(self, x, x0, tab=None):
        gate = self.router(x)  # [B, K]
        weighted = sum(gate[:, i:i + 1] * self.experts[i](x) for i in range(self.K))
        return weighted * x0 + x, gate


class DCNv2MoE_LowRank(nn.Module):
    """低秩全维专家 MoE（Stage B 主体）。

    3 层 CrossExpertLayerLR + 2 层 DNN(Linear dim,dim) + head(Linear dim,15)。
    与 DCNv2（5×Linear dim,dim）构成 same-FLOPs / same-total / same-latency 对照：
    每 MoE 交叉层 FLOPs ≈ 2*K*dim*r，取 r = dim/(2K) → 2*K*dim*(dim/2K) = dim²，
    即与 dense 交叉层相等；DNN 与 head 二者一致。
    """

    def __init__(self, dim=360, K=4, r=45, routing="data"):
        super().__init__()
        self.dim, self.K, self.r, self.routing = dim, K, r, routing
        self.sparse = Sparse()
        self.cross_layers = nn.ModuleList(
            [CrossExpertLayerLR(dim, K, r, routing=routing) for _ in range(3)]
        )
        self.dnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.head = nn.Linear(dim, 15)

    def set_top_k(self, top_k):
        # 低秩全维专家每专家输出全维，top-k 路由不适用；保留接口以保持
        # evaluate_all_scenarios 的统一调用（gate 统计量仍照常计算）。
        pass

    def embed(self, batch):
        return self.sparse(batch)

    def forward(self, batch):
        x0 = self.embed(batch)
        x = x0
        gates = []
        CRs = [x0.detach()]
        for layer in self.cross_layers:
            x, gate = layer(x, x0)
            gates.append(gate)
            CRs.append(x.detach())
        for layer in self.dnn:
            x = layer(x.relu())
            CRs.append(x.detach())
        batch["logit"] = self.head(x)[range(len(x)), batch["tab"]]
        batch["_gate"] = gates
        self.last_gates = gates  # 供 collect_gate_stats 汇聚门控结构
        if hasattr(self, "CRs"):
            for uid, c in zip(batch["user_id"], torch.stack(CRs, 1)):
                self.CRs[uid] *= 0.99
                self.CRs[uid] += c

    @staticmethod
    def param_summary(model) -> str:
        groups = {
            "Sparse": ["sparse"],
            "LowRank Experts": ["cross_layers.0.experts", "cross_layers.1.experts",
                                "cross_layers.2.experts"],
            "Routers": ["cross_layers.0.router", "cross_layers.1.router",
                        "cross_layers.2.router"],
            "DNN": ["dnn"],
            "Head": ["head"],
        }
        lines = ["Parameter Summary (LowRank):"]
        total = 0
        for name, prefixes in groups.items():
            count = 0
            for n, p in model.named_parameters():
                if any(n.startswith(px) for px in prefixes):
                    count += p.numel()
            total += count
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {name:20s}: {count:>10,}  ({pct:5.1f}%)")
        lines.append(f'  {"TOTAL":20s}: {total:>10,}')
        return "\n".join(lines)


# ============================================================
#  Task-Conditioned Mixture Routing (TCMR)
# ============================================================

TASK_CONDITIONED_ROUTING_MODES = (
    "frozen-uniform-routing",
    "data-only-routing",
    "task-only-routing",
    "data-and-task-routing",
    "data-and-task-consistency-routing",
)


class TaskConditionedRouter(nn.Module):
    """Router with explicit data and task pathways.

    Every mode constructs both pathways so that model construction consumes the
    same random-number sequence. The selected mode only changes which logits are
    used and which router parameters receive gradients.
    """

    def __init__(self, dim=360, K=4, num_tasks=15,
                 mode="data-and-task-routing"):
        super().__init__()
        if mode not in TASK_CONDITIONED_ROUTING_MODES:
            raise ValueError(
                f"unknown task-conditioned routing mode {mode!r}; "
                f"expected one of {TASK_CONDITIONED_ROUTING_MODES}"
            )
        self.dim, self.K, self.num_tasks, self.mode = dim, K, num_tasks, mode
        self.data_projection = nn.Linear(dim, K)
        self.task_embedding = nn.Embedding(num_tasks, K)
        nn.init.zeros_(self.data_projection.weight)
        nn.init.zeros_(self.data_projection.bias)
        nn.init.zeros_(self.task_embedding.weight)
        self._configure_trainable_pathways()

    def _configure_trainable_pathways(self):
        data_trainable = self.mode in (
            "data-only-routing",
            "data-and-task-routing",
            "data-and-task-consistency-routing",
        )
        task_trainable = self.mode in (
            "task-only-routing",
            "data-and-task-routing",
            "data-and-task-consistency-routing",
        )
        self.data_projection.requires_grad_(data_trainable)
        self.task_embedding.requires_grad_(task_trainable)

    @staticmethod
    def _symmetric_kl(first, second):
        epsilon = torch.finfo(first.dtype).eps
        first = first.clamp_min(epsilon)
        second = second.clamp_min(epsilon)
        first_to_second = (first * (first.log() - second.log())).sum(-1)
        second_to_first = (second * (second.log() - first.log())).sum(-1)
        return 0.5 * (first_to_second + second_to_first).mean()

    def forward(self, x, task_ids):
        data_logits = self.data_projection(x)
        task_logits = self.task_embedding(task_ids)

        if self.mode == "frozen-uniform-routing":
            gate = torch.full_like(data_logits, 1.0 / self.K)
        elif self.mode == "data-only-routing":
            gate = data_logits.softmax(-1)
        elif self.mode == "task-only-routing":
            gate = task_logits.softmax(-1)
        else:
            gate = (data_logits + task_logits).softmax(-1)

        auxiliary_loss = gate.new_zeros(())
        if self.mode == "data-and-task-consistency-routing":
            data_gate = data_logits.softmax(-1)
            auxiliary_loss = self._symmetric_kl(data_gate, gate)
        return gate, auxiliary_loss


class TaskConditionedLowRankCrossExpertLayer(nn.Module):
    """Low-rank full-dimensional expert layer with task-conditioned routing."""

    def __init__(self, dim=360, K=4, r=45, num_tasks=15,
                 routing_mode="data-and-task-routing"):
        super().__init__()
        self.dim, self.K, self.r = dim, K, r
        self.experts = nn.ModuleList([LowRankExpert(dim, r) for _ in range(K)])
        self.router = TaskConditionedRouter(
            dim=dim, K=K, num_tasks=num_tasks, mode=routing_mode,
        )

    def forward(self, x, x0, task_ids):
        gate, auxiliary_loss = self.router(x, task_ids)
        weighted = sum(
            gate[:, index:index + 1] * expert(x)
            for index, expert in enumerate(self.experts)
        )
        return weighted * x0 + x, gate, auxiliary_loss


class DCNv2MoE_TaskConditionedLowRank(nn.Module):
    """DCNv2 low-rank full-dimensional MoE for TCMR experiments.

    MoE means Mixture of Experts. TCMR means Task-Conditioned Mixture Routing.
    This class is isolated from ``DCNv2MoE_LowRank`` so historical Stage B
    checkpoints and parameter names remain unchanged.
    """

    def __init__(self, dim=360, K=4, r=45, num_tasks=15,
                 routing_mode="data-and-task-routing"):
        super().__init__()
        if routing_mode not in TASK_CONDITIONED_ROUTING_MODES:
            raise ValueError(f"unknown routing_mode={routing_mode!r}")
        self.dim, self.K, self.r = dim, K, r
        self.num_tasks, self.routing_mode = num_tasks, routing_mode
        self.sparse = Sparse()
        self.cross_layers = nn.ModuleList([
            TaskConditionedLowRankCrossExpertLayer(
                dim=dim, K=K, r=r, num_tasks=num_tasks,
                routing_mode=routing_mode,
            )
            for _ in range(3)
        ])
        self.dnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.head = nn.Linear(dim, num_tasks)

    def embed(self, batch):
        return self.sparse(batch)

    def forward(self, batch):
        x0 = self.embed(batch)
        x = x0
        gates = []
        auxiliary_losses = []
        representations = [x0.detach()]
        task_ids = batch["tab"]
        for layer in self.cross_layers:
            x, gate, auxiliary_loss = layer(x, x0, task_ids)
            gates.append(gate)
            auxiliary_losses.append(auxiliary_loss)
            representations.append(x.detach())
        for layer in self.dnn:
            x = layer(x.relu())
            representations.append(x.detach())
        batch["logit"] = self.head(x)[range(len(x)), task_ids]
        batch["_gate"] = gates
        batch["_routing_auxiliary_loss"] = torch.stack(auxiliary_losses).mean()
        self.last_gates = gates
        if hasattr(self, "CRs"):
            for user_id, representation in zip(
                    batch["user_id"], torch.stack(representations, 1)):
                self.CRs[user_id] *= 0.99
                self.CRs[user_id] += representation


# ============================================================
#  Function-preserving shared-residual fully-routed MoE
# ============================================================


class ScenarioSpecificHead(nn.Module):
    """Independent per-scenario heads so inactive rows never receive decay."""

    def __init__(self, dim=360, num_tasks=15):
        super().__init__()
        self.dim, self.num_tasks = dim, num_tasks
        self.rows = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_tasks)])

    def load_dense(self, dense_head):
        with torch.no_grad():
            for task_id, row in enumerate(self.rows):
                row.weight.copy_(dense_head.weight[task_id:task_id + 1])
                row.bias.copy_(dense_head.bias[task_id:task_id + 1])

    def forward(self, x, task_ids):
        output = x.new_empty(len(x))
        for task_id in task_ids.unique():
            task_index = int(task_id.item())
            mask = task_ids == task_id
            output[mask] = self.rows[task_index](x[mask]).squeeze(-1)
        return output


class SharedResidualCrossExpertLayer(nn.Module):
    """Dense shared path plus zero-initialized, fully-routed residual experts.

    The shared transform is copied from a dense Cross layer. Every specialist
    returns exactly zero immediately after ``load_dense`` because its up
    projection is zeroed. Therefore conversion preserves the dense function
    while retaining a usable gradient path into each specialist up projection.
    """

    def __init__(self, dim=360, K=4, r=45, num_tasks=15,
                 routing_mode="frozen-uniform-routing", residual_scale=1.0):
        super().__init__()
        self.dim, self.K, self.r = dim, K, r
        self.residual_scale = float(residual_scale)
        self.shared = nn.Linear(dim, dim)
        self.experts = nn.ModuleList([LowRankExpert(dim, r) for _ in range(K)])
        self.router = TaskConditionedRouter(
            dim=dim, K=K, num_tasks=num_tasks, mode=routing_mode,
        )
        self.zero_specialists()

    def zero_specialists(self):
        with torch.no_grad():
            for expert in self.experts:
                nn.init.zeros_(expert.up.weight)
                nn.init.zeros_(expert.up.bias)

    def load_dense(self, dense_layer):
        self.shared.load_state_dict(dense_layer.state_dict())
        self.zero_specialists()

    def forward(self, x, x0, task_ids):
        gate, auxiliary_loss = self.router(x, task_ids)
        residual = sum(
            gate[:, index:index + 1] * expert(x)
            for index, expert in enumerate(self.experts)
        )
        mixed = self.shared(x) + self.residual_scale * residual
        return mixed * x0 + x, gate, auxiliary_loss


class DCNv2MoE_SharedResidual(nn.Module):
    """Function-preserving DCNv2 upcycle target for continual-learning gates."""

    def __init__(self, dim=360, K=4, r=45, num_tasks=15,
                 routing_mode="frozen-uniform-routing", residual_scale=1.0):
        super().__init__()
        if routing_mode not in TASK_CONDITIONED_ROUTING_MODES:
            raise ValueError(f"unknown routing_mode={routing_mode!r}")
        self.dim, self.K, self.r = dim, K, r
        self.num_tasks, self.routing_mode = num_tasks, routing_mode
        self.sparse = Sparse()
        self.cross_layers = nn.ModuleList([
            SharedResidualCrossExpertLayer(
                dim=dim, K=K, r=r, num_tasks=num_tasks,
                routing_mode=routing_mode, residual_scale=residual_scale,
            )
            for _ in range(3)
        ])
        self.dnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.head = ScenarioSpecificHead(dim=dim, num_tasks=num_tasks)

    def load_pretrained(self, pretrained):
        """Copy a dense ``DCNv2`` checkpoint without changing its function."""
        self.sparse.load_state_dict(pretrained.sparse.state_dict())
        for layer_index, cross_layer in enumerate(self.cross_layers):
            cross_layer.load_dense(pretrained.layers[layer_index])
        self.dnn[0].load_state_dict(pretrained.layers[3].state_dict())
        self.dnn[1].load_state_dict(pretrained.layers[4].state_dict())
        self.head.load_dense(pretrained.layers[5])
        return self

    def embed(self, batch):
        return self.sparse(batch)

    def forward(self, batch):
        x0 = self.embed(batch)
        x = x0
        gates = []
        auxiliary_losses = []
        task_ids = batch["tab"]
        for layer in self.cross_layers:
            x, gate, auxiliary_loss = layer(x, x0, task_ids)
            gates.append(gate)
            auxiliary_losses.append(auxiliary_loss)
        for layer in self.dnn:
            x = layer(x.relu())
        batch["logit"] = self.head(x.relu(), task_ids)
        batch["_gate"] = gates
        batch["_routing_auxiliary_loss"] = torch.stack(auxiliary_losses).mean()
        self.last_gates = gates

    def parameter_blocks(self):
        """Return named blocks used by block-wise optimizers and audits."""
        return {
            "backbone": list(self.sparse.parameters()) + list(self.dnn.parameters()),
            "shared": [
                parameter
                for layer in self.cross_layers
                for parameter in layer.shared.parameters()
            ],
            "specialists": [
                parameter
                for layer in self.cross_layers
                for expert in layer.experts
                for parameter in expert.parameters()
            ],
            "router": [
                parameter
                for layer in self.cross_layers
                for parameter in layer.router.parameters()
            ],
            "head": list(self.head.parameters()),
        }

    def specialist_parameters(self, expert_index):
        if not 0 <= expert_index < self.K:
            raise IndexError(f"expert_index={expert_index} outside [0, {self.K})")
        return [
            parameter
            for layer in self.cross_layers
            for parameter in layer.experts[expert_index].parameters()
        ]

    def task_head_parameters(self, task_id):
        return list(self.head.rows[int(task_id)].parameters())


# ============================================================
#  Capacity-Scale MoE: full-rank experts + true top-k sparse dispatch
#
#  这是仓库中第一个「同激活、K× 总容量」的 MoE：K 个 full-rank
#  nn.Linear(dim,dim) 专家（每个专家容量 == dense 交叉层），router 只激活
#  top_k 个专家，未选中专家的前向完全不被执行（区别于 V2 的「全算后加权」）。
#
#  设计取舍（见 docs/20260813-1642-capacity-MoE-驱动.md §3.2）：
#  放弃严格函数保持，所有专家初始化为 dense 副本 + 乘法扰动以打破对称性，
#  router 用非零小随机 init，换取「路由可特化」。初始 AUC 略低于 dense。
# ============================================================


class CapacityCrossExpertLayer(nn.Module):
    """Full-rank capacity MoE Cross layer with true top-k sparse dispatch.

    K 个 full-rank ``nn.Linear(dim, dim)`` 专家 + data-driven router
    ``nn.Linear(dim, K)``。forward 只对被 top-k 选中的专家做前向，未选中
    专家的前向完全不执行（真实稀疏计算，激活 FLOPs ≈ O(B * top_k * dim²)）。

    Args:
        dim:         Cross layer dimension (360)
        K:           number of experts (4)
        top_k:       number of active experts per token (1)
        noise_scale: multiplicative perturbation std when upcycling from dense
        lb_alpha:    load-balancing loss coefficient (Switch Transformer style)
    """

    def __init__(self, dim=360, K=4, top_k=1, noise_scale=0.01, lb_alpha=0.01):
        super().__init__()
        self.dim, self.K = dim, K
        self.top_k = min(top_k, K)
        self.noise_scale = noise_scale
        self.lb_alpha = lb_alpha

        # Full-rank experts: each has the capacity of a dense Cross layer.
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(K)])

        # Data-driven router: non-zero small init breaks the uniform deadlock.
        self.router = nn.Linear(dim, K)
        nn.init.normal_(self.router.weight, std=0.01)
        nn.init.zeros_(self.router.bias)

    def set_top_k(self, k: int):
        self.top_k = min(k, self.K)

    def upcycle_from_dense(self, dense_linear: nn.Linear):
        """Copy a dense Cross layer into every expert + multiplicative noise.

        放弃严格函数保持：每个专家 = dense 权重 * (1 + noise_scale * randn)。
        这样专家不再完全对称，router 有可特化的梯度路径。
        """
        device = self.experts[0].weight.device
        w = dense_linear.weight.data.to(device)
        b = dense_linear.bias.data.to(device)
        with torch.no_grad():
            for expert in self.experts:
                expert.weight.copy_(
                    w * (1.0 + self.noise_scale * torch.randn_like(w)))
                expert.bias.copy_(
                    b + self.noise_scale * torch.randn_like(b))

    def _load_balance_loss(self, clean_prob: torch.Tensor,
                           topk_idx: torch.Tensor) -> torch.Tensor:
        """Switch Transformer load-balancing loss.

        L_balance = K * Σ_i f_i * P_i, where f_i = fraction of tokens
        dispatched to expert i, P_i = mean clean softmax prob of expert i.

        Vectorized via ``scatter_add_`` (equivalent to the historical
        ``(topk_idx == i).any(-1).float().mean()`` because ``topk`` returns
        distinct indices per token) so no per-expert Python loop / host sync
        happens on the training hot path.
        """
        if self.lb_alpha <= 0:
            return torch.zeros((), device=clean_prob.device)
        B = clean_prob.shape[0]
        P_i = clean_prob.mean(0)  # [K]
        flat = topk_idx.reshape(-1)
        disp = torch.zeros(self.K, device=clean_prob.device, dtype=clean_prob.dtype)
        disp.scatter_add_(0, flat, torch.ones_like(flat, dtype=clean_prob.dtype))
        f_i = disp / max(B, 1)
        return self.lb_alpha * self.K * (f_i * P_i).sum()

    def forward(self, x, x0, tab=None):
        """Cross layer forward with true sparse dispatch.

        Returns:
            out:             [B, dim] (weighted * x0 + x)
            clean_prob:      [B, K] full-rank softmax (for entropy sentinel + lb)
            lb_loss:         scalar load-balance loss
            dispatch_counts: [K] int list — tokens actually computed per expert
        """
        B = x.shape[0]
        logits = self.router(x)            # [B, K]
        clean_prob = F.softmax(logits, dim=-1)  # [B, K] (full K, not truncated)

        if self.top_k >= self.K:
            # Full soft routing (non-sparse fallback, e.g. set_top_k(K)).
            expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
            weighted = (clean_prob.unsqueeze(-1) * expert_outs).sum(1)
            dispatch_counts = [B] * self.K
            lb_loss = torch.tensor(0.0, device=x.device)
            return weighted * x0 + x, clean_prob, lb_loss, dispatch_counts

        # True top-k sparse dispatch: only compute selected experts.
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)   # [B, top_k]
        topk_gates = F.softmax(topk_logits, dim=-1)               # [B, top_k]

        # Straight-through estimator (STE): forward uses the hard sparse gate,
        # backward propagates through the full K-way softmax. Without this the
        # top-1 argmax is non-differentiable and the router receives ZERO
        # gradient from the BCE objective (top_k=1 softmax over a single logit
        # is identically 1.0). See docs/20260813-1642-capacity-MoE-驱动.md.
        hard_gate = torch.zeros_like(clean_prob).scatter(
            -1, topk_idx, topk_gates)                             # [B, K] sparse
        gate = (clean_prob + (hard_gate - hard_gate.detach())
                if self.training else hard_gate)                  # [B, K] STE

        # Flatten each (token, selected slot) into a single dispatch record.
        flat_idx = topk_idx.reshape(-1)           # [B*top_k]
        sample_idx = torch.arange(B, device=x.device).repeat_interleave(
            self.top_k)                            # [B*top_k]

        out = torch.zeros_like(x)
        for e in range(self.K):
            # Empty selections are legal (Linear/index_add_ no-op on 0 rows), so
            # the historical `.sum().item()` + `.any()` guards are dropped: they
            # forced 2 extra host syncs per expert on the training hot path.
            sel_samples = sample_idx[flat_idx == e]               # [n]
            sel_gates = gate[sel_samples, e].unsqueeze(-1)       # [n, 1] (STE)
            expert_out = self.experts[e](x[sel_samples])         # [n, dim]
            out.index_add_(0, sel_samples, sel_gates * expert_out)

        # Dispatch counts need a device→host sync, so only materialize them off
        # the training path (sentinel/analysis run under `infer`, i.e. eval).
        dispatch_counts = ([] if self.training else
                           torch.bincount(flat_idx, minlength=self.K).tolist())

        lb_loss = self._load_balance_loss(clean_prob, topk_idx)
        return out * x0 + x, clean_prob, lb_loss, dispatch_counts


class DCNv2CapacityMoE(nn.Module):
    """DCNv2 upcycled into a capacity-scale MoE (full-rank + true sparse).

    3 × CapacityCrossExpertLayer + 2 × DNN(Linear dim,dim) + head(Linear dim,15).
    Isolated from all historical MoE classes so checkpoints/param names stay
    unchanged.
    """

    def __init__(self, dim=360, K=4, top_k=1, noise_scale=0.01, lb_alpha=0.01):
        super().__init__()
        self.dim, self.K = dim, K
        self.top_k = min(top_k, K)
        self.noise_scale = noise_scale
        self.lb_alpha = lb_alpha
        self.sparse = Sparse()
        self.cross_layers = nn.ModuleList([
            CapacityCrossExpertLayer(dim, K, top_k=top_k,
                                     noise_scale=noise_scale, lb_alpha=lb_alpha)
            for _ in range(3)
        ])
        self.dnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.head = nn.Linear(dim, 15)

    def set_top_k(self, k: int):
        for layer in self.cross_layers:
            layer.set_top_k(k)

    def upcycle_from_dense(self, dense: DCNv2):
        """Upcycle a dense DCNv2 checkpoint (Sparse/DNN/Head copied exactly;
        Cross layers copied with symmetry-breaking noise)."""
        self.sparse.load_state_dict(dense.sparse.state_dict())
        for i, cross_layer in enumerate(self.cross_layers):
            cross_layer.upcycle_from_dense(dense.layers[i])
        self.dnn[0].load_state_dict(dense.layers[3].state_dict())
        self.dnn[1].load_state_dict(dense.layers[4].state_dict())
        self.head.load_state_dict(dense.layers[5].state_dict())
        return self

    def embed(self, batch):
        return self.sparse(batch)

    def forward(self, batch):
        x0 = self.embed(batch)
        x = x0
        gates = []
        dispatch_counts = []
        lb_total = torch.tensor(0.0, device=x.device)
        for layer in self.cross_layers:
            x, gate, lb, dc = layer(x, x0)
            gates.append(gate)
            dispatch_counts.append(dc)
            lb_total = lb_total + lb
        for layer in self.dnn:
            x = layer(x.relu())
        batch["logit"] = self.head(x)[range(len(x)), batch["tab"]]
        batch["_gate"] = gates                     # 3 × [B, K] clean softmax
        batch["_dispatch_counts"] = dispatch_counts  # 3 × [K] int list
        batch["_load_balance_loss"] = lb_total

    @staticmethod
    def param_summary(model) -> str:
        groups = {
            "Sparse": ["sparse"],
            "FullRank Experts": ["cross_layers.0.experts", "cross_layers.1.experts",
                                 "cross_layers.2.experts"],
            "Routers": ["cross_layers.0.router", "cross_layers.1.router",
                        "cross_layers.2.router"],
            "DNN": ["dnn"],
            "Head": ["head"],
        }
        lines = ["Parameter Summary (Capacity MoE):"]
        total = 0
        for name, prefixes in groups.items():
            count = 0
            for n, p in model.named_parameters():
                if any(n.startswith(px) for px in prefixes):
                    count += p.numel()
            total += count
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {name:20s}: {count:>10,}  ({pct:5.1f}%)")
        lines.append(f'  {"TOTAL":20s}: {total:>10,}')
        return "\n".join(lines)
