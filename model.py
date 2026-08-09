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
    """Data-driven router: Linear(dim, K) — selects experts from input features."""
    def __init__(self, dim=360, K=4):
        super().__init__()
        self.linear = nn.Linear(dim, K)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).softmax(-1)  # [B, K]


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
    """AdaTask-inspired optimizer that modulates per-expert learning rates.

    Core insight from AdaTask (Yang et al., 2023):
    AU (accumulated squared gradient) tracks per-task gradient magnitude,
    naturally identifying which parameters are important for which task.

    We adapt this to MoE experts across scenarios:
    - Track AU per (layer, expert, scenario) via backward hooks
    - Modulate per-expert gradient magnitude based on AU

    Three modes:
    - "encourage": LR ∝ AU^α — amplify high-AU experts, push specialization
    - "suppress":  LR ∝ (1/AU)^α — dampen dominant experts, force sharing
    - "none":      No modulation (baseline MoE, only tracks AU)

    This is a zero-overhead optimizer: AU tracking is done via backward hooks,
    gradient modulation is O(K) per optimizer step.
    """

    def __init__(self, model: DCNv2MoE, lr=1e-3, mode="none",
                 alpha=0.5, beta=0.99, weight_decay=0.01):
        self.model = model
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        # AU tracking: (layer_idx, expert_idx, scenario) → float
        self.AU: dict = {}
        self._hooks = []
        self._current_scenario: int | None = None

    def set_scenario(self, scenario: int):
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

    def register_hooks(self):
        """Register backward hooks on all expert parameters."""
        for li, layer in enumerate(self.model.cross_layers):
            for ei, expert in enumerate(layer.experts):
                for param in expert.parameters():
                    h = param.register_hook(self._make_hook(li, ei))
                    self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _compute_factors(self) -> dict:
        """Compute per-expert modulation factors for the current scenario.

        Returns: dict mapping (layer_idx, expert_idx) → float factor
        """
        s = self._current_scenario
        if s is None or self.mode == "none":
            return {}

        factors = {}
        for li, layer in enumerate(self.model.cross_layers):
            aus = []
            for ei in range(self.model.K):
                au = self.AU.get((li, ei, s), 0.0)
                aus.append(au)

            mean_au = sum(aus) / len(aus) + 1e-30

            for ei in range(self.model.K):
                au = aus[ei] + 1e-30
                if self.mode == "encourage":
                    factor = (au / mean_au) ** self.alpha
                elif self.mode == "suppress":
                    factor = (mean_au / au) ** self.alpha
                else:
                    factor = 1.0
                factors[(li, ei)] = float(factor)

        return factors

    def modulate_and_zero_grad(self):
        """Apply gradient modulation + optimizer.step() + zero_grad."""
        factors = self._compute_factors()
        if factors:
            for li, layer in enumerate(self.model.cross_layers):
                for ei, expert in enumerate(layer.experts):
                    f = factors.get((li, ei), 1.0)
                    for param in expert.parameters():
                        if param.grad is not None:
                            param.grad.mul_(f)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    TARGET_SCENARIOS = [0, 1, 2, 3, 4, 5, 6, 8]

    def dominance_matrix(self, layer_idx: int = 0) -> dict:
        ratios = {}
        for ei in range(self.model.K):
            au_per_scenario = {s: self.AU.get((layer_idx, ei, s), 0.0)
                               for s in self.TARGET_SCENARIOS}
            total = sum(au_per_scenario.values()) + 1e-30
            for s, au in au_per_scenario.items():
                ratios[(ei, s)] = au / total
        return ratios

    def summary(self) -> str:
        lines = []
        for li in range(3):
            ratios = self.dominance_matrix(li)
            lines.append(f"--- Cross Layer {li} Dominance ({self.mode}) ---")
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

        # Noisy gating during training (exploration)
        if training and self.noise_scale > 0:
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

        # Warmup mode (top_k == K): raw concatenation, no gating → exact vanilla equiv
        if self.top_k >= self.K:
            expert_outs = [expert(x) for expert in self.experts]
            routed_out = torch.cat(expert_outs, dim=-1)  # [B, K*dim_routed]
            gates = torch.ones(x.shape[0], self.K, device=x.device)
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
