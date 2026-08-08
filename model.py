import torch
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

    def check_and_enable(self, tracker: GradientTracker, layer_idx: int = 0):
        """Enable specialization loss if any expert shows clear scenario dominance."""
        ratios = tracker.dominance_matrix(layer_idx)
        for (ei, s), ratio in ratios.items():
            if ratio > self.threshold:
                self.enabled = True
                return True
        return False


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
