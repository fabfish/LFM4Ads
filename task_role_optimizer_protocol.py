"""按参数角色隔离子任务优化器的冻结实验协议。"""

from __future__ import annotations

from dataclasses import dataclass


NUM_SCENARIOS = 15
MACRO_SCENARIOS = (0, 1, 2, 3, 4, 5, 6, 8)
AUX_SCENARIOS = (7, 12)
FORMAL_SEEDS = (42, 123, 456, 789)
DEVELOPMENT_SEED = 202
SEED_DEVICE = {42: "cuda:0", 123: "cuda:1", 456: "cuda:2", 789: "cuda:0", 202: "cuda:1"}
EXPECTED_SAMPLE_COUNTS_27K = {
    "train": 255_474_457,
    "valid": 34_034_748,
    "test": 32_769_180,
}
EXPECTED_MODEL_PARAMS = 869_750
EXPERT_LEARNING_RATES = (2e-4, 5e-4, 1e-3)
ROUTER_LEARNING_RATE_RATIOS = (0.0, 0.02, 0.05, 0.1)
SHARED_LEARNING_RATE_RATIOS = (0.5, 1.0)
INITIAL_EXPERT_LEARNING_RATE = 5e-4
INITIAL_ROUTER_LEARNING_RATE_RATIO = 0.05
BATCH_SIZE = 10_000
MAX_EPOCHS = 20
PATIENCE = 10
TOP_K = 2
NUM_EXPERTS = 5
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01


@dataclass(frozen=True)
class FrozenTrainingConfig:
    expert_learning_rate: float = INITIAL_EXPERT_LEARNING_RATE
    router_learning_rate_ratio: float = INITIAL_ROUTER_LEARNING_RATE_RATIO
    shared_learning_rate_ratio: float = 1.0
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    num_experts: int = NUM_EXPERTS
    top_k: int = TOP_K
    betas: tuple[float, float] = BETAS
    eps: float = EPS
    weight_decay: float = WEIGHT_DECAY

    @property
    def router_learning_rate(self) -> float:
        return self.expert_learning_rate * self.router_learning_rate_ratio

    @property
    def shared_learning_rate(self) -> float:
        return self.expert_learning_rate * self.shared_learning_rate_ratio
