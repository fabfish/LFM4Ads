"""Stage A 样本加权关卡：实验代号 → 配置 的唯一映射表（单一事实来源）。

代号规则见 docs/20260811-2010-按场景训练代价消除与混合专家可竞争性验证.md §三。
每个 run-code 展开为对 run_moe_pretrain_from_scratch.py 的固定命令，机械可执行。

COMMON 为所有 run 共享的口径（控制变量）：cuda:0 独占、seed 由卡片指定、
lr=1e-3、beta2=0.999、shuffle 开、K=4、routing=data、batch=10000。
"""

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON = dict(
    device="cuda:0",
    lr=1e-3,
    beta2=0.999,
    shuffle=True,
    K=4,
    routing="data",
    batch_size=10000,
    max_epochs=300,
)

# run-code -> 与 COMMON 合并后的差异配置。
# 字段：model, vanilla_per_scenario, freeze_router, weighting, seed
RUN_CARDS = {
    # ---- Stage A 关卡：稠密 + sample 加权，3 种子 ----
    "swg-dens-sp-42":  dict(model="vanilla", vanilla_per_scenario=True,  freeze_router=False, weighting="sample", seed=42),
    "swg-dens-sp-123": dict(model="vanilla", vanilla_per_scenario=True,  freeze_router=False, weighting="sample", seed=123),
    "swg-dens-sp-456": dict(model="vanilla", vanilla_per_scenario=True,  freeze_router=False, weighting="sample", seed=456),
    # ---- 铺开（仅关卡 PASS 后）：全路由 冻结路由 + sample ----
    "swg-frout-sp-42":  dict(model="fully-routed", vanilla_per_scenario=False, freeze_router=True,  weighting="sample", seed=42),
    "swg-frout-sp-123": dict(model="fully-routed", vanilla_per_scenario=False, freeze_router=True,  weighting="sample", seed=123),
    "swg-frout-sp-456": dict(model="fully-routed", vanilla_per_scenario=False, freeze_router=True,  weighting="sample", seed=456),
    # ---- 铺开：全路由 正常路由 + sample ----
    "swg-nrout-sp-42":  dict(model="fully-routed", vanilla_per_scenario=False, freeze_router=False, weighting="sample", seed=42),
    "swg-nrout-sp-123": dict(model="fully-routed", vanilla_per_scenario=False, freeze_router=False, weighting="sample", seed=123),
    "swg-nrout-sp-456": dict(model="fully-routed", vanilla_per_scenario=False, freeze_router=False, weighting="sample", seed=456),
    # ---- 铺开：部分路由加共享 冻结路由 + sample ----
    "swg-pshr-sp-42":   dict(model="partial-shared", vanilla_per_scenario=False, freeze_router=True,  weighting="sample", seed=42),
    "swg-pshr-sp-123":  dict(model="partial-shared", vanilla_per_scenario=False, freeze_router=True,  weighting="sample", seed=123),
    "swg-pshr-sp-456":  dict(model="partial-shared", vanilla_per_scenario=False, freeze_router=True,  weighting="sample", seed=456),
}

# 关卡追加种子（仅 INCONCLUSIVE 时启用，预声明）
APPEND_SEEDS = {"swg-dens-sp-789": 789, "swg-dens-sp-2024": 2024}


def build_command(code):
    """把 run-code 展开为 run_moe_pretrain_from_scratch.py 的 argv 列表。"""
    if code in APPEND_SEEDS:
        card = dict(model="vanilla", vanilla_per_scenario=True,
                    freeze_router=False, weighting="sample",
                    seed=APPEND_SEEDS[code])
    else:
        card = RUN_CARDS[code]
    cmd = [
        "python", "run_moe_pretrain_from_scratch.py",
        "--model", card["model"],
        "--scenario-loss-weighting", card["weighting"],
        "--seed", str(card["seed"]),
        "--run-code", code,
        "--device", COMMON["device"],
        "--lr", str(COMMON["lr"]),
        "--beta2", str(COMMON["beta2"]),
        "--K", str(COMMON["K"]),
        "--routing", COMMON["routing"],
        "--batch-size", str(COMMON["batch_size"]),
        "--max-epochs", str(COMMON["max_epochs"]),
    ]
    if COMMON["shuffle"]:
        cmd.append("--shuffle")
    if card.get("vanilla_per_scenario"):
        cmd.append("--vanilla-per-scenario")
    if card.get("freeze_router"):
        cmd.append("--freeze-router")
    return cmd


def command_hash(code):
    return hashlib.sha256(json.dumps(build_command(code), sort_keys=True).encode()).hexdigest()
