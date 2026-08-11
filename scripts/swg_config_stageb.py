"""Stage B 同 FLOPs 门控对照：实验代号 → 配置 的唯一映射表（单一事实来源）。

代号规则见 docs/20260811-2310-StageB-MoE-九次训练驱动.md §二。
每个 run-code 展开为对 run_moe_pretrain_from_scratch.py 的固定命令，机械可执行。

设计（见驱动文档 §一/§二）：
- stgb-lrfd-fr  ：低秩全维专家 (rank r, 输出全维) + 冻结路由 | frozen | sample
- stgb-lrfd-soft：低秩全维专家 + soft routing（可学习 DataRouter）       | soft  | sample
- stgb-sfd      ：same-FLOPs dense 对照（普通 DCNv2，无路由）            | none  | sample

公平口径（控制变量）：cuda:0 独占、seed∈{42,123,456}、lr=1e-3、beta2=0.999、
shuffle 开、K=4、routing=data、batch=10000、max_epochs=300，全部 sample 加权。
r 默认 dim//(2K)=45，使每个低秩交叉层 FLOPs = 2*K*dim*r = dim²，
与 dense Linear(dim,dim) 同量级（same-FLOPs / same-total / same-latency）。
"""

COMMON = dict(
    device="cuda:0",
    lr=1e-3,
    beta2=0.999,
    shuffle=True,
    K=4,
    routing="data",
    batch_size=10000,
    max_epochs=300,
    rank=45,  # dim//(2K) → 与 dense 同 FLOPs
)

# run-code -> 与 COMMON 合并后的差异配置。
# 字段：model, router, weighting, seed
RUN_CARDS = {
    # ---- Stage B：低秩全维专家 + 冻结路由（frozen，Vanilla 等价） ----
    "stgb-lrfd-fr-42":   dict(model="lowrank-full-dim", router="frozen", weighting="sample", seed=42),
    "stgb-lrfd-fr-123":  dict(model="lowrank-full-dim", router="frozen", weighting="sample", seed=123),
    "stgb-lrfd-fr-456":  dict(model="lowrank-full-dim", router="frozen", weighting="sample", seed=456),
    # ---- Stage B：低秩全维专家 + soft routing（可学习门控） ----
    "stgb-lrfd-soft-42": dict(model="lowrank-full-dim", router="soft",   weighting="sample", seed=42),
    "stgb-lrfd-soft-123":dict(model="lowrank-full-dim", router="soft",   weighting="sample", seed=123),
    "stgb-lrfd-soft-456":dict(model="lowrank-full-dim", router="soft",   weighting="sample", seed=456),
    # ---- Stage B：same-FLOPs dense 对照（普通 DCNv2，无路由） ----
    "stgb-sfd-42":       dict(model="same-flops-dense", router="none",   weighting="sample", seed=42),
    "stgb-sfd-123":      dict(model="same-flops-dense", router="none",   weighting="sample", seed=123),
    "stgb-sfd-456":      dict(model="same-flops-dense", router="none",   weighting="sample", seed=456),
}


def build_command(code):
    """把 run-code 展开为 run_moe_pretrain_from_scratch.py 的 argv 列表。"""
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
        "--rank", str(card.get("rank", COMMON["rank"])),
        "--batch-size", str(COMMON["batch_size"]),
        "--max-epochs", str(COMMON["max_epochs"]),
    ]
    if COMMON["shuffle"]:
        cmd.append("--shuffle")
    if card["router"] == "frozen":
        cmd += ["--router", "frozen"]
    elif card["router"] == "soft":
        cmd += ["--router", "soft"]
    elif card["router"] == "none":
        cmd += ["--router", "none"]
    return cmd
