"""Capacity-Scale MoE — 训练与烟测入口.

实现第一个「同激活、K× 总容量」的 MoE：K 个 full-rank 专家 + 真实 top-k
稀疏 dispatch + Switch Transformer load-balancing + dense checkpoint upcycling。

流程:
  1. 加载 dense checkpoint (cache/dcnv2_vanilla.pt)，evaluate 作为臂 A 参照
  2. upcycle 到 DCNv2CapacityMoE（专家 = dense 副本 + 乘法扰动打破对称性）
  3. per-scenario forward+backward（sample 加权）+ load-balance loss
  4. 每 epoch 记录 valid AUC + clean gate 熵（哨兵）+ per-scenario 分化度
  5. 输出 result_capacity_moe.csv + cache/capacity_moe_history.json

哨兵判定见 docs/20260813-1642-capacity-MoE-驱动.md §六：
  - PASS        : 不崩 + 稀疏生效 + 三层熵均值 ≤ log K - 0.15 + 分化度 ≥ 1/3
  - FAIL        : 崩溃或稀疏未生效
  - INCONCLUSIVE: 熵下降 < 0.15 nats 或分化度 = 0/3
"""

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy

import torch
from tqdm import tqdm

import fields
from dataset import Dataset, Split
from model import DCNv2, DCNv2CapacityMoE, trainable_parameters
from train import evaluate, infer, scenario_loss

LOG_K = math.log(4)  # 哨兵锚点 log K = log 4


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="Capacity-Scale MoE smoke")
    ap.add_argument("device", nargs="?", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=4, help="number of full-rank experts")
    ap.add_argument("--top-k", type=int, default=1, help="active experts per token")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--noise-scale", type=float, default=0.01,
                    help="upcycle multiplicative perturbation std")
    ap.add_argument("--lb-alpha", type=float, default=0.01,
                    help="load-balance loss coefficient")
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="epochs of full soft routing (top_k=K) before sparsifying; "
                         "lets experts differentiate before sparse dispatch")
    ap.add_argument("--min-epochs", type=int, default=1,
                    help="minimum epochs before early-stop is allowed")
    ap.add_argument("--tag", default="", help="suffix for output artifacts")
    return ap.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])
DEVICE = ARGS.device
CACHE_DIR = "cache"
SUF = f"_{ARGS.tag}" if ARGS.tag else ""
VANILLA_PATH = f"{CACHE_DIR}/dcnv2_vanilla.pt"
RESULT_CSV = f"result_capacity_moe{SUF}.csv"
HISTORY_JSON = f"{CACHE_DIR}/capacity_moe_history{SUF}.json"

torch.manual_seed(ARGS.seed)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"[config] device={DEVICE} seed={ARGS.seed} K={ARGS.K} top_k={ARGS.top_k} "
      f"lr={ARGS.lr} beta2={ARGS.beta2} noise_scale={ARGS.noise_scale} "
      f"lb_alpha={ARGS.lb_alpha} batch_size={ARGS.batch_size} "
      f"max_epochs={ARGS.max_epochs} tag='{ARGS.tag or 'none'}'")
print(f"[sentinel] log K = {LOG_K:.4f}; PASS threshold = {LOG_K - 0.15:.4f}")


# ===========================================================
#  Step 1: dense baseline (arm A, read-only)
# ===========================================================
print("=" * 60)
print("Step 1: dense baseline (arm A)")
print("=" * 60)

if not os.path.exists(VANILLA_PATH):
    raise FileNotFoundError(
        f"dense checkpoint {VANILLA_PATH} not found; run main.py first")

vanilla = DCNv2().to(DEVICE)
vanilla.load_state_dict(torch.load(VANILLA_PATH, map_location=DEVICE))
test_set = Split("all")[2]
dense_test_auc = float(evaluate(vanilla, test_set))
print(f"  dense test AUC (all): {dense_test_auc:.4f}")


# ===========================================================
#  Step 2: upcycle to DCNv2CapacityMoE
# ===========================================================
print()
print("=" * 60)
print(f"Step 2: upcycle to DCNv2CapacityMoE (K={ARGS.K}, top_k={ARGS.top_k})")
print("=" * 60)

moe = DCNv2CapacityMoE(
    dim=360, K=ARGS.K, top_k=ARGS.top_k,
    noise_scale=ARGS.noise_scale, lb_alpha=ARGS.lb_alpha,
).to(DEVICE)
moe.upcycle_from_dense(vanilla)
print(f"\n  {DCNv2CapacityMoE.param_summary(moe)}")


def routing_sentinel(model, dataset):
    """Collect clean-gate entropy + per-scenario divergence + dispatch check.

    Returns a dict:
      entropy_per_layer: [h0, h1, h2] mean clean-gate entropy over samples
      entropy_mean: mean of the three layers
      global_argmax:  [a0, a1, a2] global argmax expert per layer
      scenario_argmax: {scenario: [a0, a1, a2]} per-scenario argmax per layer
      divergence_ratio: fraction of layers with ≥1 scenario diverging from global
      dispatch_verified: bool — every layer computes exactly B*top_k tokens
    """
    ent_acc = [0.0, 0.0, 0.0]
    n_samples = 0
    # per-scenario accumulated gate (for argmax): {scenario: [layer] -> [K] sum}
    scen_sum = {}
    scen_cnt = {}
    dispatch_ok = [True, True, True]

    for batch in infer(model, dataset):
        gates = batch.get("_gate", [])
        dc = batch.get("_dispatch_counts", [])
        tab = batch["tab"]
        B = tab.shape[0]
        for li in range(3):
            p = gates[li].clamp_min(1e-12)
            ent_acc[li] += (-(p * p.log()).sum(-1)).sum().item()
            if li < len(dc):
                total = sum(dc[li])
                if total != B * model.top_k:
                    dispatch_ok[li] = False
            # accumulate per-scenario gate sums
            g = gates[li]
            for s in tab.unique():
                si = int(s.item())
                mask = tab == s
                scen_sum.setdefault(si, [None, None, None])
                scen_cnt.setdefault(si, 0)
                if scen_sum[si][li] is None:
                    scen_sum[si][li] = torch.zeros(g.shape[1], device="cpu")
                scen_sum[si][li] += g[mask].sum(0).detach().cpu()
            # increment per-scenario count once per layer (equal across layers)
        n_samples += B
        for s in tab.unique():
            scen_cnt[int(s.item())] += int((tab == s).sum().item())

    entropy_per_layer = [e / max(n_samples, 1) for e in ent_acc]

    global_argmax = []
    scenario_argmax = {}
    scen_mean = {}
    for s, sums in scen_sum.items():
        scen_mean[s] = [
            (sums[li] / max(scen_cnt[s], 1)).argmax().item() for li in range(3)
        ]
        scenario_argmax[s] = scen_mean[s]

    for li in range(3):
        # global argmax from scenario means (unweighted across scenarios)
        am = [
            scen_mean[s][li] for s in sorted(scen_mean) if scen_mean[s][li] is not None
        ]
        global_argmax.append(max(set(am), key=am.count) if am else -1)

    diverged_layers = 0
    for li in range(3):
        if global_argmax[li] < 0:
            continue
        if any(scen_mean[s][li] != global_argmax[li] for s in scen_mean):
            diverged_layers += 1
    divergence_ratio = diverged_layers / 3.0

    return {
        "entropy_per_layer": entropy_per_layer,
        "entropy_mean": sum(entropy_per_layer) / 3.0,
        "global_argmax": global_argmax,
        "scenario_argmax": {str(s): v for s, v in scenario_argmax.items()},
        "divergence_ratio": divergence_ratio,
        "dispatch_verified": all(dispatch_ok),
    }


# ===========================================================
#  Step 3: train with per-scenario sample weighting + lb loss
# ===========================================================
print()
print("=" * 60)
print("Step 3: train DCNv2CapacityMoE")
print("=" * 60)

train_set, valid_set, test_set = Split("all")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    trainable_parameters(moe), lr=ARGS.lr, betas=(0.9, ARGS.beta2))
device = torch.device(DEVICE)

history = []
auc_best = 0.0
best_state = None
best_epoch = 0

csv_header = ("epoch,top_k,valid_auc,entropy_l0,entropy_l1,entropy_l2,"
              "entropy_mean,divergence_ratio,lb_loss,wall_clock_sec")
with open(RESULT_CSV, "w") as f:
    f.write(csv_header + "\n")

for epoch in range(1, ARGS.max_epochs + 1):
    # Warmup: full soft routing (top_k=K) lets experts differentiate before
    # sparsifying. During warmup the router has a real softmax-over-K gradient
    # path (unlike the non-differentiable top-1 argmax) and every expert is
    # trained, so the upcycled experts can recover from their perturbation.
    cur_top_k = ARGS.K if epoch <= ARGS.warmup_epochs else ARGS.top_k
    moe.set_top_k(cur_top_k)
    phase = "warmup" if cur_top_k >= ARGS.K else "sparse"

    t0 = time.time()
    loader = torch.utils.data.DataLoader(
        Dataset(train_set), batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers, shuffle=True, pin_memory=True,
    )
    lb_total_epoch = 0.0
    total_steps = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        moe.train()
        for field_name in fields.all:
            batch[field_name] = batch[field_name].to(
                device, non_blocking=True).int()
        tab_batch = batch["tab"]

        for s in tab_batch.unique():
            mask = tab_batch == s
            sub = {k: v[mask] for k, v in batch.items()}

            moe(sub)
            loss = scenario_loss(
                criterion, sub["logit"], sub["is_click"].float(),
                mask, tab_batch, "sample",
            )
            lb_loss = sub.get("_load_balance_loss",
                              torch.tensor(0.0, device=device))
            loss = loss + lb_loss
            lb_total_epoch += float(lb_loss.item())

            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        total_steps += 1

    valid_auc = float(evaluate(moe, valid_set))
    sentinel = routing_sentinel(moe, test_set)
    wall = time.time() - t0
    lb_mean = lb_total_epoch / max(total_steps, 1)

    ent = sentinel["entropy_per_layer"]
    print(f"  [{phase}] Epoch {epoch} valid AUC={valid_auc:.4f} "
          f"entropy_mean={sentinel['entropy_mean']:.4f} (logK={LOG_K:.4f}) "
          f"divergence={sentinel['divergence_ratio']:.2f} "
          f"dispatch_ok={sentinel['dispatch_verified']} "
          f"lb={lb_mean:.6f} wall={wall:.1f}s")

    record = {
        "epoch": epoch, "top_k": cur_top_k, "valid_auc": valid_auc,
        "entropy_l0": ent[0], "entropy_l1": ent[1], "entropy_l2": ent[2],
        "entropy_mean": sentinel["entropy_mean"],
        "divergence_ratio": sentinel["divergence_ratio"],
        "global_argmax": sentinel["global_argmax"],
        "scenario_argmax": sentinel["scenario_argmax"],
        "dispatch_verified": sentinel["dispatch_verified"],
        "lb_loss": lb_mean, "wall_clock_sec": wall,
    }
    history.append(record)

    with open(RESULT_CSV, "a") as f:
        f.write(f"{epoch},{cur_top_k},{valid_auc:.4f},{ent[0]:.4f},"
                f"{ent[1]:.4f},{ent[2]:.4f},{sentinel['entropy_mean']:.4f},"
                f"{sentinel['divergence_ratio']:.2f},{lb_mean:.6f},{wall:.1f}\n")

    if valid_auc > auc_best + 0.001:
        auc_best = valid_auc
        best_state = deepcopy(moe.state_dict())
        best_epoch = epoch
    elif epoch <= ARGS.min_epochs or epoch <= ARGS.warmup_epochs:
        # Protected phase: do not early-stop during warmup or before min_epochs,
        # so the MoE gets enough steps to recover from upcycle perturbation.
        print(f"  [no-early-stop] epoch {epoch} protected "
              f"(min_epochs={ARGS.min_epochs}, warmup_epochs={ARGS.warmup_epochs})")
    else:
        print(f"  [early-stop] valid AUC no +0.001 gain over best "
              f"({auc_best:.4f} @ epoch {best_epoch}); stop")
        break

if best_state is not None:
    moe.load_state_dict(best_state)
    print(f"  restored best state (valid AUC={auc_best:.4f} @ epoch {best_epoch})")

# ===========================================================
#  Step 4: final test AUC + sentinel verdict
# ===========================================================
print()
print("=" * 60)
print("Step 4: final test AUC + sentinel")
print("=" * 60)

test_auc = float(evaluate(moe, test_set))
final_sentinel = routing_sentinel(moe, test_set)
ent_mean = final_sentinel["entropy_mean"]
div = final_sentinel["divergence_ratio"]

if not final_sentinel["dispatch_verified"]:
    verdict = "FAIL"
elif ent_mean <= LOG_K - 0.15 and div >= 1.0 / 3.0:
    verdict = "PASS"
elif ent_mean > LOG_K - 0.15:
    verdict = "INCONCLUSIVE"
else:
    verdict = "INCONCLUSIVE"

print(f"  dense test AUC (arm A)      : {dense_test_auc:.4f}")
print(f"  capacity-MoE test AUC (arm) : {test_auc:.4f}  "
      f"Δ={test_auc - dense_test_auc:+.4f}")
print(f"  entropy_mean = {ent_mean:.4f}  (log K = {LOG_K:.4f}, "
      f"threshold = {LOG_K - 0.15:.4f})")
print(f"  divergence_ratio = {div:.2f}  dispatch_verified = "
      f"{final_sentinel['dispatch_verified']}")
print(f"  SENTINEL VERDICT = {verdict}")

torch.save(moe.state_dict(),
           f"{CACHE_DIR}/dcnv2_capacity_moe{SUF}.pt")

with open(HISTORY_JSON, "w") as f:
    json.dump({
        "config": {
            "device": DEVICE, "seed": ARGS.seed, "K": ARGS.K,
            "top_k": ARGS.top_k, "lr": ARGS.lr, "beta2": ARGS.beta2,
            "noise_scale": ARGS.noise_scale, "lb_alpha": ARGS.lb_alpha,
            "batch_size": ARGS.batch_size, "max_epochs": ARGS.max_epochs,
            "warmup_epochs": ARGS.warmup_epochs, "min_epochs": ARGS.min_epochs,
            "tag": ARGS.tag,
        },
        "log_k": LOG_K,
        "entropy_pass_threshold": LOG_K - 0.15,
        "dense_test_auc": dense_test_auc,
        "test_auc": test_auc,
        "best_valid_auc": auc_best,
        "best_epoch": best_epoch,
        "final_sentinel": final_sentinel,
        "verdict": verdict,
        "history": history,
    }, f, indent=2)

print(f"\nDone! Outputs:")
print(f"  {RESULT_CSV}     — per-epoch AUC / entropy / divergence / lb / wall")
print(f"  {HISTORY_JSON} — full history + frozen config + verdict")
