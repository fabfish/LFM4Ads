import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from dataset import Dataset, GpuBatches, Split
from model import (
    DCNv2,
    DCNv2CapacityMoE,
    apply_freeze,
    freeze_summary,
    trainable_parameters,
)
from train import evaluate, evaluate_gpu, infer, infer_gpu, scenario_loss

LOG_K = math.log(4)  # 哨兵锚点 log K = log 4


def _eval(model, source):
    """AUC on either a DataFrame split or a GPU-resident batch source."""
    if isinstance(source, GpuBatches):
        return evaluate_gpu(model, source)
    return evaluate(model, source)


def _iter_infer(model, source):
    """Forward-pass iterator for either source kind (batches stay mutated)."""
    if isinstance(source, GpuBatches):
        return infer_gpu(model, source)
    return infer(model, source)


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
    ap.add_argument("--train-dense-ref", action="store_true",
                    help="also continue-train the vanilla dense checkpoint for "
                         "max_epochs (dense-continued arm A') to strip the "
                         "'more training' confound from the necessity delta")
    ap.add_argument("--freeze", default="",
                    help="comma-separated param groups to freeze, applied "
                         "IDENTICALLY to both arms (e.g. 'sparse' freezes the "
                         "84M embedding table = 97.9%% of params, which is "
                         "unrelated to the MoE change and is the dominant "
                         "overfitting source)")
    ap.add_argument("--lr-router", type=float, default=None,
                    help="separate LR for the randomly-initialized routers "
                         "(defaults to --lr). Upcycled experts carry pretrained "
                         "weights and need a small LR, while a fresh router "
                         "needs a larger one.")
    ap.add_argument("--full-batch-loss", action="store_true",
                    help="single full-batch forward/backward instead of the "
                         "per-scenario sub-batch loop. Mathematically equivalent "
                         "for 'sample' weighting (verified R_gain≈0.999, see "
                         "DRIVERS.md §2) but 8× larger kernels and 8× fewer host "
                         "syncs, so GPU utilization is far higher.")
    ap.add_argument("--gpu-resident-data", action="store_true",
                    help="keep the whole split on the GPU as one int32 table and "
                         "slice batches device-side, instead of the pandas "
                         "per-row DataLoader. Verified bit-identical AUC and "
                         "~14× faster evaluation.")
    ap.add_argument("--reinit-cross", action="store_true",
                    help="randomly re-initialize the Cross layers in BOTH arms "
                         "(dense layers.0-2 / every MoE expert) while keeping "
                         "the pretrained frozen embeddings + DNN + head. This "
                         "creates real learning headroom in exactly the "
                         "component the MoE modifies, which is the only regime "
                         "where extra Cross capacity can pay off.")
    ap.add_argument("--tag", default="", help="suffix for output artifacts")
    return ap.parse_args(argv)


ARGS = _parse_args(sys.argv[1:])
DEVICE = ARGS.device
CACHE_DIR = "cache"
SUF = f"_{ARGS.tag}" if ARGS.tag else ""
VANILLA_PATH = f"{CACHE_DIR}/dcnv2_vanilla.pt"
RESULT_CSV = f"result_capacity_moe{SUF}.csv"
RESULT_DENSE_CONT_CSV = f"result_dense_cont{SUF}.csv"
HISTORY_JSON = f"{CACHE_DIR}/capacity_moe_history{SUF}.json"

torch.manual_seed(ARGS.seed)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"[config] device={DEVICE} seed={ARGS.seed} K={ARGS.K} top_k={ARGS.top_k} "
      f"lr={ARGS.lr} lr_router={ARGS.lr_router or ARGS.lr} beta2={ARGS.beta2} "
      f"noise_scale={ARGS.noise_scale} "
      f"lb_alpha={ARGS.lb_alpha} batch_size={ARGS.batch_size} "
      f"max_epochs={ARGS.max_epochs} freeze='{ARGS.freeze or 'none'}' "
      f"full_batch_loss={ARGS.full_batch_loss} tag='{ARGS.tag or 'none'}'")
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

# Read the feather once, then build the batch sources for all three splits.
train_set, valid_set, test_set = Split("all")
if ARGS.gpu_resident_data:
    TRAIN_SRC = GpuBatches(train_set, ARGS.batch_size, DEVICE,
                           shuffle=True, seed=ARGS.seed)
    VALID_SRC = GpuBatches(valid_set, ARGS.batch_size, DEVICE, shuffle=False)
    TEST_SRC = GpuBatches(test_set, ARGS.batch_size, DEVICE, shuffle=False)
    print(f"  [data] GPU-resident: train={len(TRAIN_SRC)} valid={len(VALID_SRC)} "
          f"test={len(TEST_SRC)} batches, "
          f"{(TRAIN_SRC.data.numel() + VALID_SRC.data.numel() + TEST_SRC.data.numel()) * 4 / 1e9:.2f} GB")
else:
    TRAIN_SRC, VALID_SRC, TEST_SRC = train_set, valid_set, test_set

dense_test_auc = float(_eval(vanilla, TEST_SRC))
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
if ARGS.reinit_cross:
    # Independent random experts: no upcycled copies, so the symmetry-breaking
    # noise is irrelevant and every expert must learn from scratch.
    for _layer in moe.cross_layers:
        for _e in _layer.experts:
            _e.reset_parameters()
    print("  [reinit-cross] MoE experts randomly re-initialized "
          f"({len(moe.cross_layers)} layers × {ARGS.K} experts)")
print(f"\n  {DCNv2CapacityMoE.param_summary(moe)}")


def routing_sentinel(model, source):
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

    for batch in _iter_infer(model, source):
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


def _train_dense_continued(vanilla, train_src, valid_src, test_src,
                           lr, beta2, batch_size, num_workers, max_epochs,
                           device, result_csv, freeze="", full_batch=False,
                           reinit_cross=False):
    """Continue-train the vanilla dense checkpoint (arm A', dense-continued).

    Mirrors the MoE training loop exactly (same batch source/optimizer/epochs,
    same ``freeze`` groups, same loss weighting), minus the load-balance loss —
    so the only difference from arm B is the architecture itself. Runs the full
    ``max_epochs`` (no early-stop) to expose the complete valid-AUC trajectory.

    Returns (test_auc, best_valid_auc, best_epoch, history).
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    # Arm A' must train: the checkpoint was loaded read-only for arm A, so
    # re-enable grads and let `freeze` decide what actually updates.
    vanilla.requires_grad_(True)
    if reinit_cross:
        # Same treatment as the MoE arm: Cross layers start from scratch.
        for _i in range(3):
            vanilla.layers[_i].reset_parameters()
        print("  [dense-cont] [reinit-cross] Cross layers 0-2 "
              "randomly re-initialized")
    freeze_info = apply_freeze(vanilla, freeze)
    if freeze:
        print("  [dense-cont] " + freeze_summary(freeze_info).replace(
            "\n", "\n  [dense-cont] "))
    optimizer = torch.optim.AdamW(
        trainable_parameters(vanilla), lr=lr, betas=(0.9, beta2))
    auc_best = 0.0
    best_state = None
    best_epoch = 0
    hist = []
    gpu_resident = isinstance(train_src, GpuBatches)
    with open(result_csv, "w") as f:
        f.write("epoch,valid_auc,wall_clock_sec\n")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        if gpu_resident:
            loader = train_src           # reshuffles on every __iter__
        else:
            loader = torch.utils.data.DataLoader(
                Dataset(train_src), batch_size=batch_size,
                num_workers=num_workers, shuffle=True, pin_memory=True)
        for batch in tqdm(loader, desc=f"[dense-cont] Epoch {epoch}"):
            vanilla.train()
            if not gpu_resident:
                for field_name in fields.all:
                    batch[field_name] = batch[field_name].to(
                        device, non_blocking=True).int()
            tab_batch = batch["tab"]
            if full_batch:
                # 'sample' weighting sums to the full-batch mean, so one big
                # forward/backward is equivalent (DRIVERS.md §2) and much faster.
                vanilla(batch)
                loss = criterion(batch["logit"], batch["is_click"].float())
                loss.backward()
            else:
                for s in tab_batch.unique():
                    mask = tab_batch == s
                    sub = {k: v[mask] for k, v in batch.items()}
                    vanilla(sub)
                    loss = scenario_loss(
                        criterion, sub["logit"], sub["is_click"].float(),
                        mask, tab_batch, "sample")
                    loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        valid_auc = float(_eval(vanilla, valid_src))
        wall = time.time() - t0
        print(f"  [dense-cont] Epoch {epoch} valid AUC={valid_auc:.4f} "
              f"wall={wall:.1f}s")
        with open(result_csv, "a") as f:
            f.write(f"{epoch},{valid_auc:.4f},{wall:.1f}\n")
        hist.append({"epoch": epoch, "valid_auc": valid_auc,
                     "wall_clock_sec": wall})
        if valid_auc > auc_best:
            # Exact argmax: a 0.001 deadband here would silently keep epoch 1
            # when the arm improves steadily in small steps.
            auc_best = valid_auc
            best_state = deepcopy(vanilla.state_dict())
            best_epoch = epoch
    if best_state is not None:
        vanilla.load_state_dict(best_state)
        print(f"  [dense-cont] restored best state "
              f"(valid AUC={auc_best:.4f} @ epoch {best_epoch})")
    test_auc = float(_eval(vanilla, test_src))
    return test_auc, auc_best, best_epoch, hist


# ===========================================================
#  Step 3: train with per-scenario sample weighting + lb loss
# ===========================================================
print()
print("=" * 60)
print("Step 3: train DCNv2CapacityMoE")
print("=" * 60)

criterion = torch.nn.BCEWithLogitsLoss()

# Freeze identically to arm A' (fairness), then give the freshly initialized
# routers their own LR: upcycled experts carry pretrained weights (small LR),
# a random router needs a larger one to escape the uniform deadlock.
freeze_info_moe = apply_freeze(moe, ARGS.freeze)
if ARGS.freeze:
    print(freeze_summary(freeze_info_moe))

LR_ROUTER = ARGS.lr_router if ARGS.lr_router is not None else ARGS.lr
_router_params, _other_params = [], []
for _n, _p in moe.named_parameters():
    if not _p.requires_grad:
        continue
    (_router_params if ".router." in _n else _other_params).append(_p)
_param_groups = [{"params": _other_params, "lr": ARGS.lr}]
if _router_params:
    _param_groups.append({"params": _router_params, "lr": LR_ROUTER})
optimizer = torch.optim.AdamW(_param_groups, lr=ARGS.lr, betas=(0.9, ARGS.beta2))
print(f"  [optim] {len(_other_params)} tensors @ lr={ARGS.lr}, "
      f"{len(_router_params)} router tensors @ lr={LR_ROUTER}")
device = torch.device(DEVICE)

# ===========================================================
#  Arm A' (optional): continue-train dense baseline
# ===========================================================
dense_cont_test_auc = None
dense_cont_best_valid = None
dense_cont_history = []
if ARGS.train_dense_ref:
    print()
    print("=" * 60)
    print(f"Arm A': continue-train dense ({ARGS.max_epochs} epochs)")
    print("=" * 60)
    (dense_cont_test_auc, dense_cont_best_valid, _,
     dense_cont_history) = _train_dense_continued(
        vanilla, TRAIN_SRC, VALID_SRC, TEST_SRC,
        lr=ARGS.lr, beta2=ARGS.beta2, batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers, max_epochs=ARGS.max_epochs,
        device=device, result_csv=RESULT_DENSE_CONT_CSV,
        freeze=ARGS.freeze, full_batch=ARGS.full_batch_loss,
        reinit_cross=ARGS.reinit_cross)
    print(f"  [dense-cont] test AUC (all): {dense_cont_test_auc:.4f}")

history = []
auc_best = 0.0
auc_stop_ref = 0.0
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
    if isinstance(TRAIN_SRC, GpuBatches):
        loader = TRAIN_SRC               # reshuffles on every __iter__
    else:
        loader = torch.utils.data.DataLoader(
            Dataset(TRAIN_SRC), batch_size=ARGS.batch_size,
            num_workers=ARGS.num_workers, shuffle=True, pin_memory=True,
        )
    lb_total_epoch = 0.0
    total_steps = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        moe.train()
        if not isinstance(TRAIN_SRC, GpuBatches):
            for field_name in fields.all:
                batch[field_name] = batch[field_name].to(
                    device, non_blocking=True).int()
        tab_batch = batch["tab"]

        if ARGS.full_batch_loss:
            # One full-batch forward/backward. For 'sample' weighting this is
            # the same objective (DRIVERS.md §2, R_gain≈0.999) with 8× larger
            # kernels, and the load-balance loss becomes a *global* statistic —
            # which is what Switch Transformer's aux loss is defined on.
            moe(batch)
            loss = criterion(batch["logit"], batch["is_click"].float())
            lb_loss = batch.get("_load_balance_loss",
                                torch.zeros((), device=device))
            loss = loss + lb_loss
            lb_total_epoch += float(lb_loss.detach())
            loss.backward()
        else:
            for s in tab_batch.unique():
                mask = tab_batch == s
                sub = {k: v[mask] for k, v in batch.items()}

                moe(sub)
                loss = scenario_loss(
                    criterion, sub["logit"], sub["is_click"].float(),
                    mask, tab_batch, "sample",
                )
                lb_loss = sub.get("_load_balance_loss",
                                  torch.zeros((), device=device))
                # BUGFIX: the per-scenario BCE is scaled by |B_s|/|B| so the
                # sum equals the full-batch mean, but lb_loss used to be added
                # UNSCALED once per scenario — i.e. amplified ~8× (one per
                # scenario) relative to the intended lb_alpha. Scale it the
                # same way so lb_alpha means what it says.
                lb_w = mask.sum().to(loss.dtype) / tab_batch.numel()
                loss = loss + lb_loss * lb_w
                lb_total_epoch += float((lb_loss * lb_w).detach())

                loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        total_steps += 1

    valid_auc = float(_eval(moe, VALID_SRC))
    sentinel = routing_sentinel(moe, TEST_SRC)
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

    if valid_auc > auc_best:
        # Exact argmax for the reported best state ...
        auc_best = valid_auc
        best_state = deepcopy(moe.state_dict())
        best_epoch = epoch
    if valid_auc > auc_stop_ref + 0.001:
        # ... while early-stop keeps its historical 0.001 deadband.
        auc_stop_ref = valid_auc
    elif epoch <= ARGS.min_epochs or epoch <= ARGS.warmup_epochs:
        # Protected phase: do not early-stop during warmup or before min_epochs,
        # so the MoE gets enough steps to recover from upcycle perturbation.
        print(f"  [no-early-stop] epoch {epoch} protected "
              f"(min_epochs={ARGS.min_epochs}, warmup_epochs={ARGS.warmup_epochs})")
    else:
        print(f"  [early-stop] valid AUC no +0.001 gain over "
              f"{auc_stop_ref:.4f} (best {auc_best:.4f} @ epoch "
              f"{best_epoch}); stop")
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

test_auc = float(_eval(moe, TEST_SRC))
final_sentinel = routing_sentinel(moe, TEST_SRC)
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
      f"Δ_smoke={test_auc - dense_test_auc:+.4f}")
if dense_cont_test_auc is not None:
    print(f"  dense-cont test AUC (arm A'): {dense_cont_test_auc:.4f}  "
          f"Δ_necessity={test_auc - dense_cont_test_auc:+.4f}")
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
            "top_k": ARGS.top_k, "lr": ARGS.lr, "lr_router": LR_ROUTER,
            "beta2": ARGS.beta2,
            "noise_scale": ARGS.noise_scale, "lb_alpha": ARGS.lb_alpha,
            "batch_size": ARGS.batch_size, "max_epochs": ARGS.max_epochs,
            "warmup_epochs": ARGS.warmup_epochs, "min_epochs": ARGS.min_epochs,
            "freeze": ARGS.freeze, "full_batch_loss": ARGS.full_batch_loss,
            "gpu_resident_data": ARGS.gpu_resident_data,
            "reinit_cross": ARGS.reinit_cross,
            "tag": ARGS.tag,
        },
        "log_k": LOG_K,
        "entropy_pass_threshold": LOG_K - 0.15,
        "dense_test_auc": dense_test_auc,
        "test_auc": test_auc,
        "dense_cont_test_auc": dense_cont_test_auc,
        "dense_cont_best_valid": dense_cont_best_valid,
        "dense_cont_history": dense_cont_history,
        "delta_necessity": (test_auc - dense_cont_test_auc)
                           if dense_cont_test_auc is not None else None,
        "best_valid_auc": auc_best,
        "best_epoch": best_epoch,
        "final_sentinel": final_sentinel,
        "verdict": verdict,
        "history": history,
    }, f, indent=2)

print(f"\nDone! Outputs:")
print(f"  {RESULT_CSV}     — per-epoch AUC / entropy / divergence / lb / wall")
print(f"  {HISTORY_JSON} — full history + frozen config + verdict")
