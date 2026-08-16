import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""E1 — retrain confirmation: are the 83,984,250 ID-embedding params dead weight?

Pre-registered in
docs/20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md §5.
Do not change arms, budget or verdict thresholds without amending that doc.

Three from-scratch arms (one process = one seed = one device, so the paired
comparison is same-seed / same-card by construction):

  full    36 fields, all embeddings trainable                    dim=360
  idzero  video_id/author_id/music_id tables zeroed AND frozen   dim=360  <- MAIN
  iddrop  those three fields physically removed                  dim=330

Main endpoint : Δ_id = test_AUC(idzero) − test_AUC(full), same seed.
Noise floor   : 0.001 (inherited from the dense-widened / reinit matrices).

`idzero` is the main control because it is architecturally identical to `full`
(same dim, same cross/DNN/head parameter counts) — the only difference is that
three input channels are pinned to the zero vector. `iddrop` quantifies the
engineering win (params / wall-clock) and is explicitly NOT the main verdict.
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import torch
from torch import nn
from tqdm import tqdm

import fields
from dataset import GpuBatches, Split
from model import DCNv2
from train import evaluate_gpu

CACHE_DIR = "cache"
OUT_DIR = f"{CACHE_DIR}/embedding_capacity"
BIG_ID_FIELDS = ("video_id", "author_id", "music_id")
BIG_ID_PARAMS = 83_984_250
# frozen sample counts from scripts/diagnose_embedding_capacity.py (sentinel S1)
EXPECTED_COUNTS = {"train": 9_281_007, "valid": 1_230_368, "test": 1_201_670}
ARMS = ("full", "idzero", "iddrop")


class SubsetSparse(nn.Module):
    """`model.Sparse` minus a set of fields (arm ``iddrop``)."""

    def __init__(self, drop=()):
        super().__init__()
        self.drop = tuple(drop)
        self.tables = nn.ModuleDict()
        for field, size in (fields.user | fields.video).items():
            if field in self.drop:
                continue
            self.tables[field] = nn.Embedding(size, 10)

    def forward(self, batch):
        return torch.cat([t(batch[f]) for f, t in self.tables.items()], -1)


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="E1 ID-embedding dead-weight test")
    ap.add_argument("device", nargs="?", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs x 20 batches, sentinels only, no verdict")
    return ap.parse_args(argv)


def build(arm, device):
    """Return (model, sentinel_dict) for one arm."""
    if arm == "iddrop":
        n_fields = len(fields.user | fields.video) - len(BIG_ID_FIELDS)
        model = DCNv2(dim=n_fields * 10)
        model.sparse = SubsetSparse(drop=BIG_ID_FIELDS)
        model = model.to(device)
    else:
        model = DCNv2().to(device)
    if arm == "idzero":
        for f in BIG_ID_FIELDS:
            t = model.sparse.tables[f]
            with torch.no_grad():
                t.weight.zero_()
            t.weight.requires_grad_(False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_sparse = sum(p.numel() for n, p in model.named_parameters()
                     if not n.startswith("sparse."))
    return model, {"arm": arm, "dim": model.layers[0].in_features,
                   "total_params": total, "trainable_params": trainable,
                   "non_sparse_params": non_sparse}


def id_channel_maxabs(model, batch):
    """Max |value| over the 30 dims produced by the three ID tables (S3)."""
    if not all(f in model.sparse.tables for f in BIG_ID_FIELDS):
        return None
    with torch.inference_mode():
        vals = [model.sparse.tables[f](batch[f]).abs().max()
                for f in BIG_ID_FIELDS]
    return float(torch.stack(vals).max())


def id_weight_maxabs(model):
    if not all(f in model.sparse.tables for f in BIG_ID_FIELDS):
        return None
    return float(max(model.sparse.tables[f].weight.abs().max()
                     for f in BIG_ID_FIELDS))


def train_arm(model, srcs, args, label, max_batches=None):
    train_src, valid_src, test_src = srcs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_state, best_epoch, hist = -1.0, None, 0, []
    csv = f"result_fieldabl_{label}{'_' + args.tag if args.tag else ''}.csv"
    with open(csv, "w") as f:
        f.write("epoch,valid_auc,wall_clock_sec\n")
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        for i, batch in enumerate(
                tqdm(train_src, desc=f"[{label}] ep{epoch}", leave=False), 1):
            model.train()
            model(batch)
            criterion(batch["logit"], batch["is_click"].float()).backward()
            optimizer.step()
            optimizer.zero_grad()
            if max_batches is not None and i >= max_batches:
                break
        valid_auc = float(evaluate_gpu(model, valid_src))
        wall = time.time() - t0
        print(f"  [{label}] epoch {epoch} valid AUC={valid_auc:.6f} "
              f"wall={wall:.1f}s")
        with open(csv, "a") as f:
            f.write(f"{epoch},{valid_auc:.6f},{wall:.1f}\n")
        hist.append({"epoch": epoch, "valid_auc": valid_auc,
                     "wall_clock_sec": wall})
        if valid_auc > best_auc:  # exact argmax, no dead zone
            best_auc, best_epoch = valid_auc, epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_auc = float(evaluate_gpu(model, test_src))
    return {"test_auc": test_auc, "best_valid_auc": best_auc,
            "best_epoch": best_epoch, "history": hist,
            "mean_wall_sec": sum(h["wall_clock_sec"] for h in hist) / len(hist)}


def main():
    args = _parse_args(sys.argv[1:])
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_json = f"{OUT_DIR}/e1_history{suffix}.json"
    if os.path.exists(out_json) and not args.smoke:
        raise SystemExit(f"{out_json} exists; evidence bundles are immutable.")

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}")

    print(f"[config] device={args.device} seed={args.seed} lr={args.lr} "
          f"batch={args.batch_size} epochs={args.max_epochs} arms={arms} "
          f"smoke={args.smoke}")

    train_set, valid_set, test_set = Split("all")
    counts = {"train": len(train_set), "valid": len(valid_set),
              "test": len(test_set)}
    s1 = counts == EXPECTED_COUNTS
    print(f"[S1] sample counts {counts} expected={EXPECTED_COUNTS} "
          f"-> {'PASS' if s1 else 'FAIL'}")
    if not s1:
        raise SystemExit("S1 FAILED: sample counts differ from frozen baseline")

    srcs = (GpuBatches(train_set, args.batch_size, args.device,
                       shuffle=True, seed=args.seed),
            GpuBatches(valid_set, args.batch_size, args.device, shuffle=False),
            GpuBatches(test_set, args.batch_size, args.device, shuffle=False))
    del train_set, valid_set, test_set
    probe = next(iter(srcs[1]))

    results, sentinels = {}, {"S1_sample_counts": {"ok": s1, "counts": counts}}
    for arm in arms:
        print("=" * 60)
        print(f"Arm {arm}")
        print("=" * 60)
        torch.manual_seed(args.seed)  # identical init stream per arm
        model, info = build(arm, args.device)
        info["id_weight_maxabs_before"] = id_weight_maxabs(model)
        info["id_channel_maxabs_before"] = id_channel_maxabs(model, probe)
        print(f"  params total={info['total_params']:,} "
              f"trainable={info['trainable_params']:,} "
              f"non_sparse={info['non_sparse_params']:,} dim={info['dim']}")
        out = train_arm(model, srcs, args, arm,
                        max_batches=20 if args.smoke else None)
        info["id_weight_maxabs_after"] = id_weight_maxabs(model)
        info["id_channel_maxabs_after"] = id_channel_maxabs(model, probe)
        out.update(info)
        results[arm] = out
        print(f"  [{arm}] test AUC={out['test_auc']:.6f} "
              f"(best valid {out['best_valid_auc']:.6f} @ep{out['best_epoch']})")
        del model
        torch.cuda.empty_cache()

    # ---- sentinels S2/S3/S4/S6 ----
    if "idzero" in results:
        z = results["idzero"]
        sentinels["S2_idzero_weights_zero_and_frozen"] = {
            "ok": z["id_weight_maxabs_before"] == 0.0
                  and z["id_weight_maxabs_after"] == 0.0,
            "maxabs_before": z["id_weight_maxabs_before"],
            "maxabs_after": z["id_weight_maxabs_after"]}
        sentinels["S3_idzero_channel_zero"] = {
            "ok": z["id_channel_maxabs_after"] == 0.0,
            "maxabs_after": z["id_channel_maxabs_after"]}
    if "full" in results and "idzero" in results:
        diff = (results["full"]["trainable_params"]
                - results["idzero"]["trainable_params"])
        sentinels["S4_trainable_param_gap"] = {
            "ok": diff == BIG_ID_PARAMS, "observed": diff,
            "expected": BIG_ID_PARAMS}
        sentinels["S6_non_sparse_params_identical"] = {
            "ok": (results["full"]["non_sparse_params"]
                   == results["idzero"]["non_sparse_params"]),
            "full": results["full"]["non_sparse_params"],
            "idzero": results["idzero"]["non_sparse_params"]}

    deltas = {}
    if "full" in results:
        for arm in ("idzero", "iddrop"):
            if arm in results:
                deltas[f"delta_{arm}_minus_full"] = (
                    results[arm]["test_auc"] - results["full"]["test_auc"])

    bundle = {
        "provenance": {
            "script": "main_field_ablation.py",
            "preregistration":
                "docs/20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md §5",
            "device": args.device, "seed": args.seed, "lr": args.lr,
            "batch_size": args.batch_size, "max_epochs": args.max_epochs,
            "arms": arms, "smoke": args.smoke,
            "noise_floor": 0.001,
            "main_endpoint": "delta_idzero_minus_full (paired, same seed/card)",
        },
        "sentinels": sentinels,
        "results": results,
        "deltas": deltas,
    }
    with open(out_json, "w") as f:
        json.dump(bundle, f, indent=2)

    print("\n" + "=" * 60)
    print("E1 summary" + (" (SMOKE — no verdict)" if args.smoke else ""))
    print("=" * 60)
    for k, v in sentinels.items():
        print(f"  {k}: {'PASS' if v['ok'] else 'FAIL'}")
    for arm in arms:
        r = results[arm]
        print(f"  {arm:<8} test={r['test_auc']:.6f} "
              f"peak_valid={r['best_valid_auc']:.6f} @ep{r['best_epoch']} "
              f"params={r['total_params']:,} wall/ep={r['mean_wall_sec']:.1f}s")
    for k, v in deltas.items():
        print(f"  {k} = {v:+.6f}")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
