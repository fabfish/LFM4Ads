"""Two-site preflight: prove that a second machine measures the SAME thing.

Why this exists
---------------
From 2026-08-17 the repo is worked by two independent sites (site A: 2 GPUs,
site B: 3 GPUs). A paired difference (moe - dense) is only meaningful if both
arms saw the same data, the same split and comparable numerics. Cross-site
numbers must therefore NEVER be paired directly (see
docs/20260817-1400-两端协作分离设计与实验分工.md §2). What CAN be shared is a
verdict, provided each site independently establishes:

  C1  identical dataset bytes                (feather sha256, first 64MB + size)
  C2  identical vocab / field config         (fields json sha256)
  C3  identical split sample counts          (train/valid/test)
  C4  identical macro scenario set           (8 scenarios, frozen)
  C5  comparable numerics                    (torch / cuda / gpu name / tf32)
  C6  reproducible dense baseline            (optional, needs one run)

C1-C5 are seconds-cheap and MUST pass before any site starts a matrix.
C6 requires one dense run and is checked by --check-baseline once available.

Usage:
    python scripts/verify/preflight_site.py --site B
    python scripts/verify/preflight_site.py --site B --check-baseline
"""

import argparse
import hashlib
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

#: frozen reference fingerprint produced by site A on 2026-08-17.
#: A second site must reproduce EVERY value here.
REFERENCE = {
    "dataset_size_bytes": 6005554962,
    "sample_counts": {"train": 255474457, "valid": 34034748, "test": 32769180},
    "macro_scenarios": [0, 1, 2, 3, 4, 5, 6, 8],
    "lightweight_total_params": 869525,     # dense, dim=330, ID tables dropped
    "moe_total_params": 869750,             # K=5, +225 router params
    "dense_baseline_macro": {               # E10 dense arm, test macro AUC
        "42": 0.7305821830038761, "123": 0.7296135286708358,
        "456": 0.7301065975683463, "789": 0.7311998278504229,
    },
    #: pre-registered tolerance for calling a baseline "reproduced"
    "baseline_tol": 5e-4,
}


def sha256_head(path, nbytes=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="site id, e.g. A or B")
    ap.add_argument("--check-baseline", action="store_true",
                    help="also verify the dense baseline runs (needs runs done)")
    args = ap.parse_args()

    checks, report = [], {"site": args.site}

    ds = os.environ.get("LFM_DATASET", os.path.join(ROOT, "dataset.feather"))
    vocab = os.environ.get("LFM_VOCAB_JSON", "")
    counts_json = os.environ.get("LFM_SAMPLE_COUNTS_JSON", "")
    out_dir = os.environ.get("LFM_MACRO_OUT", "")
    report["env"] = {"LFM_DATASET": ds, "LFM_VOCAB_JSON": vocab,
                     "LFM_SAMPLE_COUNTS_JSON": counts_json,
                     "LFM_MACRO_OUT": out_dir}

    # ---- C1 dataset bytes -------------------------------------------------
    if not os.path.exists(ds):
        checks.append(("C1 dataset exists", False, f"missing {ds}"))
    else:
        size = os.path.getsize(ds)
        report["dataset_size_bytes"] = size
        report["dataset_sha256_head64mb"] = sha256_head(ds)
        ok = size == REFERENCE["dataset_size_bytes"]
        checks.append(("C1 dataset size", ok,
                       f"{size} vs ref {REFERENCE['dataset_size_bytes']}"))

    # ---- C2 vocab ---------------------------------------------------------
    if vocab and os.path.exists(vocab):
        report["vocab_sha256"] = sha256_file(vocab)
        with open(vocab) as f:
            v = json.load(f)
        checks.append(("C2 vocab json readable", True,
                       f"user={len(v.get('user', {}))} "
                       f"video={len(v.get('video', {}))}"))
    else:
        checks.append(("C2 vocab json", False,
                       "LFM_VOCAB_JSON unset or missing (27K runs REQUIRE it)"))

    # ---- C3 sample counts -------------------------------------------------
    if counts_json and os.path.exists(counts_json):
        with open(counts_json) as f:
            c = {k: int(v) for k, v in json.load(f).items()}
        report["sample_counts"] = c
        ok = c == REFERENCE["sample_counts"]
        checks.append(("C3 sample counts", ok, f"{c}"))
    else:
        checks.append(("C3 sample counts json", False,
                       "LFM_SAMPLE_COUNTS_JSON unset — the split sentinel will "
                       "compare against 1K counts and abort every run"))

    # ---- C4 scenario set + C5 numerics + param counts ---------------------
    try:
        import torch
        from experiments.main_macro_auc import MACRO_SCENARIOS, build
        ok = list(MACRO_SCENARIOS) == REFERENCE["macro_scenarios"]
        checks.append(("C4 macro scenarios", ok, f"{list(MACRO_SCENARIOS)}"))
        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["gpus"] = [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())]
        report["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        report["cudnn_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        checks.append(("C5 numerics recorded", True,
                       f"torch={report['torch']} cuda={report['cuda']} "
                       f"gpus={len(report['gpus'])} tf32={report['tf32_matmul']}"))
        d, di = build("dense", 5, True, "cpu")
        m, mi = build("moe", 5, True, "cpu", top_k=2)
        report["lightweight_total_params"] = di["total_params"]
        report["moe_total_params"] = mi["total_params"]
        ok = (di["total_params"] == REFERENCE["lightweight_total_params"]
              and mi["total_params"] == REFERENCE["moe_total_params"])
        checks.append(("C4b param counts", ok,
                       f"dense={di['total_params']} moe={mi['total_params']} "
                       f"(router={mi['router_params']})"))
    except Exception as exc:
        checks.append(("C4/C5 import model", False, repr(exc)))

    # ---- C6 dense baseline reproduction ----------------------------------
    if args.check_baseline:
        tol = REFERENCE["baseline_tol"]
        found = 0
        for seed, ref in REFERENCE["dense_baseline_macro"].items():
            p = os.path.join(out_dir or "", f"run_b0repro_dense_s{seed}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                got = json.load(f)["test"]["macro"]
            found += 1
            checks.append((f"C6 dense baseline seed={seed}",
                           abs(got - ref) < tol,
                           f"{got:.6f} vs ref {ref:.6f} "
                           f"(diff {got - ref:+.6f}, tol {tol})"))
        if not found:
            checks.append(("C6 dense baseline", False,
                           "no run_b0repro_dense_s*.json found yet"))

    print(f"=== preflight site {args.site} ===")
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    all_ok = all(ok for _, ok, _ in checks)
    report["checks"] = [{"name": n, "ok": o, "detail": d} for n, o, d in checks]
    report["all_ok"] = all_ok

    dst = os.path.join(ROOT, "cache", f"preflight_site_{args.site}.json")
    with open(dst, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n{'ALL CHECKS PASS' if all_ok else 'PREFLIGHT FAILED — do not start any matrix'}")
    print(f"wrote {dst}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
