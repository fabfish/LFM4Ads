"""E0 — zero-training diagnosis of the 84M embedding table.

Why this runs BEFORE any embedding-widening experiment
------------------------------------------------------
`docs/HANDOFF.md` §2 fixes a hard constraint: *never add capacity before
proving a capacity gap exists*. The cross-layer route died exactly because
4x capacity was handed out for free and bought nothing.

Before spending GPU hours on an "embedding-widened" arm, this script answers
three questions from the data alone (no model, no training):

  Q1  Where do the 84,017,390 embedding parameters actually live?
  Q2  How many of those rows are ever *trainable* (i.e. appear in train),
      and how thin is their supervision (exposures per id)?
  Q3  How much of valid/test traffic hits rows that train never updated
      (OOV) — those rows stay at random init, so extra width there is
      provably worthless.

Outputs a JSON evidence bundle to cache/embedding_capacity/diagnosis.json.
Refuses to overwrite an existing bundle (immutable-evidence rule).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fields  # noqa: E402
from dataset import DATASET_PATH, Split  # noqa: E402

EMBED_DIM = 10
OUT_DIR = "cache/embedding_capacity"
OUT_JSON = f"{OUT_DIR}/diagnosis.json"
BIG_ID_FIELDS = ["video_id", "author_id", "music_id"]


def _param_breakdown():
    vocab = fields.user | fields.video
    total = sum(vocab.values()) * EMBED_DIM
    rows = []
    for field, size in sorted(vocab.items(), key=lambda kv: -kv[1]):
        rows.append({
            "field": field,
            "vocab": size,
            "params": size * EMBED_DIM,
            "share_of_embedding": size * EMBED_DIM / total,
        })
    return total, rows


def _field_stats(train, valid, test, field):
    tr = train[field].to_numpy()
    tr_counts = np.bincount(tr)
    seen = tr_counts > 0
    n_seen = int(seen.sum())
    freq = tr_counts[seen]
    out = {
        "field": field,
        "vocab": int(fields.user.get(field, fields.video.get(field))),
        "unique_in_train": n_seen,
        "rows_never_in_train": int(
            (fields.user.get(field, fields.video.get(field))) - n_seen),
        "exposures_per_seen_id_mean": float(freq.mean()),
        "exposures_per_seen_id_median": float(np.median(freq)),
        "exposures_per_seen_id_p90": float(np.percentile(freq, 90)),
        "seen_ids_with_freq_le_1": float((freq <= 1).mean()),
        "seen_ids_with_freq_le_2": float((freq <= 2).mean()),
        "seen_ids_with_freq_le_5": float((freq <= 5).mean()),
    }
    # traffic share covered by thin ids (train side)
    for thr in (1, 2, 5):
        thin = np.zeros_like(tr_counts, dtype=bool)
        thin[seen] = freq <= thr
        out[f"train_traffic_share_from_ids_freq_le_{thr}"] = float(
            thin[tr].mean())
    # OOV: valid/test rows whose id was never updated during training
    for name, split in (("valid", valid), ("test", test)):
        ids = split[field].to_numpy()
        oov = ~np.isin(ids, np.nonzero(seen)[0])
        out[f"{name}_oov_sample_share"] = float(oov.mean())
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_JSON):
        raise SystemExit(
            f"{OUT_JSON} already exists; evidence bundles are immutable. "
            f"Move it aside explicitly if a re-run is really intended.")

    total_embed_params, breakdown = _param_breakdown()
    train, valid, test = Split("all")
    n_train, n_valid, n_test = len(train), len(valid), len(test)

    per_field = [_field_stats(train, valid, test, f)
                 for f in (list(fields.user) + list(fields.video))]

    big = [r for r in breakdown if r["field"] in BIG_ID_FIELDS]
    big_params = sum(r["params"] for r in big)
    small_params = total_embed_params - big_params

    # trainable-row accounting: only rows seen in train ever get a gradient
    seen_params = sum(
        s["unique_in_train"] * EMBED_DIM for s in per_field)

    bundle = {
        "provenance": {
            "script": "scripts/diagnose/diagnose_embedding_capacity.py",
            "dataset": str(DATASET_PATH),
            "embed_dim": EMBED_DIM,
            "split": "date<20220503 / [20220503,20220506) / >=20220506",
        },
        "sample_counts": {
            "train": n_train, "valid": n_valid, "test": n_test,
            "total": n_train + n_valid + n_test,
        },
        "embedding_params_total": total_embed_params,
        "embedding_params_big_three_ids": big_params,
        "embedding_params_all_other_fields": small_params,
        "big_three_share_of_embedding": big_params / total_embed_params,
        "embedding_params_ever_trainable": seen_params,
        "embedding_params_never_trainable": total_embed_params - seen_params,
        "never_trainable_share": 1 - seen_params / total_embed_params,
        "param_breakdown": breakdown,
        "per_field": per_field,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"samples: train={n_train:,} valid={n_valid:,} test={n_test:,}")
    print(f"embedding params: {total_embed_params:,}")
    print(f"  big-3 ids : {big_params:,} "
          f"({big_params / total_embed_params:.4%})")
    print(f"  all others: {small_params:,}")
    print(f"  ever trainable (row seen in train): {seen_params:,} "
          f"({seen_params / total_embed_params:.2%})")
    print()
    hdr = (f"{'field':<24}{'vocab':>10}{'seen':>10}{'exp/id':>9}"
           f"{'freq<=2':>9}{'test_oov':>10}")
    print(hdr)
    for s in per_field:
        if s["vocab"] < 100:
            continue
        print(f"{s['field']:<24}{s['vocab']:>10,}{s['unique_in_train']:>10,}"
              f"{s['exposures_per_seen_id_mean']:>9.1f}"
              f"{s['seen_ids_with_freq_le_2']:>9.2%}"
              f"{s['test_oov_sample_share']:>10.2%}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
