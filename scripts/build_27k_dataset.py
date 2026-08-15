"""Build the KuaiRand-27K feather + vocabulary override for the LFM4Ads pipeline.

Replicates the 1K construction in ``dataset.py:__main__`` against the 27K CSVs
in ``/apdcephfs/private-xavieryu/database/KuaiRand-27K/data/``:

    log_standard_4_08_to_4_21_27k_{part1,part2}.csv   (4/08..4/21)
    log_standard_4_22_to_5_08_27k_{part1,part2}.csv   (4/22..5/08)
    user_features_27k.csv      (user fields)
    video_features_basic_27k.csv (video fields)

Outputs
-------
* ``dataset_27k.feather``  — columns == ``fields.all`` (same order as 1K), all
  feature columns factorized to int32, date/is_click/tab kept for ``Split``.
* ``cache/fields_27k.json`` — ``{"user": {...}, "video": {...}}`` vocabulary
  sizes, consumed by ``fields.py`` via ``LFM_VOCAB_JSON``.

Notes
-----
* Split thresholds (date<20220503 / [03,06) / >=06) are IDENTICAL for 1K and
  27K: both datasets span 20220408..20220508.
* ``time_ms`` is only used for ordering and is dropped from the feather; the
  final columns are exactly ``fields.all``.
* Vocab is the number of distinct codes AFTER factorize (``use_na_sentinel``
  off), so the embedding tables exactly cover the code range.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fields  # noqa: E402  (keys only; vocab comes from the data)

DATA_DIR = "/apdcephfs/private-xavieryu/database/KuaiRand-27K/data"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FEATHER = os.path.join(ROOT, "dataset_27k.feather")
OUT_VOCAB = os.path.join(ROOT, "cache", "fields_27k.json")

LOG_PARTS = [
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
]
LOG_COLS = ["user_id", "video_id", "date", "time_ms", "is_click", "tab"]
LOG_DTYPE = {"user_id": np.int32, "video_id": np.int32, "date": np.int32,
             "time_ms": np.int64, "is_click": np.int32, "tab": np.int32}

USER_COLS = list(fields.user.keys())      # 27 fields, incl. user_id
VIDEO_COLS = list(fields.video.keys())    # 9 fields, incl. video_id


def _t(s):
    return f"{time.time() - s:.0f}s"


def load_logs():
    frames = []
    t0 = time.time()
    for p in LOG_PARTS:
        path = os.path.join(DATA_DIR, p)
        df = pd.read_csv(path, usecols=LOG_COLS, dtype=LOG_DTYPE,
                         engine="c")
        frames.append(df)
        print(f"  read {p}: {len(df):,} rows  [{_t(t0)}]", flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"log total: {len(df):,} rows  [{_t(t0)}]", flush=True)
    return df


def load_features():
    # NOTE: no dtype here on purpose — several columns are categorical strings
    # (user_active_degree, *_range, video_type, upload_dt, upload_type) and
    # music_id exceeds int32. pandas infers them as object/int64 and
    # :func:`factorize` below coerces everything to int32 codes, exactly like
    # the 1K path in dataset.py:__main__.
    t0 = time.time()
    # join keys kept int32 to match the log frame exactly (avoids a merge-time
    # cast on a 322M-row frame); all other columns are inferred then factorized.
    user = pd.read_csv(os.path.join(DATA_DIR, "user_features_27k.csv"),
                       usecols=USER_COLS, dtype={"user_id": np.int32})
    print(f"  user_features: {len(user):,} rows  [{_t(t0)}]", flush=True)
    video = pd.read_csv(os.path.join(DATA_DIR, "video_features_basic_27k.csv"),
                        usecols=VIDEO_COLS, dtype={"video_id": np.int32})
    print(f"  video_features: {len(video):,} rows  [{_t(t0)}]", flush=True)
    return user, video


def main():
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    if os.path.exists(OUT_FEATHER):
        raise SystemExit(f"{OUT_FEATHER} exists — delete it first to rebuild.")

    t0 = time.time()
    print("[1/4] loading logs ...", flush=True)
    df = load_logs()

    print("[2/4] merging features ...", flush=True)
    user, video = load_features()
    df = df.merge(user, on="user_id", how="inner")
    del user
    print(f"  after user merge: {len(df):,} rows  [{_t(t0)}]", flush=True)
    df = df.merge(video, on="video_id", how="inner")
    del video
    print(f"  after video merge: {len(df):,} rows  [{_t(t0)}]", flush=True)

    print("[3/4] factorizing feature columns ...", flush=True)
    vocab = {}
    for field in USER_COLS + VIDEO_COLS:
        codes, uniques = df[field].factorize(use_na_sentinel=False)
        df[field] = codes.astype(np.int32)
        vocab[field] = int(len(uniques))
    print(f"  factorized {len(vocab)} fields  [{_t(t0)}]", flush=True)

    # sanity: tab must fit the 15-way head
    assert int(df["tab"].max()) < 15 and int(df["tab"].min()) >= 0, \
        f"tab out of range: [{df['tab'].min()}, {df['tab'].max()}]"
    assert df.isna().sum().sum() == 0, "unexpected NaN after inner merge"

    print("[4/4] writing feather + vocab ...", flush=True)
    df[fields.all].to_feather(OUT_FEATHER)
    print(f"  wrote {OUT_FEATHER} ({os.path.getsize(OUT_FEATHER) / 1e9:.1f} GB)"
          f"  [{_t(t0)}]", flush=True)

    vocab_json = {"user": {k: vocab[k] for k in USER_COLS},
                  "video": {k: vocab[k] for k in VIDEO_COLS}}
    with open(OUT_VOCAB, "w") as f:
        json.dump(vocab_json, f, indent=2)
    print(f"  wrote {OUT_VOCAB}", flush=True)

    # ---- summary ----
    print("\n=== summary ===")
    print(f"rows        : {len(df):,}")
    print(f"tab values  : {sorted(df['tab'].unique().tolist())}")
    print(f"user_id vocab : {vocab['user_id']:,}")
    print(f"video_id vocab: {vocab['video_id']:,}")
    print(f"author_id vocab: {vocab['author_id']:,}")
    print(f"music_id vocab : {vocab['music_id']:,}")
    for split, q in [("train", "date < 20220503"),
                     ("valid", "20220503 <= date < 20220506"),
                     ("test", "date >= 20220506")]:
        print(f"{split:>5}      : {len(df.query(q)):,}")
    print(f"total wall : {_t(t0)}")


if __name__ == "__main__":
    main()
