#!/usr/bin/env bash
# E15 — model-selection-endpoint sensitivity (docs/20260817-1330-E15-选择端点敏感性预注册.md)
# All four 27K env vars are REQUIRED; omitting LFM_SAMPLE_COUNTS_JSON trips the
# split-conservation sentinel (S1 FAILED) — that is by design, so launch via
# this script rather than by hand.
set -euo pipefail
cd "$(dirname "$0")/../.."
export LFM_DATASET="$PWD/dataset_27k.feather"
export LFM_VOCAB_JSON="$PWD/cache/fields_27k.json"
export LFM_SAMPLE_COUNTS_JSON="$PWD/cache/sample_counts_27k.json"
export LFM_MACRO_OUT="$PWD/cache/macro_auc_27k"
mkdir -p logs/macro_auc_27k
exec python scripts/matrix/run_macro_auc_matrix.py --stages s9sel \
  --epochs 20 --patience 10 --budget-hours 10 "$@"
