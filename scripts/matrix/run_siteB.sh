#!/usr/bin/env bash
# site B launcher (3 GPUs) — docs/20260817-1400-两端协作分离设计与实验分工.md
# Usage: ./scripts/matrix/run_siteB.sh b0repro
#        ./scripts/matrix/run_siteB.sh b1topk
# All five env vars are REQUIRED:
#   * the four 27K ones (omitting LFM_SAMPLE_COUNTS_JSON trips the split
#     sentinel and aborts every run — by design)
#   * LFM_SITE=B, which BOTH tags provenance.site and selects the 3-GPU
#     seed->device map (a seed's arms never split across cards).
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?usage: run_siteB.sh <stage>  (b0repro | b1topk)}"; shift || true
export LFM_DATASET="$PWD/dataset_27k.feather"
export LFM_VOCAB_JSON="$PWD/cache/fields_27k.json"
export LFM_SAMPLE_COUNTS_JSON="$PWD/cache/sample_counts_27k.json"
export LFM_MACRO_OUT="$PWD/cache/macro_auc_27k_siteB"
export LFM_SITE=B
mkdir -p "$LFM_MACRO_OUT" logs/macro_auc_27k_siteB
echo "[preflight] verifying comparability before dispatching anything..."
python scripts/verify/preflight_site.py --site B
exec python scripts/matrix/run_macro_auc_matrix.py --stages "$STAGE" \
  --epochs 20 --patience 10 --budget-hours 8 "$@"
