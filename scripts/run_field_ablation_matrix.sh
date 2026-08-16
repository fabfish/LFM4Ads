#!/usr/bin/env bash
# E1 matrix — ID-embedding dead-weight retrain confirmation.
# Pre-registered: docs/20260814-2212-embedding伪瓶颈证伪与特征信息侧第一步实验预注册.md §5
#
# One process per seed → the three arms of a seed always share the same card,
# satisfying the AGENTS.md pairing rule (same seed ⇒ same device).
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-15}
LR=${LR:-1e-3}
mkdir -p logs

launch() {  # seed device
  local seed=$1 dev=$2
  local tag="s${seed}_lr${LR}"
  echo "[launch] seed=$seed device=$dev tag=$tag"
  nohup python experiments/main_field_ablation.py "$dev" \
    --seed "$seed" --lr "$LR" --max-epochs "$EPOCHS" --tag "$tag" \
    > "logs/fieldabl_${tag}.log" 2>&1 &
  echo "  pid=$! log=logs/fieldabl_${tag}.log"
}

launch 42 cuda:0
launch 123 cuda:1
wait
echo "[done] E1 matrix finished"
