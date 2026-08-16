#!/usr/bin/env bash
# 子任务调制 分流 worker：在指定设备上跑一批指定配置，已有产物自动跳过。
#
# 用法:
#   bash scripts/run_subtask_worker.sh <device> <model> <seed> "rm:em:sm rm:em:sm ..."
# 例:
#   bash scripts/run_subtask_worker.sh cuda:0 partial-shared 42 "suppress:none:none suppress:none:encourage"
#
# 设计要点:
#   - 每个配置起独立进程，单配置崩溃不影响整条链（set -u 但不 set -e）。
#   - 产物 cache/subtask_modulation_<model>_r<rm>_e<em>_s<sm>_seed<seed>.json 已存在则跳过，
#     使得两张卡可安全并行、任务可重复触发而不重算。
set -u
DEVICE=${1:-cuda:0}
MODEL=${2:-partial-shared}
SEED=${3:-42}
CONFIGS=${4:-}
PY=experiments/run_moe_subtask_modulation.py
LOGD=logs
mkdir -p "$LOGD"
TAG=$(echo "$DEVICE" | tr ':' '_')
MASTER="$LOGD/worker_${MODEL}_${TAG}.log"

for cfg in $CONFIGS; do
  rm_=${cfg%%:*}
  rest=${cfg#*:}
  em_=${rest%%:*}
  sm_=${rest#*:}
  OUT="cache/subtask_modulation_${MODEL}_r${rm_}_e${em_}_s${sm_}_seed${SEED}.json"
  if [ -f "$OUT" ]; then
    echo "=== SKIP (exists) router=$rm_ expert=$em_ shared=$sm_ ===" >> "$MASTER"
    continue
  fi
  echo "=== [$MODEL @ $DEVICE] router=$rm_ expert=$em_ shared=$sm_ ===" >> "$MASTER"
  python "$PY" --model "$MODEL" \
    --router-mode "$rm_" --expert-mode "$em_" --shared-mode "$sm_" \
    --seed "$SEED" --device "$DEVICE" --epochs 1 >> "$MASTER" 2>&1
done
echo "WORKER DONE: $MODEL $DEVICE seed $SEED" >> "$MASTER"
