#!/usr/bin/env bash
# Capacity-MoE 修复后必要性重验矩阵（高 GPU 利用率版）
#
# 修复三项：
#   1. --freeze sparse   冻结 84M embedding（97.9% 参数、与 MoE 无关、过拟合元凶）
#   2. lb 按 sample 比例缩放（原实现被 8 个 scenario 重复累加 ~8 倍）
#   3. --full-batch-loss 单次全批前反向（sample 加权等价，kernel 大 8 倍、host sync 少 8 倍）
#
# 并行口径（AGENTS.md）：同一 seed 的全部配置固定同卡，避免配对差引入设备混杂因子。
#   seed 42  → cuda:0
#   seed 123 → cuda:1
# 每卡 3 个配置并发（显存 ~3GB/97GB，单 run GPU util 仅 30-60%，并发可填满）。

set -u
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-12}
K=4
TOP_K=2
NS=0.1
LB=0.001

# config_name|lr|lr_router|warmup
CONFIGS=(
  "lr1e3_w2|1e-3|1e-3|2"
  "lr2e4_w2|2e-4|1e-3|2"
  "lr2e4_w4|2e-4|2e-4|4"
)

launch_seed() {
  local seed=$1 device=$2
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name lr lr_router warmup <<< "$cfg"
    local tag="fix_s${seed}_${name}"
    local log="logs/capacity_fix_s${seed}_${name}.log"
    echo "[launch] seed=$seed device=$device cfg=$name lr=$lr lr_router=$lr_router warmup=$warmup -> $log"
    python experiments/main_moe_capacity.py "$device" \
      --seed "$seed" --K "$K" --top-k "$TOP_K" \
      --noise-scale "$NS" --lb-alpha "$LB" \
      --lr "$lr" --lr-router "$lr_router" \
      --warmup-epochs "$warmup" \
      --min-epochs "$EPOCHS" --max-epochs "$EPOCHS" \
      --freeze sparse --full-batch-loss --train-dense-ref \
      --tag "$tag" > "$log" 2>&1 &
    sleep 5
  done
}

echo "=== necessity re-verification matrix (post-fix) ==="
echo "epochs=$EPOCHS K=$K top_k=$TOP_K noise=$NS lb=$LB freeze=sparse full_batch=on"
launch_seed 42 cuda:0
launch_seed 123 cuda:1

echo "=== all launched; waiting ==="
wait
echo "=== matrix complete ==="
