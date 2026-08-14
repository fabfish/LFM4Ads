#!/usr/bin/env bash
# 决定性实验：cross 层"有学习空间"时，4x 容量到底有没有收益？
#
# 背景（前一轮矩阵结论）：在预训练 cross 层 + 冻结 embedding 下，即使把 top_k 开到
# K=4（4x 容量 + 4x 算力全激活、零稀疏化代价），MoE 仍逐 epoch 稳定落后 dense
# ~0.0004，且两臂都在 epoch 1 达到最优后单调下降 —— 说明 cross 层已收敛、容量不是
# 瓶颈，MoE 无从获益。
#
# 本轮用 --reinit-cross 把两臂的 cross 层都随机重置（保留冻结的预训练 embedding
# + DNN + head），人为制造真实学习空间，这是"额外 cross 容量"唯一可能兑现的场景。
#
# 两个正交维度：
#   top_k=4 → 纯 soft，全部专家永远激活：测"纯容量收益"（无稀疏化代价）
#   top_k=2 → 真实稀疏 dispatch：       测"稀疏 4x 容量收益"
#
# 并行口径（AGENTS.md）：同 seed 全配置同卡。GPU 常驻数据后单 run 仅 ~11s/epoch，
# 每卡 4 配置并发可把 util 打满。

set -u
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-30}
K=4
NS=0.1
LB=0.001

# name|lr|lr_router|top_k|warmup
CONFIGS=(
  "lr2e4_k4|2e-4|1e-3|4|0"
  "lr2e4_k2|2e-4|1e-3|2|2"
  "lr1e3_k4|1e-3|1e-3|4|0"
  "lr1e3_k2|1e-3|1e-3|2|2"
)

launch_seed() {
  local seed=$1 device=$2
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name lr lr_router top_k warmup <<< "$cfg"
    local tag="reinit_s${seed}_${name}"
    local log="logs/capacity_reinit_s${seed}_${name}.log"
    echo "[launch] seed=$seed dev=$device cfg=$name lr=$lr top_k=$top_k warmup=$warmup -> $log"
    python main_moe_capacity.py "$device" \
      --seed "$seed" --K "$K" --top-k "$top_k" \
      --noise-scale "$NS" --lb-alpha "$LB" \
      --lr "$lr" --lr-router "$lr_router" \
      --warmup-epochs "$warmup" \
      --min-epochs "$EPOCHS" --max-epochs "$EPOCHS" \
      --freeze sparse --full-batch-loss --gpu-resident-data \
      --reinit-cross --train-dense-ref \
      --tag "$tag" > "$log" 2>&1 &
    sleep 4
  done
}

echo "=== reinit-cross capacity-benefit matrix ==="
echo "epochs=$EPOCHS K=$K noise=$NS lb=$LB freeze=sparse full_batch=on gpu_resident=on reinit_cross=on"
launch_seed 42 cuda:0
launch_seed 123 cuda:1

echo "=== all launched; waiting ==="
wait
echo "=== matrix complete ==="
