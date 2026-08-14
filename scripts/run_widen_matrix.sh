#!/usr/bin/env bash
# Dense-widened 4x 容量对照矩阵（决定性实验）
#
# 回答: cross 层容量到底是不是 AUC 瓶颈？
#   臂 A' = dense-continued（1x Cross, reinit）
#   臂 B  = DenseWidenedDCNv2（4x Cross hidden width, 无路由）
# 若 B > A' → 容量是瓶颈，MoE 路由是元凶；若 B ≈ A' → 容量不是瓶颈，MoE 路线关闭。
#
# 并行口径: 同 seed 全配置同卡。GPU 常驻数据后单 run 仅 3.5-6.2s/epoch，
# 每卡 2 配置（lr 2e-4 / 1e-3）并发打满 util。

set -u
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-30}
WIDTH=2

launch_seed() {
  local seed=$1 device=$2
  for lr in 2e-4 1e-3; do
    local name=$(echo "$lr" | tr -d '-')
    local tag="widen_s${seed}_lr${name}"
    local log="logs/widen_s${seed}_lr${name}.log"
    echo "[launch] seed=$seed dev=$device lr=$lr -> $log"
    python main_dense_widened.py "$device" \
      --seed "$seed" --lr "$lr" --width "$WIDTH" \
      --max-epochs "$EPOCHS" --freeze sparse \
      --tag "$tag" > "$log" 2>&1 &
    sleep 4
  done
}

echo "=== dense-widened capacity-control matrix ==="
echo "epochs=$EPOCHS width=$WIDTH freeze=sparse gpu_resident=on full_batch=on"
launch_seed 42 cuda:0
launch_seed 123 cuda:1

echo "=== all launched; waiting ==="
wait
echo "=== matrix complete ==="
