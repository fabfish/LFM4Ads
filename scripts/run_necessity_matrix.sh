#!/usr/bin/env bash
# run_necessity_matrix.sh — capacity-MoE 必要性验证矩阵调度器
#
# 每个 run 内部串行跑三臂：
#   A   dense 冻结参照（只 evaluate）
#   A'  dense-continued（--train-dense-ref 开启，继续训练 16 epoch）
#   B   capacity-MoE optimal（upcycle + warmup 2 + sparse 16 epoch）
# 必要性主指标 Δ_necessity = B_test_best − A'_test_best（逐 seed 同卡配对）。
#
# 并行布局（遵守 AGENTS.md：同 seed 全部臂同卡；不同 seed 分卡）:
#   - GPU 0: seed=42（A'→B 串行，同进程）
#   - GPU 1: seed=123（A'→B 串行，同进程）
#
# 冻结协议（见 docs/20260814-1120-capacity-MoE-必要性验证驱动.md §三）:
#   K=4 top_k=2 noise_scale=0.1 lb_alpha=0.001 warmup=2 min_epochs=16 max_epochs=16
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
mkdir -p logs

run_necessity() {
    local seed="$1"
    local gpu="$2"
    local tag="s${seed}_necessity"
    echo "[$(date '+%F %T')] gpu${gpu} seed=${seed} tag=${tag} (dense-cont + MoE-optimal, 16ep)"
    python main_moe_capacity.py "cuda:${gpu}" \
        --seed "$seed" \
        --K 4 --top-k 2 \
        --noise-scale 0.1 --lb-alpha 0.001 \
        --warmup-epochs 2 --min-epochs 16 --max-epochs 16 \
        --train-dense-ref \
        --tag "$tag" \
        > "logs/capacity_moe_${tag}.log" 2>&1
    echo "[$(date '+%F %T')] gpu${gpu} DONE ${tag} (exit $?)"
}

( run_necessity 42 0 ) > logs/necessity_gpu0.log 2>&1 &
( run_necessity 123 1 ) > logs/necessity_gpu1.log 2>&1 &

echo "necessity matrix launched: gpu0 -> seed 42 ; gpu1 -> seed 123"
echo "per-run log in logs/capacity_moe_s{42,123}_necessity.log"
wait
echo "necessity matrix done"
