#!/usr/bin/env bash
# run_capacity_warmup_matrix.sh — capacity-MoE warmup + 温和 lb 扫描
#
# 扫描 lb_alpha ∈ {0, 0.001, 0.003} × top_k ∈ {1, 2} × seed ∈ {42, 123} = 12 runs。
# 固定 warmup_epochs=2（软路由让 router 特化）、noise_scale=0.1。
# 目的：在「防坍缩（需要 lb）」与「不阻碍特化（lb 不能太大）」之间找平衡点。
#
# 并行布局（同 seed 全部变体同卡）:
#   - GPU 0: seed=42 的 6 个变体（串行）
#   - GPU 1: seed=123 的 6 个变体（串行）
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
mkdir -p logs

WARMUP=2
NOISE=0.1
MAX_EPOCHS=10
MIN_EPOCHS=6

lb_tag() {
    case "$1" in
        0)      echo "0" ;;
        0.001)  echo "1e3" ;;
        0.003)  echo "3e3" ;;
        *)      echo "$1" | sed 's/\.//' ;;
    esac
}

run_variants() {
    local seed="$1"
    local gpu="$2"
    for topk in 1 2; do
        for lb in 0 0.001 0.003; do
            local lbtag
            lbtag="$(lb_tag "$lb")"
            local tag="s${seed}_k${topk}_w${WARMUP}_lb${lbtag}"
            echo "[$(date '+%F %T')] gpu${gpu} seed=${seed} top_k=${topk} lb_alpha=${lb} tag=${tag}"
            python experiments/main_moe_capacity.py "cuda:${gpu}" \
                --top-k "$topk" --noise-scale "$NOISE" --seed "$seed" \
                --warmup-epochs "$WARMUP" --min-epochs "$MIN_EPOCHS" \
                --max-epochs "$MAX_EPOCHS" --lb-alpha "$lb" --tag "$tag" \
                > "logs/capacity_moe_${tag}.log" 2>&1
            echo "[$(date '+%F %T')] gpu${gpu} DONE ${tag} (exit $?)"
        done
    done
}

( run_variants 42 0 ) > logs/capacity_warmup_matrix_gpu0.log 2>&1 &
( run_variants 123 1 ) > logs/capacity_warmup_matrix_gpu1.log 2>&1 &

echo "warmup+lb matrix launched: gpu0 seed42 (6 runs), gpu1 seed123 (6 runs)"
echo "12 runs total; per-run log in logs/capacity_moe_<tag>.log"
wait
echo "warmup+lb matrix done"
