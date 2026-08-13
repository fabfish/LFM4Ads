#!/usr/bin/env bash
# run_capacity_matrix.sh — capacity-MoE 并行探索矩阵调度器
#
# 矩阵: noise_scale ∈ {0.01, 0.05, 0.1} × top_k ∈ {1, 2} × seed ∈ {42, 123, 456}
#       = 18 runs（每个 run 内部 early-stop，通常 2~3 epoch）。
#
# 并行布局（遵守 AGENTS.md 并行规则：同 seed 的全部路由模式必须同卡）:
#   - GPU 0: seed=42 的 6 个变体（串行），随后接 seed=456 的 6 个变体（串行）
#   - GPU 1: seed=123 的 6 个变体（串行）
#
# 每个 run 独立写日志 logs/capacity_moe_<tag>.log 与
# result_capacity_moe_<tag>.csv / cache/capacity_moe_history_<tag>.json。
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
mkdir -p logs

NOISE_SCALES="0.01 0.05 0.1"
TOP_KS="1 2"

run_variants() {
    local seed="$1"
    local gpu="$2"
    for topk in $TOP_KS; do
        for ns in $NOISE_SCALES; do
            local nstag
            nstag="$(printf '%.2f' "$ns" | sed 's/\.//')"
            local tag="s${seed}_k${topk}_ns${nstag}"
            echo "[$(date '+%F %T')] gpu${gpu} seed=${seed} top_k=${topk} noise_scale=${ns} tag=${tag}"
            python main_moe_capacity.py "cuda:${gpu}" \
                --top-k "$topk" --noise-scale "$ns" --seed "$seed" --tag "$tag" \
                > "logs/capacity_moe_${tag}.log" 2>&1
            echo "[$(date '+%F %T')] gpu${gpu} DONE ${tag} (exit $?)"
        done
    done
}

# 卡 0: seed 42 → 456（串行）；卡 1: seed 123（串行）
( run_variants 42 0; run_variants 456 0 ) > logs/capacity_matrix_gpu0.log 2>&1 &
( run_variants 123 1 ) > logs/capacity_matrix_gpu1.log 2>&1 &

echo "matrix launched: gpu0 -> seed 42,456 ; gpu1 -> seed 123"
echo "18 runs total; per-run log in logs/capacity_moe_<tag>.log"
wait
echo "matrix done"
