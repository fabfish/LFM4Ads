#!/usr/bin/env bash
# d10_chain.sh — D10 双卡链式自循环调度器
# 每卡跑完一个 job 自动启动下一个，产物隔离，全部 rx-only / K=4 / seed=42 / max-epochs=8。
# 用法: bash cache/d10_chain.sh cuda:0   |   bash cache/d10_chain.sh cuda:1
set -uo pipefail
cd "$(dirname "$0")/.."
DEV="${1:?need device cuda:0|cuda:1}"

COMMON="--freeze dnn,head,sparse --K 4 --top-k-target 2 --lb-alpha 0.01 \
        --lr 1e-3 --beta2 0.999 --max-epochs 8 --skip-downstream"

run_job() {
  local tag="$1"; shift
  local log="cache/lfm4ads_${tag}.log"
  echo "[$(date '+%H:%M:%S')] START $tag on $DEV -> $log"
  python main_moe_v2.py "$DEV" "$@" --tag "$tag" > "$log" 2>&1 < /dev/null
  echo "[$(date '+%H:%M:%S')] DONE  $tag (exit=$?)"
}

case "$DEV" in
  cuda:0)
    # job1 (beta2_09) 已由外部启动并在跑；此链只补 job2、job3
    wait_pid="$(cat /tmp/d10_cuda0_job1.pid 2>/dev/null || true)"
    if [[ -n "$wait_pid" ]]; then
      echo "[chain0] waiting job1 PID=$wait_pid ..."
      while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
    fi
    run_job d10_beta2_095 --freeze dnn,head,sparse --K 4 --top-k-target 2 \
            --lb-alpha 0.01 --lr 3e-3 --beta2 0.95 --max-epochs 8 --skip-downstream
    run_job d10_topk_3    --freeze dnn,head,sparse --K 4 --top-k-target 3 \
            --lb-alpha 0.01 --lr 1e-3 --beta2 0.999 --max-epochs 8 --skip-downstream
    ;;
  cuda:1)
    wait_pid="$(cat /tmp/d10_cuda1_job1.pid 2>/dev/null || true)"
    if [[ -n "$wait_pid" ]]; then
      echo "[chain1] waiting job1 PID=$wait_pid ..."
      while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
    fi
    run_job d10_lbalpha_10 --freeze dnn,head,sparse --K 4 --top-k-target 2 \
            --lb-alpha 0.10 --lr 1e-3 --beta2 0.999 --max-epochs 8 --skip-downstream
    run_job d10_topk_4     --freeze dnn,head,sparse --K 4 --top-k-target 4 \
            --lb-alpha 0.01 --lr 1e-3 --beta2 0.999 --max-epochs 8 --skip-downstream
    ;;
  *) echo "unknown device $DEV"; exit 1 ;;
esac
echo "[$(date '+%H:%M:%S')] CHAIN $DEV COMPLETE"
