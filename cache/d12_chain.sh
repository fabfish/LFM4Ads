#!/usr/bin/env bash
# D12 chain: push the winning regime (V1 rx-only) — LR sensitivity + beta2
# interaction at the *optimal* lr, and V2-soft vs V1 head-to-head at best lr.
set -u
cd /root/Documents/LFM4Ads || exit 1
DEV="$1"

launch() {
    local tag="$1"; shift
    local script="$1"; shift
    echo "[$(date '+%H:%M:%S')] START $tag on $DEV"
    setsid nohup python "$script" "$DEV" "$@" --tag "$tag" \
        > "cache/lfm4ads_${tag}.log" 2>&1 < /dev/null &
    local pid=$!
    echo "[$(date '+%H:%M:%S')] $tag PID=$pid"
    while ps -p "$pid" > /dev/null 2>&1; do sleep 30; done
    echo "[$(date '+%H:%M:%S')] DONE  $tag"
}

V1_RX="--freeze dnn,head,sparse --skip-downstream --K 4"

if [ "$DEV" = "cuda:1" ]; then
    # Is lr=1e-3 already optimal for the winning V1 rx-only regime?
    launch d12_v1_lr5e4     main_moe.py $V1_RX --lr 5e-4
    launch d12_v1_lr2e3     main_moe.py $V1_RX --lr 2e-3
    # beta2 at the *good* lr: does it help when lr is not overshooting?
    launch d12_v1_lr1e3_b95 main_moe.py $V1_RX --lr 1e-3 --beta2 0.95
else
    # V2 soft routing at the best-known lr, low-lr side (fair head-to-head vs V1)
    launch d12_v2soft_lr5e4 main_moe_v2.py --freeze dnn,head,sparse --skip-downstream \
        --K 4 --top-k-target 4 --lb-alpha 0.01 --lr 5e-4
    launch d12_v2soft_b95   main_moe_v2.py --freeze dnn,head,sparse --skip-downstream \
        --K 4 --top-k-target 4 --lb-alpha 0.01 --lr 1e-3 --beta2 0.95
fi

echo "[$(date '+%H:%M:%S')] D12 CHAIN COMPLETE on $DEV"
