#!/usr/bin/env bash
# D14: close G-1 (no variance estimate). Multi-seed at the optimum
# (V1 rx-only, K=4, lr=5e-4) and at the V2-soft best, for a fair
# variance-aware head-to-head.
set -u
cd /root/Documents/LFM4Ads || exit 1
DEV="$1"
launch() {
    local tag="$1"; shift; local script="$1"; shift
    echo "[$(date '+%H:%M:%S')] START $tag on $DEV"
    setsid nohup python "$script" "$DEV" "$@" --tag "$tag" > "cache/lfm4ads_${tag}.log" 2>&1 < /dev/null &
    local pid=$!; echo "[$(date '+%H:%M:%S')] $tag PID=$pid"
    while ps -p "$pid" > /dev/null 2>&1; do sleep 30; done
    echo "[$(date '+%H:%M:%S')] DONE  $tag"
}
V1_RX="--freeze dnn,head,sparse --skip-downstream --K 4 --lr 5e-4"
V2_SOFT="--freeze dnn,head,sparse --skip-downstream --K 4 --top-k-target 4 --lb-alpha 0.01 --lr 5e-4"
if [ "$DEV" = "cuda:0" ]; then
    launch d14_v1_s1 main_moe.py $V1_RX --seed 1
    launch d14_v1_s7 main_moe.py $V1_RX --seed 7
    launch d14_v2_s1 main_moe_v2.py $V2_SOFT --seed 1
else
    launch d14_v1_s2024 main_moe.py $V1_RX --seed 2024
    launch d14_v2_s7    main_moe_v2.py $V2_SOFT --seed 7
    launch d14_v2_s2024 main_moe_v2.py $V2_SOFT --seed 2024
fi
echo "[$(date '+%H:%M:%S')] D14 CHAIN COMPLETE on $DEV"
