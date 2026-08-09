#!/usr/bin/env bash
# D13: find the LR floor for the winning V1 rx-only regime (lr=5e-4 was best so far).
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
V1_RX="--freeze dnn,head,sparse --skip-downstream --K 4"
if [ "$DEV" = "cuda:1" ]; then
    launch d13_v1_lr2e4 main_moe.py $V1_RX --lr 2e-4
    launch d13_v1_lr1e4 main_moe.py $V1_RX --lr 1e-4
else
    launch d13_v1_lr3e4 main_moe.py $V1_RX --lr 3e-4
    launch d13_v1_lr5e5 main_moe.py $V1_RX --lr 5e-5
fi
echo "[$(date '+%H:%M:%S')] D13 CHAIN COMPLETE on $DEV"
