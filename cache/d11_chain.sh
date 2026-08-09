#!/usr/bin/env bash
# D11 dual-GPU chain scheduler.
# Each GPU runs a serial chain of jobs; the next job starts as soon as the
# previous PID exits.  Usage:  bash cache/d11_chain.sh cuda:0
set -u
cd /root/Documents/LFM4Ads || exit 1

DEV="$1"
SLOT="${DEV//[^0-9]/}"
PIDFILE="/tmp/d11_cuda${SLOT}_job1.pid"

wait_pid() {
    local p="$1"
    while [ -n "$p" ] && ps -p "$p" > /dev/null 2>&1; do sleep 30; done
}

launch() {
    # launch <tag> <script> <args...>
    local tag="$1"; shift
    local script="$1"; shift
    echo "[$(date '+%H:%M:%S')] START $tag on $DEV"
    setsid nohup python "$script" "$DEV" "$@" --tag "$tag" \
        > "cache/lfm4ads_${tag}.log" 2>&1 < /dev/null &
    local pid=$!
    echo "[$(date '+%H:%M:%S')] $tag PID=$pid"
    wait_pid "$pid"
    echo "[$(date '+%H:%M:%S')] DONE  $tag"
}

# wait for the already-running job1 to finish
if [ -f "$PIDFILE" ]; then
    wait_pid "$(cat "$PIDFILE")"
    echo "[$(date '+%H:%M:%S')] job1 finished on $DEV"
fi

COMMON_V2="--freeze dnn,head,sparse --max-epochs 8 --skip-downstream"

if [ "$DEV" = "cuda:0" ]; then
    # chain 0: extrapolate soft-routing to K=8, plus strong-sparse control
    launch d11_topk_K8      main_moe_v2.py $COMMON_V2 --lr 1e-3 --K 8 --top-k-target 8 --lb-alpha 0.01
    launch d11_K8_sparse2   main_moe_v2.py $COMMON_V2 --lr 1e-3 --K 8 --top-k-target 2 --lb-alpha 0.01
else
    # chain 1: soft routing under high-lr/low-beta2, and V1 beta2 replication
    launch d11_topk4_lr3e3_b95 main_moe_v2.py $COMMON_V2 --lr 3e-3 --beta2 0.95 --K 4 --top-k-target 4 --lb-alpha 0.01
    launch d11_v1_rx_lr3e3_b95 main_moe.py    --freeze dnn,head,sparse --skip-downstream --lr 3e-3 --beta2 0.95 --K 4
fi

echo "[$(date '+%H:%M:%S')] CHAIN COMPLETE on $DEV"
