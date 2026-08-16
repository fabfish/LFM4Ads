#!/usr/bin/env bash
# D18: close D9-A — every MoE experiment so far used --skip-downstream, so the
# three-level downstream usage (Feature/Module/Model) was never evaluated for
# the winning rx-only regime. Run it at the established optimum.
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
if [ "$DEV" = "cuda:0" ]; then
    # winning regime, WITH downstream (no --skip-downstream)
    launch d18_v1_rx_downstream main_moe.py \
        --freeze dnn,head,sparse --K 4 --lr 5e-4 --shuffle --seed 42
else
    launch d18_v2_soft_downstream main_moe_v2.py \
        --freeze dnn,head,sparse --K 4 --top-k-target 4 --lb-alpha 0.01 \
        --lr 5e-4 --shuffle --seed 42
fi
echo "[$(date '+%H:%M:%S')] D18 CHAIN COMPLETE on $DEV"
