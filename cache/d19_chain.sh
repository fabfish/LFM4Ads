#!/usr/bin/env bash
# D19: D18 showed the rx-only MoE gains NOTHING downstream (mean -0.0027, and
# ModuleUsage -0.0137, the worst). Hypothesis: freezing the embedding table
# forces the Cross experts to over-specialise on the upstream objective,
# producing representations that transfer worse. Test: compare downstream
# transfer across freeze regimes at matched lr.
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
    # full finetune (no freeze) WITH downstream: does the transfer gap close?
    launch d19_v1_full_downstream main_moe.py --K 4 --lr 5e-4 --shuffle --seed 42
else
    # partial freeze (embeddings trainable) WITH downstream
    launch d19_v1_fdh_downstream main_moe.py --freeze dnn,head --K 4 --lr 5e-4 --shuffle --seed 42
fi
echo "[$(date '+%H:%M:%S')] D19 CHAIN COMPLETE on $DEV"
