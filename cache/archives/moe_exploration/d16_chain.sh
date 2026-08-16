#!/usr/bin/env bash
# D16: with a working seed (--shuffle), re-establish the two headline claims
# under variance: (a) the LR peak (5e-4 vs 1e-3), (b) sparse-vs-soft routing.
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
V1B="--freeze dnn,head,sparse --skip-downstream --K 4 --shuffle"
V2B="--freeze dnn,head,sparse --skip-downstream --K 4 --lb-alpha 0.01 --lr 1e-3 --shuffle"
if [ "$DEV" = "cuda:0" ]; then
    # (a) is lr=5e-4 > lr=1e-3 beyond noise? 3 seeds at lr=1e-3
    launch d16_v1_lr1e3_s42   main_moe.py $V1B --lr 1e-3 --seed 42
    launch d16_v1_lr1e3_s1    main_moe.py $V1B --lr 1e-3 --seed 1
    launch d16_v1_lr1e3_s7    main_moe.py $V1B --lr 1e-3 --seed 7
else
    # (b) sparse (top_k=2) vs soft (top_k=4) under variance, 3 seeds sparse
    launch d16_v2_tk2_s42 main_moe_v2.py $V2B --top-k-target 2 --seed 42
    launch d16_v2_tk2_s1  main_moe_v2.py $V2B --top-k-target 2 --seed 1
    launch d16_v2_tk2_s7  main_moe_v2.py $V2B --top-k-target 2 --seed 7
fi
echo "[$(date '+%H:%M:%S')] D16 CHAIN COMPLETE on $DEV"
