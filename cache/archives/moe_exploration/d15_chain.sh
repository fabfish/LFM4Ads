#!/usr/bin/env bash
# D15: close G-1 for real. Data-order shuffle is the only stochastic source,
# so --shuffle is mandatory for --seed to matter (verified: without it, seeds
# 1/42/2024 give bit-identical AUC 0.7750083012867384).
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
V1="--freeze dnn,head,sparse --skip-downstream --K 4 --lr 5e-4 --shuffle"
V2="--freeze dnn,head,sparse --skip-downstream --K 4 --top-k-target 4 --lb-alpha 0.01 --lr 5e-4 --shuffle"
if [ "$DEV" = "cuda:0" ]; then
    launch d15_v1_s42   main_moe.py $V1 --seed 42
    launch d15_v1_s1    main_moe.py $V1 --seed 1
    launch d15_v1_s7    main_moe.py $V1 --seed 7
    launch d15_v1_s2024 main_moe.py $V1 --seed 2024
else
    launch d15_v2_s42   main_moe_v2.py $V2 --seed 42
    launch d15_v2_s1    main_moe_v2.py $V2 --seed 1
    launch d15_v2_s7    main_moe_v2.py $V2 --seed 7
    launch d15_v2_s2024 main_moe_v2.py $V2 --seed 2024
fi
echo "[$(date '+%H:%M:%S')] D15 CHAIN COMPLETE on $DEV"
