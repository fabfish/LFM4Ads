#!/usr/bin/env bash
# D17: close D9-E — is the K sweep (K=1/2/4/8) real or seed noise?
# Now that --seed works (needs --shuffle), redo at the optimal lr=5e-4.
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
V1="--freeze dnn,head,sparse --skip-downstream --lr 5e-4 --shuffle"
if [ "$DEV" = "cuda:0" ]; then
    launch d17_k1_s42 main_moe.py $V1 --K 1 --seed 42
    launch d17_k1_s1  main_moe.py $V1 --K 1 --seed 1
    launch d17_k2_s42 main_moe.py $V1 --K 2 --seed 42
    launch d17_k2_s1  main_moe.py $V1 --K 2 --seed 1
else
    launch d17_k8_s42 main_moe.py $V1 --K 8 --seed 42
    launch d17_k8_s1  main_moe.py $V1 --K 8 --seed 1
    launch d17_k2_s7  main_moe.py $V1 --K 2 --seed 7
    launch d17_k8_s7  main_moe.py $V1 --K 8 --seed 7
fi
echo "[$(date '+%H:%M:%S')] D17 CHAIN COMPLETE on $DEV"
