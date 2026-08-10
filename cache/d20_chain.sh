#!/usr/bin/env bash
# D20: does the D18/D19 negative downstream result survive REAL seed variance?
#
# Defect G-5 (found while closing D19): train.py's downstream train()/infer()
# had NO shuffle, so every D18/D19 downstream trial was deterministic given the
# backbone -- the only stochastic source was the head init. The reported
# t-values therefore measure a NARROWER noise source than true run-to-run
# variance. Same class of trap as G-4 (upstream seed no-op).
#
# Fix: train.py infer()/train() now take shuffle=; main_moe.py exposes
# --shuffle-downstream (default OFF so all D18/D19 numbers stay reproducible).
#
# D20 re-runs the rx-only arm -- the ONLY arm that was significant (mean
# -0.00445, t=-2.72) -- with downstream shuffle ON across 3 fresh seeds.
# Decision rule:
#   * if mean Delta stays ~-0.004 and the between-seed sd is small
#       -> the negative transfer result is REAL, D9-A "no" is locked in.
#   * if mean Delta scatters across zero with sd comparable to the effect
#       -> D18/D19 significance was an artefact of understated variance,
#          and the honest claim degrades to "no detectable transfer".
# Either outcome is publishable; the point is that it is currently UNTESTED.
#
# 3 seeds x ~4.5h, split over 2 cards: card0 takes 2, card1 takes 1.
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
COMMON="--freeze dnn,head,sparse --K 4 --lr 5e-4 --shuffle --shuffle-downstream"
if [ "$DEV" = "cuda:0" ]; then
    launch d20_rx_ds_s1    main_moe.py $COMMON --seed 1
    launch d20_rx_ds_s2024 main_moe.py $COMMON --seed 2024
else
    launch d20_rx_ds_s7    main_moe.py $COMMON --seed 7
fi
echo "[$(date '+%H:%M:%S')] D20 CHAIN COMPLETE on $DEV"
