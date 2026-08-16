#!/usr/bin/env bash
# 阶段四：特征级表征用法对照实验
#
# 目的：回答「上游子任务调制的收益能否传导到下游表征用法」。
#      对每类模型，比较 调制后 backbone vs 未调制 backbone 在特征级用法上的下游 AUC。
#
# 配置选取原则：只用逐格 3 种子判定中**成立**的配置，不在变号单元上浪费算力。
#   全路由   最优成立配置：router=suppress expert=encourage (Δ=+0.0031)
#   部分路由 最优成立配置：router=suppress expert=none shared=encourage (Δ=+0.0068)
#
# 用法: bash scripts/run_stage4.sh <cuda:0|cuda:1>
#   cuda:0 : 全路由   调制 + 未调制对照
#   cuda:1 : 部分路由 调制 + 未调制对照
set -u
DEVICE=${1:-cuda:0}
SEED=${2:-42}
MOD=experiments/run_moe_subtask_modulation.py
REP=experiments/run_moe_representation_usage.py
LOGD=logs
mkdir -p "$LOGD"
TAG=$(echo "$DEVICE" | tr ':' '_')
LOG="$LOGD/stage4_${TAG}.log"

if [ "$DEVICE" = "cuda:0" ]; then
  MODEL=fully-routed; RM=suppress; EM=encourage; SM=none
else
  MODEL=partial-shared; RM=suppress; EM=none; SM=encourage
fi

BB="cache/subtask_backbone_${MODEL}_r${RM}_e${EM}_s${SM}_seed${SEED}.pt"
if [ ! -f "$BB" ]; then
  echo "=== [落盘调制 backbone] $MODEL r=$RM e=$EM s=$SM ===" >> "$LOG"
  python "$MOD" --model "$MODEL" --router-mode "$RM" --expert-mode "$EM" \
    --shared-mode "$SM" --seed "$SEED" --device "$DEVICE" --epochs 1 \
    --save-backbone >> "$LOG" 2>&1
fi

echo "=== [特征级: 调制后 backbone] $MODEL ===" >> "$LOG"
python "$REP" --model "$MODEL" --backbone modulated \
  --router-mode "$RM" --expert-mode "$EM" --shared-mode "$SM" \
  --seed "$SEED" --device "$DEVICE" >> "$LOG" 2>&1

echo "=== [特征级: 未调制对照] $MODEL ===" >> "$LOG"
python "$REP" --model "$MODEL" --backbone pretrain \
  --seed "$SEED" --device "$DEVICE" >> "$LOG" 2>&1

echo "STAGE4 DONE: $DEVICE $MODEL" >> "$LOG"
