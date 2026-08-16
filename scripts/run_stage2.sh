#!/usr/bin/env bash
# 阶段二调度：公平基准补齐 + 关键调制信号多种子验证
#
# 用法: bash scripts/run_stage2.sh <cuda:0|cuda:1>
#
# cuda:0 链：基准稠密模型同协议从零重训 seed 42 -> 123（补齐公平对照缺口）
# cuda:1 链：关键调制配置在 seed 123/456 上复现 -> 再补基准稠密 seed 456
#
# 关键配置来源（seed 42 交叉网格 36 配置的极值与主效应代表）：
#   全路由:   基线 / 抑制路由 / 抑制路由+促进专家(最优) / 抑制路由+抑制专家
#   部分路由: 基线 / 抑制路由+促进专家(最优) / 抑制路由+促进共享 /
#             纯促进共享(共享主效应) / 促进路由+抑制共享(最差，验证负向协同)
set -u
DEVICE=${1:-cuda:0}
PRE=experiments/run_moe_pretrain_from_scratch.py
W=scripts/run_subtask_worker.sh
LOGD=logs
mkdir -p "$LOGD"
TAG=$(echo "$DEVICE" | tr ':' '_')
MASTER="$LOGD/stage2_${TAG}.log"

FULLY_CFGS="none:none:none suppress:none:none suppress:encourage:none suppress:suppress:none"
PARTIAL_CFGS="none:none:none suppress:encourage:none suppress:none:encourage none:none:encourage encourage:none:suppress"

pretrain_vanilla () {
  local seed=$1
  if [ -f "cache/vanilla_from_scratch_seed${seed}.pt" ]; then
    echo "=== SKIP vanilla from-scratch seed${seed} (exists) ===" >> "$MASTER"
    return
  fi
  echo "=== [vanilla from-scratch] seed=${seed} @ $DEVICE ===" >> "$MASTER"
  python "$PRE" --model vanilla --device "$DEVICE" --seed "$seed" >> "$MASTER" 2>&1
}

if [ "$DEVICE" = "cuda:0" ]; then
  pretrain_vanilla 42
  pretrain_vanilla 123
else
  for s in 123 456; do
    bash "$W" "$DEVICE" fully-routed   "$s" "$FULLY_CFGS"   >> "$MASTER" 2>&1
    bash "$W" "$DEVICE" partial-shared "$s" "$PARTIAL_CFGS" >> "$MASTER" 2>&1
  done
  pretrain_vanilla 456
fi
echo "STAGE2 DONE: $DEVICE" >> "$MASTER"
