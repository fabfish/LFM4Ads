#!/usr/bin/env bash
# 子任务调制 多维交叉网格启动器
#   全路由混合专家:   router(3) × expert(3)                 = 9  配置
#   部分路由加共享:   router(3) × expert(3) × shared(3)     = 27 配置
# 每个配置独立进程（加载 backbone + 1 轮按场景调制训练），串行跑满整条链。
#
# 用法:
#   bash scripts/run_subtask_grid.sh cuda:0 fully-routed 42
#   bash scripts/run_subtask_grid.sh cuda:1 partial-shared 42
set -u
DEVICE=${1:-cuda:0}
MODEL=${2:-fully-routed}
SEED=${3:-42}
MODES="none encourage suppress"
PY=run_moe_subtask_modulation.py
LOGD=logs
mkdir -p "$LOGD"
MASTER="$LOGD/grid_${MODEL}.log"

if [ "$MODEL" = "fully-routed" ]; then
  for rm in $MODES; do
    for em in $MODES; do
      echo "=== [fully-routed] router=$rm expert=$em shared=none ===" >> "$MASTER"
      python "$PY" --model fully-routed \
        --router-mode "$rm" --expert-mode "$em" --shared-mode none \
        --seed "$SEED" --device "$DEVICE" --epochs 1 >> "$MASTER" 2>&1
    done
  done
else
  for rm in $MODES; do
    for em in $MODES; do
      for sm in $MODES; do
        echo "=== [partial-shared] router=$rm expert=$em shared=$sm ===" >> "$MASTER"
        python "$PY" --model partial-shared \
          --router-mode "$rm" --expert-mode "$em" --shared-mode "$sm" \
          --seed "$SEED" --device "$DEVICE" --epochs 1 >> "$MASTER" 2>&1
      done
    done
  done
fi
echo "GRID DONE: $MODEL seed $SEED" >> "$MASTER"
