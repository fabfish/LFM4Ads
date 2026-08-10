#!/usr/bin/env bash
# d20b_chain.sh — D20b 链式调度器（D9-A 冻结强度第三/四档扫描）
#
# ⚠️ 方法学约束（AGENTS.md §2 / DRIVERS.md §4）：
#    D20b 的 full 与 freeze-dnn-head 同 D20 的 rx-only 属同一横向对比
#    （D9-A 冻结强度三档），必须**单卡独占、串行**，不可与在跑 rx-only 混跑。
#    本脚本用于 rx-only 跑完之后，占 cuda:0 单卡串行补完两臂，保证 D20 内部可比。
#
# 用法：bash cache/d20b_chain.sh
set -e

DEVICE=cuda:0
GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F, '{gsub(/ /,"",$2); print $1":"$2"MiB"}')
echo "[d20b $(date +%H:%M:%S)] start on $DEVICE | GPU: $GPUS"

# 第一臂：full（仅冻结 sparse，解冻 dnn+head，从头训）
for S in 1 7 42; do
  echo "[d20b $(date +%H:%M:%S)] FULL seed=$S"
  LFM_NUM_WORKERS=10 python main_moe.py $DEVICE \
      --freeze sparse --K 4 --lr 5e-4 --shuffle --shuffle-downstream \
      --seed $S --tag d20b_full_s$S
done

# 第二臂：freeze-dnn-head（冻结 dnn+head，仅 sparse 可训）
for S in 1 7 42; do
  echo "[d20b $(date +%H:%M:%S)] FREEZE-DNN-HEAD seed=$S"
  LFM_NUM_WORKERS=10 python main_moe.py $DEVICE \
      --freeze dnn,head,sparse --K 4 --lr 5e-4 --shuffle --shuffle-downstream \
      --seed $S --tag d20b_freeze_dnn_head_s$S
done

echo "[d20b $(date +%H:%M:%S)] DONE — D9-A 冻结强度三档（rx-only / full / freeze-dnn-head）闭合"
