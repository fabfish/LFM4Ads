#!/usr/bin/env bash
# 启动 site B 三卡占位守护（默认 GPU0/1/2；可用参数覆盖，如：start_gpu_keeper_siteB.sh 0 1）
# 与 site A 的 start_gpu_keeper.sh 完全隔离：pidfile / 日志均带 siteB 前缀，互不抢锁。
cd "$(dirname "$0")/.." || exit 1

GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0 1 2)
fi

# cron 等精简环境里 PATH 可能找不到带 torch 的 python，这里固定用 conda 解释器
if [ -x /opt/conda/envs/torch-base/bin/python3 ]; then
  PYTHON="/opt/conda/envs/torch-base/bin/python3"
else
  PYTHON="$(command -v python3)"
fi
export PYTHON

for GPU_ID in "${GPUS[@]}"; do
  if [ "$GPU_ID" = "0" ]; then
    LOG="logs/gpu_keeper_siteB.log"
    PIDFILE="logs/gpu_keeper_siteB.pid"
  else
    LOG="logs/gpu_keeper_siteB_gpu${GPU_ID}.log"
    PIDFILE="logs/gpu_keeper_siteB_gpu${GPU_ID}.pid"
  fi

  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "gpu_keeper_siteB GPU${GPU_ID} 已在运行 (pid=$(cat "$PIDFILE"))"
    continue
  fi

  # 最终单实例保护由 Python 进程对每张卡各自的 pidfile 执行 flock；
  # 即使 cron 同时拉起重复进程，重复实例也会立即退出。
  setsid nohup env GK_GPU_ID="${GPU_ID}" \
    GK_IDLE_THRESHOLD_S="${GK_IDLE_THRESHOLD_S:-60}" \
    GK_FAKE_DURATION_S="${GK_FAKE_DURATION_S:-1200}" \
    GK_REST_S="${GK_REST_S:-1200}" \
    GK_HOLD_MEMORY_MB="${GK_HOLD_MEMORY_MB:-2048}" \
    GK_TARGET_UTIL="${GK_TARGET_UTIL:-25}" \
    "$PYTHON" scripts/gpu_keeper_siteB.py >> "$LOG" 2>&1 &
  echo "gpu_keeper_siteB GPU${GPU_ID} 已启动，log=$LOG"
done
