#!/usr/bin/env bash
# 停止 site B 三卡占位守护（默认 GPU0/1/2；可用参数覆盖，如：stop_gpu_keeper_siteB.sh 0 1）
cd "$(dirname "$0")/.." || exit 1

GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0 1 2)
fi

for GPU_ID in "${GPUS[@]}"; do
  if [ "$GPU_ID" = "0" ]; then
    PIDFILE="logs/gpu_keeper_siteB.pid"
  else
    PIDFILE="logs/gpu_keeper_siteB_gpu${GPU_ID}.pid"
  fi

  if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
      echo "停止 gpu_keeper_siteB GPU${GPU_ID} (pid=$PID) ..."
      kill "$PID" 2>/dev/null
      sleep 2
    fi
    rm -f "$PIDFILE"
  fi
  echo "GPU${GPU_ID} gpu_keeper_siteB 已停止"
done
