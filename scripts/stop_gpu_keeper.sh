#!/usr/bin/env bash
# 停止指定 GPU 的占位守护（默认 GPU0；可传 GPU 编号，如：stop_gpu_keeper.sh 1）
cd "$(dirname "$0")/.." || exit 1
GPU_ID="${1:-0}"
if [ "$GPU_ID" = "0" ]; then
  PIDFILE=logs/gpu_keeper.pid
else
  PIDFILE="logs/gpu_keeper_gpu${GPU_ID}.pid"
fi

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "停止 gpu_keeper (pid=$PID) ..."
    kill "$PID" 2>/dev/null
    # 等待其子进程（假计算程序）一并退出
    sleep 2
  fi
  rm -f "$PIDFILE"
fi

echo "GPU${GPU_ID} gpu_keeper 已停止"
