#!/usr/bin/env bash
# 停止 GPU0 占位守护（含其假计算子进程）
cd "$(dirname "$0")/.." || exit 1
PIDFILE=logs/gpu_keeper.pid

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

# 兜底：按名字清理残留
if pgrep -f "gpu_keeper.py" >/dev/null 2>&1; then
  pkill -f "gpu_keeper.py"
  echo "已按名称清理残留进程"
fi
echo "done"
