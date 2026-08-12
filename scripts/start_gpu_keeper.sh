#!/usr/bin/env bash
# 启动 GPU0 占位守护（长驻后台，退出 shell 也不中断）
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gpu_keeper.log
PIDFILE=logs/gpu_keeper.pid

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "gpu_keeper 已在运行 (pid=$(cat "$PIDFILE"))，log=$LOG"
  exit 0
fi

# 支持通过环境变量覆盖默认配置，例如：
#   GK_IDLE_THRESHOLD_S=300 GK_FAKE_DURATION_S=0 ./scripts/start_gpu_keeper.sh
setsid nohup env "$@" python3 scripts/gpu_keeper.py >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "gpu_keeper 已启动 (pid=$(cat "$PIDFILE"))，log=$LOG"
echo "查看日志: tail -f $LOG"
