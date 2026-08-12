#!/usr/bin/env bash
# 启动 GPU0 占位守护（长驻后台，退出 shell 也不中断）
cd "$(dirname "$0")/.." || exit 1
LOG=logs/gpu_keeper.log
PIDFILE=logs/gpu_keeper.pid

# cron 等精简环境里 PATH 可能找不到带 torch 的 python，这里固定用 conda 解释器
if [ -x /opt/conda/envs/torch-base/bin/python3 ]; then
  PYTHON="/opt/conda/envs/torch-base/bin/python3"
else
  PYTHON="$(command -v python3)"
fi
export PYTHON

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "gpu_keeper 已在运行 (pid=$(cat "$PIDFILE"))，log=$LOG"
  exit 0
fi

# 单实例保护：若监测进程已在运行则跳过。
# 注意只匹配监测主进程（命令行以 gpu_keeper.py 结尾、不带 --fake 的），
# 否则会把“假计算子进程”也算进去。
if pgrep -f "gpu_keeper.py$" >/dev/null 2>&1; then
  echo "gpu_keeper 监测已在运行，log=$LOG"
  exit 0
fi

# 支持通过环境变量覆盖默认配置，例如：
#   GK_IDLE_THRESHOLD_S=300 GK_FAKE_DURATION_S=0 ./scripts/start_gpu_keeper.sh
setsid nohup env "$@" "$PYTHON" scripts/gpu_keeper.py >> "$LOG" 2>&1 &
echo "gpu_keeper 已启动，log=$LOG"
echo "查看日志: tail -f $LOG"
