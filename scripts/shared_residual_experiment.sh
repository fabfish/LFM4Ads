#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$REPO/logs/shared_residual_experiment.pid"
HOST_LOG="$REPO/logs/shared_residual_experiment_host.log"
START_LOCK="$REPO/logs/shared_residual_experiment_start.lock"
RUNTIME_LOCK="$REPO/logs/shared_residual_experiment_runtime.lock"
RUNNER="$REPO/scripts/run_shared_residual_experiment_matrix.py"

cd "$REPO"
mkdir -p logs

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-}" in
  plan)
    python "$RUNNER" plan --devices cuda:0 cuda:1
    ;;
  preflight)
    if [[ -f cache/audit/shared_residual_continual/shared_residual_experiment_invariants.json ]]; then
      python - <<'PY'
import sys
sys.path.insert(0, "scripts")
from run_shared_residual_experiment_matrix import driver_path, verify_invariants
verify_invariants(driver_path())
print("existing immutable invariant report is current and PASS")
PY
    else
      python scripts/verify_shared_residual_experiment.py --write-report
    fi
    ;;
  start)
    exec 9>"$START_LOCK"
    if ! flock -n 9; then
      echo "another start operation is in progress"
      exit 1
    fi
    if is_running; then
      echo "already running: pid=$(cat "$PID_FILE")"
      exit 1
    fi
    if [[ -e "$HOST_LOG" ]]; then
      echo "refusing to overwrite immutable host log: $HOST_LOG"
      exit 1
    fi
    if [[ ! -f cache/audit/shared_residual_continual/shared_residual_experiment_invariants.json ]]; then
      python scripts/verify_shared_residual_experiment.py --write-report
    fi
    nohup setsid flock -n "$RUNTIME_LOCK" \
      python "$RUNNER" execute --devices cuda:0 cuda:1 \
      >"$HOST_LOG" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "runner exited during startup; inspect $HOST_LOG"
      tail -n 40 "$HOST_LOG" || true
      exit 1
    fi
    echo "started pid=$pid log=$HOST_LOG"
    ;;
  status)
    if is_running; then
      echo "runner=running pid=$(cat "$PID_FILE")"
    else
      echo "runner=not-running"
    fi
    python "$RUNNER" status --devices cuda:0 cuda:1 || true
    if [[ -f "$HOST_LOG" ]]; then
      echo "--- host log tail ---"
      tail -n 20 "$HOST_LOG"
    fi
    ;;
  summarize)
    python scripts/summarize_shared_residual_experiment.py --write-results
    ;;
  stop)
    if ! is_running; then
      echo "runner is not running"
      exit 0
    fi
    pid="$(cat "$PID_FILE")"
    if ! ps -p "$pid" -o args= | grep -q "run_shared_residual_experiment_matrix.py"; then
      echo "refusing to stop pid=$pid because its command does not match the runner"
      exit 1
    fi
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    echo "sent TERM to process group $pid"
    ;;
  *)
    echo "usage: $0 {plan|preflight|start|status|summarize|stop}" >&2
    exit 2
    ;;
esac
