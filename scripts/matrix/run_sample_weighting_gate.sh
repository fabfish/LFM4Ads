#!/usr/bin/env bash
# Stage A 样本加权关卡：long-run-watch 托管的薄包装。
# 用法： bash scripts/matrix/run_sample_weighting_gate.sh <run-code>
#       bash scripts/matrix/run_sample_weighting_gate.sh run-all --stage gate
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/run_sample_weighting_gate.py" "$@"
