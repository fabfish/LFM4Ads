#!/usr/bin/env bash
# 阶段三：把完整交叉网格补齐到 3 种子（42/123/456）
#
# 目的：seed42 已有全部 36 配置；阶段二仅验证 18 个关键配置的多种子。
#      本阶段把 36 个单元全部补到 3 种子，使三因子方差分解与逐单元判定可靠。
#
# 分卡策略（按 模型×种子×路由子集 划分，保证两卡永不触碰同一配置，避免竞态）：
#   cuda:0 : 全路由 seed123 + seed456 全部 9 配置；部分路由 seed123 的 路由=不调制/促进 子集
#   cuda:1 : 部分路由 seed456 全部 27 配置；部分路由 seed123 的 路由=抑制 子集
#
# 用法: bash scripts/run_stage3.sh <cuda:0|cuda:1>
set -u
DEVICE=${1:-cuda:0}
W=scripts/run_subtask_worker.sh
MODES="none encourage suppress"

# 生成全路由 9 配置（shared 恒为 none）
fully_all () {
  for rm in $MODES; do for em in $MODES; do echo -n "$rm:$em:none "; done; done
}
# 生成部分路由配置，可限定 router 子集
partial_by_router () {
  for rm in "$@"; do
    for em in $MODES; do for sm in $MODES; do echo -n "$rm:$em:$sm "; done; done
  done
}

if [ "$DEVICE" = "cuda:0" ]; then
  bash "$W" "$DEVICE" fully-routed   123 "$(fully_all)"
  bash "$W" "$DEVICE" fully-routed   456 "$(fully_all)"
  bash "$W" "$DEVICE" partial-shared 123 "$(partial_by_router none encourage)"
else
  bash "$W" "$DEVICE" partial-shared 456 "$(partial_by_router none encourage suppress)"
  bash "$W" "$DEVICE" partial-shared 123 "$(partial_by_router suppress)"
fi
echo "STAGE3 DONE: $DEVICE" >> "logs/stage3_$(echo "$DEVICE" | tr ':' '_').log"
