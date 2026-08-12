#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_keeper_uptime.py — 解析 logs/gpu_keeper.log，输出“假计算程序泡上”的时间线。

用法:
  python3 scripts/gpu_keeper_uptime.py            # 打印全部时段 + 累计时长
  python3 scripts/gpu_keeper_uptime.py --min 60  # 只显示持续 >=60s 的“有效”时段
"""
import re
import os
import sys
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
START = "已启动假计算程序"
TRIG_REAL = "检测到真实 GPU 任务"
TRIG_EXPIRE = "假程序已运行"
TRIG_SELFEXIT = "假程序已自行退出"
MONITOR_START = "监测启动"
SIGNAL = "收到信号"


def _close(cur, ts, reason, sessions):
    sessions.append((cur, ts, reason))
    return None


def parse(log_path):
    sessions = []
    cur = None
    if not os.path.exists(log_path):
        return sessions
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = TS.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

            if cur is not None and SIGNAL in line:
                # 守护被 SIGTERM 终止（如手动 stop / 重启 / cron 兜底清理）
                cur = _close(cur, ts, "被信号终止(守护重启)", sessions)
                continue
            if cur is not None and MONITOR_START in line:
                # 新的监测进程启动，意味着上一个守护已退出，前一轮未正常闭合
                cur = _close(cur, ts, "守护重启(前一轮未正常结束)", sessions)

            if START in line:
                if cur:
                    sessions.append((cur, None, "未记录结束(异常双实例?)"))
                cur = ts
            elif cur is not None and (
                TRIG_REAL in line or TRIG_EXPIRE in line or TRIG_SELFEXIT in line
            ):
                if TRIG_REAL in line:
                    reason = "让位(真任务挂上)"
                elif TRIG_EXPIRE in line:
                    reason = "到期冷却"
                else:
                    reason = "自行退出"
                cur = _close(cur, ts, reason, sessions)
    if cur is not None:
        sessions.append((cur, None, "仍在运行(截至日志末)"))
    return sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="GPU 编号，默认 0")
    ap.add_argument("--min", type=float, default=0, help="只显示持续 >= N 秒的时段")
    args = ap.parse_args()

    log_name = "gpu_keeper.log" if args.gpu == 0 else f"gpu_keeper_gpu{args.gpu}.log"
    log_path = os.path.normpath(os.path.join(_HERE, "..", "logs", log_name))
    sessions = parse(log_path)
    print(f"GPU{args.gpu} 日志: {log_path}")
    total = 0.0
    shown = 0
    for s, e, r in sessions:
        dur = (e - s).total_seconds() if e else None
        if dur is not None:
            total += dur
        if args.min and (dur is None or dur < args.min):
            continue
        shown += 1
        end_str = e.strftime("%m-%d %H:%M:%S") if e else "进行中"
        if dur is None:
            dur_str = "进行中"
        else:
            tag = " [抖动]" if dur < 10 else ""
            dur_str = f"{int(dur)}s ({dur/60:.1f}min){tag}"
        print(f"{s.strftime('%m-%d %H:%M:%S')}  ~  {end_str}   时长 {dur_str}   原因:{r}")

    print(
        f"\n共 {len(sessions)} 段（显示 {shown} 段，阈值>={args.min}s）；"
        f"累计泡上(已结束段) = {int(total)}s = {total/60:.1f} min"
    )


if __name__ == "__main__":
    main()
