#!/usr/bin/env bash
# ============================================================================
# watch_tick.sh -- ONE tick of a long-horizon watch loop.
#
# A tick = block for SLEEP seconds in the FOREGROUND, then print one compact
# status snapshot and the delta against the previous tick. Blocking in the
# foreground is the whole point: it suspends the agent turn without handing
# control back to the user, so an hours-long job can be followed inside a
# single conversation.
#
# Usage:
#   bash .codebuddy/skills/long-run-watch/scripts/watch_tick.sh [SLEEP_SEC]
#   SLEEP_SEC defaults to 600; 0 (or --no-sleep) probes immediately.
#
# Configure probes via env vars (all optional):
#   WATCH_NAME        label for this watch; picks the state file (default: default)
#   WATCH_PROC_PAT    pgrep -f pattern for the tracked processes
#                     (default: 'training/main.py|infer_native_moe')
#   WATCH_LOGS        space-separated log files; the last tqdm-style progress
#                     token of each is extracted
#   WATCH_COUNTS      semicolon-separated 'label=glob' pairs, counted by
#                     shell globbing, e.g. 'a1=out/**/eval/results-*.json'
#   WATCH_CMD         extra shell command whose stdout is appended verbatim
#   WATCH_MAX_SLEEP   hard cap per tick (default 900) -- keeps each tool call
#                     comfortably under any harness timeout
#   WATCH_STATE_DIR   state directory (default /tmp/long_run_watch)
#
# Exit code is always 0; the caller reads the snapshot, never the status.
# ============================================================================
set -u
shopt -s nullglob globstar 2>/dev/null || true
# Ignore SIGPIPE: a caller piping this output through `head` would otherwise
# kill the script mid-run and skip the state write, corrupting the next tick's
# delta. (Piping is discouraged anyway -- the output is already compact.)
trap '' PIPE

SLEEP="${1:-600}"
[ "$SLEEP" = "--no-sleep" ] && SLEEP=0
case "$SLEEP" in ''|*[!0-9]*) SLEEP=600 ;; esac
MAX_SLEEP="${WATCH_MAX_SLEEP:-900}"
[ "$SLEEP" -gt "$MAX_SLEEP" ] && SLEEP="$MAX_SLEEP"

NAME="${WATCH_NAME:-default}"
PROC_PAT="${WATCH_PROC_PAT:-training/main.py|infer_native_moe}"
LOGS="${WATCH_LOGS:-}"
COUNTS="${WATCH_COUNTS:-}"
EXTRA_CMD="${WATCH_CMD:-}"
STATE_DIR="${WATCH_STATE_DIR:-/tmp/long_run_watch}"
mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/${NAME}.state"
HIST="$STATE_DIR/${NAME}.history"

if [ "$SLEEP" -gt 0 ]; then
    echo "[tick] sleeping ${SLEEP}s ... (started $(date '+%F %T'))"
    sleep "$SLEEP"
fi

TS=$(date '+%F %T')
TICK=$(( $(cat "$STATE_DIR/${NAME}.n" 2>/dev/null || echo 0) + 1 ))
echo "$TICK" > "$STATE_DIR/${NAME}.n"

CUR=""   # canonical "key=value" lines used for delta detection

# Freshest watched log, computed up-front so the process probe can tell
# "job died" apart from "WATCH_PROC_PAT is wrong".
MIN_LOG_AGE=999999
LOG_DONE=0      # a watched log reached N/N or reported a clean exit
for f in ${WATCH_LOGS:-}; do
    [ -f "$f" ] || continue
    a=$(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0) ))
    [ "$a" -lt "$MIN_LOG_AGE" ] && MIN_LOG_AGE=$a
    _t=$(tail -c 4000 "$f" 2>/dev/null | tr '\r' '\n')
    grep -qE 'exits successfully|Training completed|100%\|' <<< "$_t" && LOG_DONE=1
    _p=$(grep -oE '[0-9]+/[0-9]+ \[' <<< "$_t" | tail -1 | tr -d ' [')
    [ -n "$_p" ] && [ "${_p%%/*}" = "${_p##*/}" ] && LOG_DONE=1
done

echo "===== TICK #$TICK  $TS  (name=$NAME) ====="

# ---------------------------------------------------------------- processes --
echo "--- proc ---"
# Sorted oldest-first: the real ranks outlive their dataloader workers, which
# respawn every epoch. A naive pid-ordered listing shows the workers and hides
# the ranks. Self, parent shell and the watch script are filtered out -- the
# caller's own command line usually contains the pattern too.
PROC_N=0
PROC_LINES=""
_pids=$(pgrep -f "$PROC_PAT" 2>/dev/null | grep -vE "^($$|$PPID)\$" | paste -sd, -)
if [ -n "$_pids" ]; then
    PROC_LINES=$(ps -o pid=,etimes=,etime=,args= -p "$_pids" 2>/dev/null \
                 | grep -vE 'watch_tick\.sh|WATCH_PROC_PAT' | sort -k2 -nr)
    PROC_N=$(printf '%s\n' "$PROC_LINES" | grep -c '[^[:space:]]')
fi
if [ "$PROC_N" -gt 0 ]; then
    printf '%s\n' "$PROC_LINES" | head -6 | while read -r pid _ et rest; do
        echo "  pid=$pid elapsed=$et ${rest:0:100}"
    done
    [ "$PROC_N" -gt 6 ] && echo "  ... and $((PROC_N - 6)) more (dataloader workers etc.)"
fi
if [ "$PROC_N" -eq 0 ]; then
    echo "  (no process matching: $PROC_PAT)"
    if [ "$LOG_DONE" -eq 1 ]; then
        echo "  >>> job FINISHED: log reached N/N or reported a clean exit -- verify the output artifacts, then stop the watch."
    elif [ "$MIN_LOG_AGE" -lt 180 ]; then
        echo "  >>> BAD PATTERN: a watched log was written ${MIN_LOG_AGE}s ago, so the job is ALIVE."
        echo "      Fix WATCH_PROC_PAT before concluding anything (check: ps -o cmd= -p \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader))."
    else
        echo "  >>> job appears DOWN (no process, no fresh log output)"
    fi
fi
echo "  proc_count=$PROC_N"
# Delta key is liveness, not the raw count: worker respawns would otherwise
# masquerade as progress and suppress stall detection.
CUR+="proc_alive=$([ "$PROC_N" -gt 0 ] && echo 1 || echo 0)"$'\n'

# ---------------------------------------------------------------------- gpu --
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- gpu ---"
    while IFS=, read -r idx used util; do
        idx=$(echo "$idx" | tr -d ' '); used=$(echo "$used" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
        echo "  gpu$idx mem=${used}MiB util=${util}%"
        CUR+="gpu${idx}_busy=$([ "${used:-0}" -gt 2000 ] && echo 1 || echo 0)"$'\n'
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
fi

# ----------------------------------------------------------------- counters --
if [ -n "$COUNTS" ]; then
    echo "--- counters ---"
    IFS=';' read -r -a _pairs <<< "$COUNTS"
    for pair in "${_pairs[@]}"; do
        [ -z "$pair" ] && continue
        label="${pair%%=*}"; pat="${pair#*=}"
        # Expand into an array and test existence. Do NOT pipe `ls` here: with
        # nullglob an unmatched glob vanishes, `ls -1d --` then lists the cwd
        # and the counter silently reports a phantom 1.
        _arr=(); eval "_arr=($pat)" 2>/dev/null
        n=0
        for _e in ${_arr[@]+"${_arr[@]}"}; do [ -e "$_e" ] && n=$((n + 1)); done
        echo "  $label=$n"
        CUR+="count_${label}=$n"$'\n'
    done
fi

# --------------------------------------------------------------------- logs --
if [ -n "$LOGS" ]; then
    echo "--- logs ---"
    for f in $LOGS; do
        if [ ! -f "$f" ]; then echo "  $(basename "$f"): MISSING"; continue; fi
        age=$(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0) ))
        prog=$(tail -c 4000 "$f" 2>/dev/null | tr '\r' '\n' \
               | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+[^]]*' | tail -1)
        err=$(tail -c 4000 "$f" 2>/dev/null \
              | grep -aoE 'Traceback|OutOfMemoryError|CUDA error|Killed|nan|RuntimeError' | tail -1)
        echo "  $(basename "$f"): last_write=${age}s ago${prog:+ | $prog}${err:+ | ALERT:$err}"
        CUR+="log_$(basename "$f")=${prog:-none}"$'\n'
    done
fi

# -------------------------------------------------------------- extra probe --
if [ -n "$EXTRA_CMD" ]; then
    echo "--- extra ---"
    eval "$EXTRA_CMD" 2>&1 | sed 's/^/  /' | head -40
fi

# --------------------------------------------------------------- delta/stall --
echo "--- delta ---"
if [ -f "$STATE" ]; then
    changed=$(diff <(echo "$CUR") "$STATE" 2>/dev/null | grep '^<' | sed 's/^< //')
    if [ -z "$changed" ]; then
        STALL=$(( $(cat "$STATE_DIR/${NAME}.stall" 2>/dev/null || echo 0) + 1 ))
        echo "  NO CHANGE since last tick (consecutive stalls: $STALL)"
        [ "$STALL" -ge 3 ] && echo "  >>> STALL WARNING: 3+ ticks without progress -- diagnose, do not keep sleeping blindly."
    else
        STALL=0
        echo "$changed" | sed 's/^/  changed: /'
    fi
else
    STALL=0
    echo "  (first tick, no baseline)"
fi
echo "$STALL" > "$STATE_DIR/${NAME}.stall"
printf '%s' "$CUR" > "$STATE"
printf '%s\t%s\n' "$TS" "$(printf '%s' "$CUR" | tr '\n' ' ')" >> "$HIST"

echo "===== END TICK #$TICK ====="
exit 0
