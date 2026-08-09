#!/usr/bin/env bash
# ============================================================================
# checkpoint_commit.sh -- periodic checkpoint of a long-horizon watch:
#   1. re-run the doc/metric regeneration hook (so committed docs are not stale)
#   2. append a timestamped entry to the run journal
#   3. stage a WHITELIST of paths and commit (never push, never amend)
#
# Usage:
#   bash .codebuddy/skills/long-run-watch/scripts/checkpoint_commit.sh \
#        --note "R3 eval done 30/36; s/it drifted 3.1->13.9 (eval contention)" \
#        [--subject "watch(s2): tick 6 checkpoint"] [--no-refresh] [--dry-run]
#
# Env:
#   WATCH_NAME          journal/section label (default: default)
#   WATCH_JOURNAL       journal file (default: docs/run_journal/<WATCH_NAME>.md);
#                       set to /dev/null to skip journaling for this checkpoint
#   WATCH_REFRESH_CMD   regeneration hook; empty string disables it
#                       (default: python scripts/OLMoE/collect_eval_matrix.py --patch-docs)
#   WATCH_COMMIT_PATHS  space-separated pathspec whitelist
#                       (default: "docs scripts .codebuddy")
#
# Git safety: no push, no --amend, no config change, no hook skipping. Pushing
# is a separate, explicit, user-requested action.
# ============================================================================
set -u

REPO=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "[ckpt] ERROR: not a git repo"; exit 1; }
cd "$REPO" || exit 1

NAME="${WATCH_NAME:-default}"
JOURNAL="${WATCH_JOURNAL:-docs/run_journal/${NAME}.md}"
REFRESH="${WATCH_REFRESH_CMD-python scripts/OLMoE/collect_eval_matrix.py --patch-docs}"
PATHS="${WATCH_COMMIT_PATHS:-docs scripts .codebuddy}"

NOTE=""; SUBJECT=""; DO_REFRESH=1; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --note)       NOTE="${2:-}"; shift 2 ;;
        --subject)    SUBJECT="${2:-}"; shift 2 ;;
        --no-refresh) DO_REFRESH=0; shift ;;
        --dry-run)    DRY=1; shift ;;
        *) echo "[ckpt] unknown arg: $1"; exit 2 ;;
    esac
done
TS=$(date '+%F %T')
[ -z "$SUBJECT" ] && SUBJECT="watch(${NAME}): checkpoint ${TS}"

# ------------------------------------------------------------------ refresh --
if [ "$DO_REFRESH" -eq 1 ] && [ -n "$REFRESH" ]; then
    echo "[ckpt] refresh: $REFRESH"
    if eval "$REFRESH" >/tmp/watch_refresh_${NAME}.log 2>&1; then
        tail -3 /tmp/watch_refresh_${NAME}.log | sed 's/^/  /'
    else
        echo "  WARN: refresh failed (rc=$?), see /tmp/watch_refresh_${NAME}.log; committing anyway"
        tail -5 /tmp/watch_refresh_${NAME}.log | sed 's/^/  /'
    fi
fi

# ------------------------------------------------------------------ journal --
if [ -n "$NOTE" ]; then
    mkdir -p "$(dirname "$JOURNAL")"
    if [ ! -f "$JOURNAL" ]; then
        {
            echo "# Run journal -- ${NAME}"
            echo
            echo "Append-only log written by \`.codebuddy/skills/long-run-watch\`."
            echo "One bullet per checkpoint: observation + interpretation, newest at the bottom."
            echo
        } > "$JOURNAL"
    fi
    echo "- **${TS}** ${NOTE}" >> "$JOURNAL"
    echo "[ckpt] journal += $JOURNAL"
fi

# ------------------------------------------------------------------- commit --
# Pre-existing staged work (the user's own, from before the watch) must never be
# swept into an automated commit. Everything below is therefore scoped to
# $PATHS: `git add -- $PATHS` then `git commit -- $PATHS`, which produces a
# partial commit and leaves the rest of the index exactly as it was.
PRE_OUT=$(git diff --cached --name-only 2>/dev/null | grep -vE "^($(echo $PATHS | tr ' ' '|'))" | head -10)
if [ -n "$PRE_OUT" ]; then
    echo "[ckpt] NOTE: pre-existing staged files outside the watch scope stay untouched:"
    echo "$PRE_OUT" | sed 's/^/    /'
fi

# shellcheck disable=SC2086
git add -A -- $PATHS 2>/dev/null

# shellcheck disable=SC2086
if git diff --cached --quiet -- $PATHS; then
    echo "[ckpt] nothing changed in scope [$PATHS] -- no commit this checkpoint"
    exit 0
fi

echo "[ckpt] to commit (scope: $PATHS):"
# shellcheck disable=SC2086
git diff --cached --stat -- $PATHS | tail -15 | sed 's/^/  /'

if [ "$DRY" -eq 1 ]; then
    echo "[ckpt] --dry-run: not committing"
    exit 0
fi

BODY="Automated checkpoint of a long-running job watch."
[ -n "$NOTE" ] && BODY="$NOTE"
# shellcheck disable=SC2086
git commit -q -m "$SUBJECT" -m "$BODY" -- $PATHS && \
    echo "[ckpt] committed: $(git --no-pager log -1 --oneline)"
echo "[ckpt] NOT pushed (push only on explicit request)"
