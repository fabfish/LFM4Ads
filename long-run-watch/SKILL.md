---
name: long-run-watch
description: Drives hours-long jobs (model training, eval matrix filling, batch inference, benchmarks) to completion inside a single conversation, by explicitly suspending with foreground `sleep` ticks, analysing each segment's results incrementally, and committing docs/metrics to git at a fixed cadence. This skill should be used whenever a task cannot finish within one tool call and the user expects continuous unattended progress — e.g. "盯着这个训练跑完", "等它跑完再汇报", "长程执行不要中断", "挂起等待", "定时更新并提交", "monitor until done", "keep going until the run finishes".
---

# Long-Run Watch

## Purpose

Turn "start the job, then hand control back and ask the user to come back later"
into "start the job, stay alive, report each segment, commit as evidence
accumulates". The agent remains the operator of the run from launch to landing.

## When to use

Use when the work item's wall-clock time exceeds a single tool call: multi-hour
training, GPU eval matrices, long benchmarks, or any watch-until-condition task.
Do not use for jobs finishing in seconds — just run them.

## The contract (non-negotiable)

1. **Never end the turn to wait.** Ending a turn with "I will check back later",
   "让我们等一会儿再看", or a request to be pinged is a failure of this skill.
   The only legitimate stops are: the terminal condition is reached, a decision
   requires the user's authority (spending, destructive ops, research choices),
   or the same failure has defeated three distinct repair attempts.
2. **Suspend in the foreground.** Waiting is done by a blocking `sleep` inside a
   tool call. Never background the wait (`sleep &`) — that returns instantly and
   defeats the mechanism.
3. **Background the job, foreground the watch.** The job itself launches
   detached (`setsid nohup ... &`) so it survives; only the watch blocks.
4. **One tick = one tool call.** Fold sleep, probe, and delta into a single
   command. Never spend a whole tool call on a bare `sleep`.
5. **Every tick produces analysis, not a dump.** Report the delta and what it
   implies, in ≤5 lines. Silence between ticks is not allowed.
6. **Checkpoint to git on a cadence**, so a crash or context loss never costs
   more than one interval of findings.

## Workflow

### Step 0 — Preflight

State the terminal condition **numerically** before sleeping even once (e.g.
"A1 36/36 eval json present AND `<run>/0/` weight dir exists"). A watch without
a machine-checkable exit condition loops forever.

Then check whether the job is already running before starting anything —
`pgrep -af '<pattern>'`, `nvidia-smi`, existing driver logs. Launching a
duplicate is the most common way to cause resource-contention failures.

### Step 1 — Launch detached

```bash
setsid nohup bash <job>.sh >/tmp/<name>.log 2>&1 < /dev/null &
```

Record pid, log path, and expected duration. Verify liveness once (a 30 s tick)
before settling into long ticks — most failures happen in the first minute.

### Step 2 — Tick loop

```bash
WATCH_NAME=<name> \
WATCH_PROC_PAT='<pgrep pattern>' \
WATCH_LOGS='/path/to/train.log' \
WATCH_COUNTS='cells=<glob for output artifacts>' \
bash .codebuddy/skills/long-run-watch/scripts/watch_tick.sh 600
```

`watch_tick.sh` sleeps, then prints processes (oldest-first, so real ranks show
above their dataloader workers), GPU state, artifact counts, progress tokens,
log alerts, and the diff against the previous tick. It tracks consecutive
no-change ticks and raises a stall warning at three. If no process matches while
a watched log is still being written, it says so — that means the pattern is
wrong, not that the job died.

Do not pipe tick output through `head`/`tail`: the output is already compact,
and truncating it hides the delta line that carries the actual finding.

Verify `WATCH_PROC_PAT` against a live process list once per new job type. A
pattern that silently matches nothing turns every subsequent tick into a lie.

Tick budget by phase:

| Phase | Tick |
|---|---|
| just launched / just changed something | 30–60 s |
| steady progress, ETA > 1 h | 600–900 s |
| near the terminal condition | 120–300 s |
| stall suspected | 60 s + active diagnosis |

Keep each tick ≤ 900 s (the script caps it) so no single tool call approaches a
harness timeout; a 4-hour wait is ~20 ticks, which is normal and expected.

### Step 3 — Report each segment

Per tick, report only: progress delta, rate/ETA change, anomalies, and the next
action. Compare against the previous tick rather than restating absolutes. Call
out rate drift (contention), silent logs (hang), and disappeared processes
(kill) explicitly — see `references/repo_probes.md` for signature-to-diagnosis
mappings.

### Step 4 — Checkpoint commit

Every ~6 ticks (≈1 h) **and** at every state transition (a phase finishes, a
failure is diagnosed, a metric lands):

```bash
WATCH_NAME=<name> bash .codebuddy/skills/long-run-watch/scripts/checkpoint_commit.sh \
  --note "R3 eval 30/36; s/it 3.1->13.9 = eval contention, not a hang" \
  --subject "watch(<name>): tick 12, R3 landed"
```

It re-runs the doc/metric regeneration hook, appends the note to
`docs/run_journal/<name>.md`, then stages **and commits within a path
whitelist** (`git commit -- <paths>`), so unrelated work the user had already
staged is listed and left alone rather than swept into an automated commit. Set
`WATCH_COMMIT_PATHS` as narrowly as the checkpoint actually needs. It never
pushes, never amends, never touches git config, and no-ops cleanly when nothing
in scope changed — so calling it on a fixed cadence is safe. Push only when the
user explicitly asks.

### Step 5 — Terminate and settle

On the terminal condition: run the final regeneration, verify the artifacts
actually exist (do not trust the loop's own counter), make a final commit, and
report the whole run as one narrative — total duration, incidents, final
numbers, and what remains open.

## Failure handling inside the loop

Diagnose and repair without leaving the loop. Resource contention → move to a
free device or wait for the holder to exit. Crash → read the tail, fix, relaunch,
reset the tick cadence to 30 s. Stall ≥3 ticks → stop sleeping and investigate
(`py-spy dump`, `nvidia-smi`, file mtimes). Escalate to the user only after
three distinct repair attempts fail, and escalate with a diagnosis, not a
question.

## Anti-patterns

- Ending the turn because "this will take hours".
- `sleep 7200` in one call, or `sleep &`.
- Ticks that print raw logs with no interpretation.
- Committing only at the very end.
- Trusting a monitor's counter instead of checking artifacts on disk.
- Starting a second worker while an existing driver already owns the resource.

## Bundled resources

- `scripts/watch_tick.sh` — one suspend-and-probe tick with delta and stall
  detection; configured entirely through `WATCH_*` environment variables.
- `scripts/checkpoint_commit.sh` — regenerate, journal, stage whitelist, commit.
- `references/repo_probes.md` — this repository's run layout, ready-made
  `WATCH_*` settings, log-reading rules, failure signatures, refresh
  entrypoints, existing background drivers, and git conventions. Read it before
  watching any OLMoE training or eval run.
