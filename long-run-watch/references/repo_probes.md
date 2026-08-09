# Repo-specific probes for `trace_prototype`

Concrete probe recipes, failure signatures, and refresh entrypoints for this
repository. Load when watching an OLMoE continual-learning training or eval run.

## 1. Where runs live

| Kind | Path pattern |
|---|---|
| seqFT | `outputs_prototype/OLMoE-1B-7B-0924/cl/seqft/<run>/` |
| A1 adatask | `outputs_prototype/OLMoE-1B-7B-0924/cl/adatask/adatask/<run>/` |
| single-task IFT | `outputs_prototype/OLMoE-1B-7B-0924/cl/ift/<task>/<run>/` |
| eval results | `<run>/eval/results-<round>-<task>.json` |
| weights | `<run>/<round>/` — **written only after ALL epochs finish** |

Task order (R0..R5): `MeetingBank Py150 NumGLUE-cm NumGLUE-ds 20Minuten C-STANCE`.

Weights land in one `save_model` call at the very end. A job killed mid-way
leaves `data_cache/`, `train_command.sh`, `train.log` and **no weight dir** —
there is no resume path, so a killed run must be retrained from scratch. Always
report "no `N/` dir yet" as *expected* until the final step, not as a failure.

## 2. Ready-made `WATCH_*` settings

**Process identity:** training ranks run as
`<conda>/python3.1 -u training/main.py --local_rank=N ...` — one process per
GPU, launched by DeepSpeed. The repo-root `train.py` is *not* what executes, so
`pgrep -f train.py` returns 0 for a perfectly healthy run. Match
`training/main.py`. Eval workers run `inference/infer_native_moe.py`.

Training run (single task):

```bash
export WATCH_NAME=ift_numglue_cm
export WATCH_PROC_PAT='training/main.py'
export WATCH_LOGS='outputs_prototype/OLMoE-1B-7B-0924/cl/ift/NumGLUE-cm/<run>/train.log'
export WATCH_COUNTS='weights=outputs_prototype/OLMoE-1B-7B-0924/cl/ift/NumGLUE-cm/<run>/0'
```

Eval matrix fill:

```bash
export WATCH_NAME=s2_eval
export WATCH_PROC_PAT='infer_native_moe|eval_matrix.sh'
export WATCH_COUNTS='a1=outputs_prototype/OLMoE-1B-7B-0924/cl/adatask/adatask/<run>/eval/results-*.json;ift=outputs_prototype/OLMoE-1B-7B-0924/cl/ift/*/*/eval/results-*.json'
export WATCH_CMD='tail -3 /tmp/s2_driver.log'
```

## 3. Reading `train.log`

- tqdm writes with `\r`; always pipe through `tr '\r' '\n'` before `tail`/`grep`.
- Progress token: `605/939 [1:53:48<1:02:11, 3.11s/it]` → done/total, elapsed,
  ETA, per-step time.
- Total steps = `samples / (micro_bs * n_gpu) * epochs`; changing GPU count
  changes the denominator, so step totals are **not** comparable across runs.
- `s/it` drifting upward (e.g. 3.1 → 13.9) means GPU contention, not a hang.
  Check whether an eval worker moved onto the same device.

## 4. Failure signatures

| Signature in log / system | Diagnosis |
|---|---|
| progress bar stops mid-run, **no** Traceback, **no** OOM text, `last -x reboot` shows no reboot | external `SIGKILL` (someone freed the GPU); retrain whole run |
| `torch.OutOfMemoryError: Tried to allocate ... GiB` + another PID holding memory | train/eval contention on the same device; move to free GPUs or wait |
| `Loss: 0.0000` repeated | **normal** for short-answer tasks (NumGLUE); confirm by sampling early steps for a `0.11 → 0.008 → 0.0000` trajectory before calling it a bug |
| process alive, GPU util 0%, log unchanged for many ticks | real hang; capture `py-spy dump` / stack before killing |
| `pgrep` finds nothing but the log is seconds fresh | wrong `WATCH_PROC_PAT` (see §2), **not** a dead job |

**The box is shared.** `nvidia-smi` regularly shows jobs owned by other users
(e.g. `/home/xjx/...` evaluation scripts on GPU0). Resolve ownership with
`ps -o user=,cmd= -p <pid>` before treating a GPU as free or a memory holder as
one of ours — never kill a PID that is not ours.

## 5. Refresh + gate entrypoints

```bash
python scripts/OLMoE/collect_eval_matrix.py --patch-docs      # matrices + progress into docs
python scripts/OLMoE/_cl_metrics.py --run <run> --json docs/.../S2_gate_metrics.json
python scripts/OLMoE/summarize_fourway.py                      # four-way comparison
```

`collect_eval_matrix.py` resolves the NumGLUE-cm IFT run dynamically (newest run
containing a `0/` weight dir) and derives the completion denominator from the
checkpoints that actually exist, so a finished retrain is absorbed with no code
change. Prefer extending that resolution logic over hardcoding new run ids.

**Axis convention when regenerating any matrix:** horizontal axis is always
`R0..R5` (round / checkpoint); the vertical axis is the evaluated task. `Rx`
must never appear on the vertical axis. Reading one row left-to-right is that
task's forgetting curve.

## 6. Existing background drivers (do not duplicate)

| Script | Role |
|---|---|
| `scripts/OLMoE/_s2_two_phase_driver.sh` | self-healing filler: scans missing eval cells, dispatches to free GPUs, per-cell + per-GPU locks, `MAX_AGE` relaunch |
| `scripts/OLMoE/_eval_monitor_loop.sh` | 30-min loop; auto-runs `--patch-docs` when complete, guarded by a sentinel file |
| `scripts/OLMoE/_gpu_pool.sh` | sourceable worker pool pinning one job per GPU |

Check whether one of these is already running (`pgrep -af s2_driver`,
`tail /tmp/s2_driver.log`) **before** launching anything. Racing a driver is how
the GPU-contention OOMs happen.

**Known blind spots in `_s2_two_phase_driver.sh`** (as of 2026-08-09): its
`ift_missing()` list hardcodes five tasks and omits `NumGLUE-cm`, its
`ckpt_dir()` resolves with `ls -d ... | head -1`, i.e. the **oldest** run dir
(for NumGLUE-cm that is the dead `20260724_135527` with no weights), and its
completion target is a hardcoded `/30`. A freshly retrained NumGLUE-cm ckpt is
therefore never dispatched. Fix by mirroring `collect_eval_matrix.py`'s
newest-run-with-weights resolution — but **never edit the file while the driver
is running**: bash reads a script incrementally by byte offset, so rewriting it
in place can make the live loop execute garbage. Kill and relaunch, or wait for
it to exit.

## 7. Launch + git conventions

Detached launch that survives the session:

```bash
setsid nohup bash scripts/OLMoE/<script>.sh >/tmp/<name>.log 2>&1 < /dev/null &
```

Never launch a GPU job in the foreground of a tool call — only `watch_tick.sh`
blocks. Working branch is `icml-moe-test`. Commit subjects follow
`type(scope): summary` (`docs(exp):`, `fix(moe):`, `feat(olmoe):`); watch
checkpoints use `watch(<name>):`.
