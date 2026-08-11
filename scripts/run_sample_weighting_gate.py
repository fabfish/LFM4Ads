#!/usr/bin/env python
"""Stage A 样本加权关卡的 run-code 执行器（long-run-watch 托管单元）。

用法：
  python scripts/run_sample_weighting_gate.py run <run-code>
  python scripts/run_sample_weighting_gate.py run-all --stage gate
  python scripts/run_sample_weighting_gate.py run-all --stage moe

行为（见驱动文档 §五/§八/§十）：
  - 每个 run-code 独立 manifest + 独立日志 + 独立产物，禁止覆盖既有产物。
  - 先写 manifest(planned/running)，启动训练并 tee 到 logs/sample_weighting_<code>.log，
    结束后读 summary 填 results，写回 manifest(succeeded/failed)。
  - run-all 串行执行；任一 run 失败（OOM/NaN/缺产物/非零退出）即停止队列（§十）。
  - 不执行 git commit（§四）。
"""

import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO)

import swg_config  # noqa: E402
import swg_config_stageb  # noqa: E402

CACHE = os.path.join(REPO, "cache")
MANIFEST_DIR = os.path.join(CACHE, "manifests", "sample_weighting")
LOGS = os.path.join(REPO, "logs")

os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def run(cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return run(["git", "rev-parse", "HEAD"]), run(["git", "status", "--porcelain"])


def paths(code):
    return {
        "checkpoint": os.path.join(CACHE, f"{code}.pt"),
        "summary": os.path.join(CACHE, f"moe_pretrain_summary_{code}.json"),
        "log": os.path.join(LOGS, f"sample_weighting_{code}.log"),
        "csv": os.path.join(REPO, f"result_{code}.csv"),
    }


def _config_module_for(code):
    if code in swg_config.RUN_CARDS or code in swg_config.APPEND_SEEDS:
        return swg_config
    if code in swg_config_stageb.RUN_CARDS:
        return swg_config_stageb
    return None


def run_one(code):
    mod = _config_module_for(code)
    if mod is None:
        print(f"[ERROR] unknown run-code {code!r}; known: "
              f"{sorted(list(swg_config.RUN_CARDS) + list(swg_config.APPEND_SEEDS)
                        + list(swg_config_stageb.RUN_CARDS))}")
        return 2

    p = paths(code)
    manifest_path = os.path.join(MANIFEST_DIR, f"{code}.json")

    # 不覆盖守卫（§五.4 / §十）：已完成的产物绝不覆盖
    for kind, path in p.items():
        if os.path.exists(path):
            print(f"[ERROR] refuse to overwrite existing {kind}: {path}")
            return 3
    # manifest 恢复：succeeded 绝不覆盖；running/failed 视为崩溃残留，删除后重跑
    if os.path.exists(manifest_path):
        try:
            prev = json.load(open(manifest_path))
        except Exception:
            prev = {}
        if prev.get("status") == "succeeded":
            print(f"[ERROR] refuse to overwrite succeeded manifest: {manifest_path}")
            return 3
        os.remove(manifest_path)

    cmd = mod.build_command(code)
    commit, status = git_info()

    # 解析卡片 → config + 路由语义（审计可追溯性）
    if mod is swg_config and code in swg_config.APPEND_SEEDS:
        card = {"model": "vanilla", "vanilla_per_scenario": True,
                "freeze_router": False, "weighting": "sample",
                "seed": swg_config.APPEND_SEEDS[code]}
    else:
        card = dict(mod.RUN_CARDS[code])
    C = mod.COMMON
    router = card.get("router")
    freeze_router = bool(card.get("freeze_router", False))
    if freeze_router or router == "frozen":
        router_semantics = ("frozen: router weights zeroed (uniform, "
                            "zero-noise, vanilla-equiv)")
    elif router == "soft":
        router_semantics = "soft: learnable DataRouter (data routing)"
    elif router == "none":
        router_semantics = "none: dense, no router"
    else:
        router_semantics = "noisy_routing: trainable router + gating noise"
    config = {
        "model": card["model"],
        "vanilla_per_scenario": bool(card.get("vanilla_per_scenario", False)),
        "freeze_router": freeze_router,
        "router": router,
        "router_semantics": router_semantics,
        "scenario_loss_weighting": card["weighting"],
        "device": C["device"], "seed": card["seed"],
        "batch_size": C["batch_size"], "lr": C["lr"],
        "beta2": C["beta2"], "shuffle": C["shuffle"],
        "K": C["K"], "routing": C["routing"],
    }
    if card["model"] == "lowrank-full-dim":
        config["rank"] = card.get("rank", 360 // (2 * C["K"]))
    manifest = {
        "run_code": code,
        "status": "running",
        "git_commit": commit,
        "git_status": status,
        "command": " ".join(cmd),
        "config": config,
        "timing": {"start": datetime.datetime.now().isoformat(),
                   "end": None, "exit_code": None},
        "paths": p,
        "results": {"pooled_auc": None, "mean_per_scenario_auc": None,
                    "per_scenario_auc": None, "anomaly": None},
        "checks": {"equal_vs_sample_invariant": "see equ-swg marker"},
        "provenance": {
            "config_hash": hashlib.sha256(
                json.dumps(cmd, sort_keys=True).encode()).hexdigest(),
            "source_hashes": {
                "train.py": sha256_file(os.path.join(REPO, "train.py")),
                "run_moe_pretrain_from_scratch.py":
                    sha256_file(os.path.join(REPO, "run_moe_pretrain_from_scratch.py")),
                "swg_config.py": sha256_file(os.path.join(SCRIPT_DIR, "swg_config.py")),
                "swg_config_stageb.py":
                    sha256_file(os.path.join(SCRIPT_DIR, "swg_config_stageb.py")),
            },
        },
        "retries": 0,
        "notes": "",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[run] {code}: {' '.join(cmd)}")
    print(f"[run] log -> {p['log']}")

    with open(p["log"], "w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    rc = proc.returncode

    # 日志异常扫描
    anomaly = None
    with open(p["log"], "r", errors="replace") as logf:
        text = logf.read()
    if "OutOfMemoryError" in text or "CUDA out of memory" in text or "RuntimeError" in text:
        anomaly = "OOM/RuntimeError"
    elif "NaN" in text or "nan" in text:
        anomaly = "NaN"

    results = {"pooled_auc": None, "mean_per_scenario_auc": None,
               "per_scenario_auc": None, "anomaly": anomaly}
    status_out = "failed"
    if rc == 0 and os.path.exists(p["summary"]):
        try:
            with open(p["summary"]) as f:
                s = json.load(f)
            results["pooled_auc"] = s.get("test_auc_all")
            results["mean_per_scenario_auc"] = s.get("mean_per_scenario_auc")
            results["per_scenario_auc"] = s.get("per_scenario_auc")
            status_out = "succeeded"
        except Exception as e:
            results["anomaly"] = f"summary-parse-error: {e}"
            status_out = "failed"
    elif rc != 0:
        results["anomaly"] = results["anomaly"] or f"exit_code={rc}"
        status_out = "failed"
    else:
        results["anomaly"] = results["anomaly"] or "no summary produced"
        status_out = "failed"

    manifest["status"] = status_out
    manifest["timing"]["end"] = datetime.datetime.now().isoformat()
    manifest["timing"]["exit_code"] = rc
    manifest["results"] = results
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] {code}: status={status_out} "
          f"pooled_auc={results['pooled_auc']} anomaly={results['anomaly']}")
    return 0 if status_out == "succeeded" else 1


def run_all(stage):
    if stage == "gate":
        codes = ["swg-dens-sp-42", "swg-dens-sp-123", "swg-dens-sp-456"]
    elif stage == "moe":
        codes = [
            "swg-frout-sp-42", "swg-frout-sp-123", "swg-frout-sp-456",
            "swg-nrout-sp-42", "swg-nrout-sp-123", "swg-nrout-sp-456",
            "swg-pshr-sp-42", "swg-pshr-sp-123", "swg-pshr-sp-456",
        ]
    elif stage == "stageb":
        codes = [
            "stgb-lrfd-fr-42", "stgb-lrfd-fr-123", "stgb-lrfd-fr-456",
            "stgb-lrfd-soft-42", "stgb-lrfd-soft-123", "stgb-lrfd-soft-456",
            "stgb-sfd-42", "stgb-sfd-123", "stgb-sfd-456",
        ]
    else:
        print(f"[ERROR] unknown stage {stage!r}")
        return 2
    for code in codes:
        rc = run_one(code)
        if rc != 0:
            print(f"[queue] stop after {code} (rc={rc}) per §十")
            return rc
    return 0


def main():
    if len(sys.argv) < 2:
        print("usage: run_sample_weighting_gate.py (run <code> | run-all --stage gate|moe|stageb)")
        return 2
    if sys.argv[1] == "run":
        if len(sys.argv) < 3:
            print("usage: run_sample_weighting_gate.py run <code>")
            return 2
        return run_one(sys.argv[2])
    if sys.argv[1] == "run-all":
        stage = "gate"
        rest = sys.argv[2:]
        # 支持两种写法：--stage stageb 与 --stage=stageb
        if "--stage" in rest:
            i = rest.index("--stage")
            if i + 1 < len(rest):
                stage = rest[i + 1]
        else:
            for a in rest:
                if a.startswith("--stage="):
                    stage = a.split("=", 1)[1]
                    break
        return run_all(stage)
    print(f"[ERROR] unknown subcommand {sys.argv[1]!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
