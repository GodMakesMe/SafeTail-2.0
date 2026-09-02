#!/usr/bin/env python
"""
[SAFETAIL][PACKAGE] Bundle the SafeTail 1.0 (original) baseline run into a
self-contained, shareable folder + zip.

Everything needed to read, check or reproduce the baseline result, and nothing
else: the per-seed run logs, the trained model snapshots, the exact
hyperparameters, the environment flags, the comparison tables and the figures.

Usage:
    python tools/package_baseline.py
    python tools/package_baseline.py --out dist --name safetail_v1_baseline
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

RUN_FILES = [
    "latency_log.csv", "request_wise_access_log.csv", "episode_rewards.csv",
    "step_rewards.csv", "access_rate_log.csv", "policy_report.json",
]


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def copy_run(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    info = {"source": str(src.relative_to(REPO)), "files": []}
    for f in RUN_FILES:
        p = src / f
        if p.is_file():
            shutil.copy2(p, dst / f)
            info["files"].append(f)
    snap = src / "snapshots"
    if snap.is_dir():
        shutil.copytree(snap, dst / "snapshots", dirs_exist_ok=True)
        info["snapshots"] = sorted(p.name for p in snap.iterdir())
    lat = dst / "latency_log.csv"
    if lat.is_file():
        info["requests"] = sum(1 for _ in lat.open(encoding="utf-8", errors="replace")) - 1
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dist"))
    ap.add_argument("--name", default="safetail_v1_baseline")
    args = ap.parse_args()

    stage = Path(args.out) / args.name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    manifest: dict = {
        "package": args.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "runs": {},
    }

    # ---- runs -------------------------------------------------------------
    runs_dir = stage / "runs"
    for seed in (0, 1, 2):
        src = RESULTS / f"safetail_v1_legacy_s{seed}"
        if src.is_dir():
            manifest["runs"][f"seed{seed}"] = copy_run(src, runs_dir / f"seed{seed}")
    slow = RESULTS / "v1_legacy_s0"
    if slow.is_dir():
        manifest["runs"]["slowpath_seed0"] = copy_run(slow, runs_dir / "slowpath_seed0")

    # ---- config -----------------------------------------------------------
    cfgdir = stage / "config"
    cfgdir.mkdir()
    for f in (REPO / "baselines" / "safetail_v1" / "config_v1.py",
              REPO / "baselines" / "safetail_v1" / "reward_v1.py",
              REPO / "baselines" / "safetail_v1" / "state_v1.py",
              REPO / "baselines" / "safetail_v1" / "model_v1.py",
              REPO / "baselines" / "safetail_v1" / "train_v1.py",
              REPO / "baselines" / "safetail_v1" / "policy_v1.py"):
        if f.is_file():
            shutil.copy2(f, cfgdir / f.name)

    try:
        sys.path[:0] = [str(REPO / "src"), str(REPO / "baselines")]
        from safetail_v1 import config_v1 as cfg
        manifest["hyperparameters"] = {
            k: getattr(cfg, k) for k in
            ("BETA", "ALPHA", "LR", "GAMMA", "EPS_START", "EPS_MIN", "GAMMA_DECAY",
             "GAMMA_DECAY_NATIVE", "EPS_DECAY_FRACTION", "REPLAY_EVERY", "TOTAL_STEPS",
             "BATCH", "KERAS_BATCH", "EPOCHS", "VAL_SPLIT", "REPLAY_MAXLEN",
             "NS", "NA", "MAX_LOAD", "TAU_BY_TYPE", "LOG_COMPRESS",
             "FREEZE_AFTER_EPISODES")
            if hasattr(cfg, k)
        }
    except Exception as e:  # noqa: BLE001
        manifest["hyperparameters"] = f"could not import config_v1: {e}"

    manifest["environment"] = {
        "SAFETAIL_LEGACY_ENV": 1,
        "note": ("legacy mode restores the four pre-fix behaviours that change the "
                 "LATENCY MODEL (D-02 regressors->server1, D-12/D-34 transmission "
                 "coin flip, D-18 double draw, D-23 wall-clock wait) so the baseline "
                 "is measured under the same physics as the published heterogeneous "
                 "results. Reward-side fixes are NOT reverted; they cannot affect a "
                 "baseline that uses its own tau reward."),
        "requests": 15225, "chunks": 3045, "episodes": 1015, "seeds": [0, 1, 2],
    }

    # ---- tables + figures -------------------------------------------------
    tdir = stage / "results"
    tdir.mkdir()
    for f in ("table_per_seed.csv", "table_main.csv"):
        p = FIGURES / f
        if p.is_file():
            shutil.copy2(p, tdir / f)
    fdir = stage / "figures"
    fdir.mkdir()
    for p in sorted(FIGURES.glob("F*")):
        shutil.copy2(p, fdir / p.name)

    for doc in (RESULTS / "RUN_PROVENANCE.md",
                REPO / "baselines" / "safetail_v1" / "README.md"):
        if doc.is_file():
            shutil.copy2(doc, stage / (
                "RUN_PROVENANCE.md" if doc.name == "RUN_PROVENANCE.md"
                else "FAITHFULNESS_REGISTER.md"))

    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str),
                                         encoding="utf-8")
    (stage / "README.md").write_text(_readme(manifest), encoding="utf-8")

    # ---- zip --------------------------------------------------------------
    archive = shutil.make_archive(str(stage), "zip", root_dir=stage.parent,
                                  base_dir=stage.name)
    size = Path(archive).stat().st_size / 1024 / 1024
    print(f"[SAFETAIL][PACKAGE] folder -> {stage}")
    print(f"[SAFETAIL][PACKAGE] zip    -> {archive}  ({size:.1f} MB)")
    for name, info in manifest["runs"].items():
        print(f"    {name:<16} {info.get('requests', '?'):>6} requests"
              f"{'  + snapshot' if info.get('snapshots') else ''}")
    return 0


def _readme(m: dict) -> str:
    runs = "\n".join(
        f"| `runs/{k}/` | {v.get('requests','?')} | `{v['source']}` |"
        for k, v in m["runs"].items())
    hp = m.get("hyperparameters", {})
    hp_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in hp.items()) if isinstance(hp, dict) else ""
    return f"""# SafeTail 1.0 (original) — baseline run package

The SafeTail **1.0** algorithm run over the **same data** as the heterogeneous
SafeTail 2.0 experiment, for direct comparison against the existing
heterogeneous results.

Created {m['created_utc']} · git `{m['git_commit'][:10]}`{' (dirty)' if m['git_dirty'] else ''}

## What this is

`baselines/safetail_v1/` is a **port**, not a re-run of the old repo: the 1.0
*algorithm* (flat homogeneous state, sigmoid×2+BN network with a softmax head and
categorical-crossentropy loss, the τ-referenced 5-case reward, ε-greedy replay)
driving the SafeTail **2.0** environment. Both sides therefore see the same
servers, the same traces and the same admission rules — the only thing that
differs is the scheduling policy.

## Contents

| path | what |
|---|---|
| `runs/seed{{0,1,2}}/` | the three seeded runs (latency, access-rate, rewards, model snapshot) |
| `runs/slowpath_seed0/` | an independent seed-0 run using Keras `fit()` directly — the fidelity reference for the optimised training path |
| `runs/*/snapshots/` | trained model `.keras` + a `.json` with ε, replay counts, τ and the schedule that produced it |
| `config/` | the exact source of the ported policy (hyperparameters, reward, state, model, trainer) |
| `results/table_per_seed.csv` | percentiles for every policy **per seed** |
| `results/table_main.csv` | aggregate: median across seeds with min/max |
| `figures/` | F1 tail bars · F2 CCDF · F6 per-request-type · F7 latency decomposition (each with its companion `.csv`) |
| `RUN_PROVENANCE.md` | how the runs were produced and the stated limitations |
| `FAITHFULNESS_REGISTER.md` | every deviation from SafeTail 1.0, with the reason |
| `MANIFEST.json` | machine-readable record of all of the above |

### Runs

| folder | requests | source |
|---|---|---|
{runs}

## Hyperparameters

| key | value |
|---|---|
{hp_rows}

## Reading the numbers

* Metric is `total_latency` (ms) = computation + propagation + transmission —
  the **same column with the same meaning** as in the heterogeneous results.
  `end_to_end_latency` (+ queue wait) is also logged.
* All policies are truncated to a **common request count** before percentiles,
  so none can win on sample size.
* Percentiles are computed **per seed**, then reported as median with min/max —
  a single seed is not a result.

## Caveats

Read `RUN_PROVENANCE.md` before quoting anything. The short version: arrival
timing differs slightly from the original socket-driven run (computation mean
8.638 → 8.863 ms, +2.6%), the heterogeneous side is **unfixed** code, and the
comparison is not replication-budget controlled.
"""


if __name__ == "__main__":
    sys.exit(main())
