#!/usr/bin/env python
"""
[SAFETAIL][PACKAGE] Bundle the 3x-data-budget experiment into its own
self-contained folder + zip.

This is the experiment that answers "were these policies under-trained?".
Both SafeTail 1.0 and the repaired SafeTail 2.0 were re-run at 3x the data
(45,675 requests / 3,045 episodes, epsilon schedule rescaled), 3 seeds each,
under SAFETAIL_LEGACY_ENV=1. It is packaged separately from the 1x baseline
deliverable because it answers a different question and must not be mixed into
the matched-budget comparison.

Usage:
    python tools/package_budget3x.py
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

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
PCTS = (50, 90, 95, 99)

RUN_FILES = ["latency_log.csv", "request_wise_access_log.csv", "episode_rewards.csv",
             "step_rewards.csv", "access_rate_log.csv", "policy_report.json"]

SERIES = [
    ("safetail_v1_1x",  "results/safetail_v1_legacy_s{}",         "SafeTail-1.0",          "1x"),
    ("safetail_v1_3x",  "results/safetail_v1_legacy_s{}_long3x",  "SafeTail-1.0",          "3x"),
    ("safetail_v2_1x",  "results/native_legacy_s{}",              "SafeTail-2.0 (updated)", "1x"),
    ("safetail_v2_3x",  "results/native_legacy_s{}_long3x",       "SafeTail-2.0 (updated)", "3x"),
]


def _git(*a) -> str:
    try:
        return subprocess.check_output(["git", *a], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def copy_run(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    info = {"source": str(src.relative_to(REPO)), "files": []}
    for f in RUN_FILES:
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f); info["files"].append(f)
    if (src / "snapshots").is_dir():
        shutil.copytree(src / "snapshots", dst / "snapshots", dirs_exist_ok=True)
        info["snapshots"] = sorted(p.name for p in (src / "snapshots").iterdir())
    lat = dst / "latency_log.csv"
    if lat.is_file():
        info["requests"] = sum(1 for _ in lat.open(encoding="utf-8", errors="replace")) - 1
    return info


def summarise() -> pd.DataFrame:
    """Per-seed percentiles for the full run AND the converged (last) third."""
    rows = []
    for key, pat, label, budget in SERIES:
        for s in (0, 1, 2):
            f = REPO / pat.format(s) / "latency_log.csv"
            if not f.is_file():
                continue
            d = pd.read_csv(f)
            v = d.total_latency.to_numpy(float)
            ar = (REPO / pat.format(s) / "request_wise_access_log.csv")
            k = np.nan
            if ar.is_file():
                a = pd.read_csv(ar)
                if len(a) and a.access_rate.sum() > 0:
                    k = float(a.access_rate.mean() * 5)
            for phase, vv in (("full", v), ("converged_last_third", v[int(len(v) * 2 / 3):])):
                rows.append({"series": key, "policy": label, "budget": budget, "seed": s,
                             "phase": phase, "n": len(vv), "mean_K": round(k, 3) if k == k else None,
                             **{f"p{p}": round(float(np.percentile(vv, p)), 3) for p in PCTS},
                             "mean": round(float(vv.mean()), 3)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dist"))
    ap.add_argument("--name", default="safetail_budget3x_experiment")
    args = ap.parse_args()

    stage = Path(args.out) / args.name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    manifest: dict = {
        "package": args.name,
        "experiment": "3x data-budget test: were SafeTail 1.0 / 2.0 under-trained?",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": platform.node(), "python": sys.version.split()[0],
        "environment": {
            "SAFETAIL_LEGACY_ENV": 1,
            "1x": {"requests": 15225, "chunks": 3045, "episodes": 1015},
            "3x": {"requests": 45675, "chunks": 9135, "episodes": 3045,
                   "SAFETAIL_V1_TOTAL_STEPS": 45675,
                   "note": "epsilon schedule rescaled so it still decays over 45% "
                           "of the longer run instead of bottoming out at 15%"},
            "seeds": [0, 1, 2],
        },
        "runs": {},
    }

    for key, pat, label, budget in SERIES:
        for s in (0, 1, 2):
            src = REPO / pat.format(s)
            if src.is_dir():
                manifest["runs"][f"{key}_s{s}"] = copy_run(src, stage / "runs" / key / f"seed{s}")

    tbl = summarise()
    (stage / "results").mkdir()
    tbl.to_csv(stage / "results" / "budget3x_per_seed.csv", index=False)
    agg = (tbl.groupby(["policy", "budget", "phase"])[[f"p{p}" for p in PCTS] + ["mean", "mean_K"]]
             .median().round(2).reset_index())
    agg.to_csv(stage / "results" / "budget3x_summary.csv", index=False)

    (stage / "figures").mkdir()
    for p in list(FIGURES.glob("F8_budget_3x.*")) + list(FIGURES.glob("F5_convergence.*")):
        shutil.copy2(p, stage / "figures" / p.name)

    rep = RESULTS / "BASELINE_COMPARISON_REPORT.md"
    if rep.is_file():
        shutil.copy2(rep, stage / "BASELINE_COMPARISON_REPORT.md")

    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (stage / "README.md").write_text(_readme(agg, manifest), encoding="utf-8")

    archive = shutil.make_archive(str(stage), "zip", root_dir=stage.parent, base_dir=stage.name)
    print(f"[SAFETAIL][PACKAGE] folder -> {stage}")
    print(f"[SAFETAIL][PACKAGE] zip    -> {archive}  "
          f"({Path(archive).stat().st_size/1024/1024:.1f} MB)")
    print(f"\n{agg.to_string(index=False)}")
    return 0


def _readme(agg: pd.DataFrame, m: dict) -> str:
    conv = agg[agg.phase == "converged_last_third"]
    tbl = "\n".join(
        f"| {r.policy} — {r.budget} | {r.mean_K:.2f} | {r.p50:.2f} | {r.p90:.2f} | "
        f"**{r.p95:.2f}** | {r.p99:.2f} |" for r in conv.itertuples())
    return f"""# 3× data-budget experiment

**Question.** Were SafeTail 1.0 and the repaired SafeTail 2.0 under-trained at
the 15,225-request budget used for the matched comparison?

**Answer: no. Neither is data-limited.** Both had already converged at 1×.

Created {m['created_utc']} · git `{m['git_commit'][:10]}`

## Design

Both policies re-run at **3× the data** — 45,675 requests / 3,045 episodes vs
15,225 / 1,015 — 3 seeds each, under `SAFETAIL_LEGACY_ENV=1` so the latency
physics match the published heterogeneous run. The ε schedule was rescaled
(`SAFETAIL_V1_TOTAL_STEPS=45675`) so it still decays over 45% of the longer run
rather than bottoming out at 15% of it.

## Why the converged phase, not the full run

A full-run percentile mixes the exploration phase with the converged phase, and
a 3× run **dilutes the exploration share** — so a naive full-run comparison
overstates the benefit of more data. Both are in
`results/budget3x_per_seed.csv` (`phase` column); the honest comparison is
`converged_last_third`.

## Result — converged phase, median of 3 seeds

| run | mean K | p50 | p90 | p95 | p99 |
|---|---|---|---|---|---|
{tbl}

Reference points from the shipped heuristics (original code, same physics):
**MinProp-3** (K=3) p95 49.18 / p99 56.31 · **Oracle** (K=5) p95 49.53 / p99 52.45.

## Conclusions

1. **Tripling the data moves converged p95 by 1.6% (SafeTail 1.0) and 3.0%
   (SafeTail 2.0).** Neither was under-trained.
2. **SafeTail 1.0's ceiling is architectural.** It plateaus at p95 ≈ 84–86 ms at
   any budget, and its p99 gets *worse* with more data (111.13 → 114.36) — the
   instability signature. Its `softmax` + `categorical_crossentropy` head cannot
   represent negative real Q-values, and no amount of data fixes that.
3. **The repaired SafeTail 2.0 reaches oracle-competitive tail latency.** At its
   plateau: p95 50.28 / p99 55.47 at mean K = 3.00, versus MinProp-3 (K=3) at
   49.18 / 56.31 and the Oracle (K=5) at 49.53 / 52.45. At **matched replication
   budget** it ties the best heuristic and essentially matches the oracle while
   using 3 servers instead of 5.
4. This **substantially revises audit finding D-10**. "SafeTail loses to MinProp
   by 54% at p99" is a property of the shipped, unfixed, exploration-contaminated
   run — not of the approach. It is still **not a win** (parity at K=3 against a
   heuristic that manages it at K=2 is 50% more compute for equal tail latency)
   and must not be reported as one.

## Contents

| path | what |
|---|---|
| `runs/safetail_v1_{{1x,3x}}/seed{{0,1,2}}/` | SafeTail 1.0 at both budgets |
| `runs/safetail_v2_{{1x,3x}}/seed{{0,1,2}}/` | repaired SafeTail 2.0 at both budgets |
| `results/budget3x_per_seed.csv` | every seed, both phases, all percentiles + mean K |
| `results/budget3x_summary.csv` | median across seeds |
| `figures/F8_budget_3x.*` | converged-phase 1× vs 3×, with MinProp-3 / Oracle reference lines |
| `figures/F5_convergence.*` | p95 by decile — how each policy converges (or degrades) |
| `BASELINE_COMPARISON_REPORT.md` | the full report; §3 covers this experiment |

## Reproduce

```bash
SAFETAIL_V1_TOTAL_STEPS=45675 python tools/run_matrix.py \\
    --policies safetail_v1 native --seeds 0 1 2 \\
    --chunks 9135 --episodes 3045 --legacy --tag _long3x
python tools/make_figures.py
python tools/package_budget3x.py
```
"""


if __name__ == "__main__":
    sys.exit(main())
