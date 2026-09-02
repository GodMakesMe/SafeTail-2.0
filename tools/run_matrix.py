#!/usr/bin/env python
"""
[SAFETAIL][RUN][PERF] Parallel run orchestrator -- the real use for 24 cores.

Why this and not "train on more cores": the training loop is ONLINE RL. Each
replay depends on the transition the previous step produced, so the loop itself
is inherently sequential and cannot be data-parallelised. What IS parallel is
the run MATRIX -- every (policy, seed) pair is an independent process. Running
9 of them at once on 24 cores is a ~9x wall-clock win on the matrix, with no
change to the algorithm and no effect on any result.

Each worker is given a thread budget so N workers do not oversubscribe the
machine and thrash. TF on a 2k-parameter net gets nothing from many threads
anyway -- 2 intra-op threads per worker is plenty.

Usage:
    python tools/run_matrix.py --policies safetail_v1 oracle native --seeds 0 1 2
    python tools/run_matrix.py --policies safetail_v1 --seeds 0 1 2 --jobs 3
    python tools/run_matrix.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)


def build_jobs(policies, seeds, chunks, episodes, legacy, tag):
    jobs = []
    for pol in policies:
        for seed in seeds:
            name = f"{pol}_{'legacy' if legacy else 'fixed'}_s{seed}{tag}"
            out = REPO / "results" / name
            if pol == "native":
                cmd = [str(PY), "src/main.py", "--run",
                       "--chunks", str(chunks), "--episodes", str(episodes),
                       "--out", str(out), "--label", name]
            else:
                cmd = [str(PY), "baselines/run_baseline.py", "--policy", pol, "--run",
                       "--chunks", str(chunks), "--episodes", str(episodes),
                       "--out", str(out), "--label", name]
            jobs.append({"name": name, "cmd": cmd, "out": out, "seed": seed, "policy": pol})
    return jobs


def run_job(job, threads, legacy, logdir: Path):
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "SAFETAIL_SEED": str(job["seed"]),
        # thread budget so parallel workers do not oversubscribe
        "TF_NUM_INTRAOP_THREADS": str(threads),
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": str(threads),
        "TF_CPP_MIN_LOG_LEVEL": "3",
    })
    if legacy:
        env["SAFETAIL_LEGACY_ENV"] = "1"

    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{job['name']}.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call(job["cmd"], cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    lat = job["out"] / "latency_log.csv"
    n = sum(1 for _ in lat.open(encoding="utf-8", errors="replace")) - 1 if lat.is_file() else 0
    status = "ok" if rc == 0 else f"FAILED rc={rc}"
    print(f"[SAFETAIL][RUN] {job['name']:<34} {status:<12} {n:>6} req  {dt/60:5.1f} min  -> {log}",
          flush=True)
    return {"name": job["name"], "rc": rc, "requests": n, "minutes": dt / 60, "log": str(log)}


def main() -> int:
    ncpu = os.cpu_count() or 4
    ap = argparse.ArgumentParser(description="Parallel SafeTail run matrix")
    ap.add_argument("--policies", nargs="+", default=["safetail_v1", "oracle"],
                    help="safetail_v1 | oracle | native (the 2.0 DQN)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--chunks", type=int, default=3045, help="3045 => 15,225 requests")
    ap.add_argument("--episodes", type=int, default=1015)
    ap.add_argument("--jobs", type=int, default=0, help="0 => auto (cores // threads-per-job)")
    ap.add_argument("--threads", type=int, default=2, help="TF intra-op threads per worker")
    ap.add_argument("--legacy", action="store_true", default=True,
                    help="run under SAFETAIL_LEGACY_ENV (default: on, for comparability)")
    ap.add_argument("--no-legacy", dest="legacy", action="store_false")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.policies, args.seeds, args.chunks, args.episodes, args.legacy, args.tag)
    parallel = args.jobs or max(1, min(len(jobs), ncpu // max(1, args.threads)))

    print(f"[SAFETAIL][RUN] {ncpu} logical cores | {len(jobs)} jobs | "
          f"{parallel} in parallel | {args.threads} TF threads each | "
          f"legacy={args.legacy} | {args.chunks * 5} requests each")
    for j in jobs:
        print(f"    {j['name']:<34} -> {j['out'].relative_to(REPO)}")
    if args.dry_run:
        return 0

    logdir = REPO / "results" / "_matrix_logs"
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        results = list(ex.map(lambda j: run_job(j, args.threads, args.legacy, logdir), jobs))

    ok = sum(1 for r in results if r["rc"] == 0)
    print(f"\n[SAFETAIL][RUN] {ok}/{len(results)} succeeded in {(time.time()-t0)/60:.1f} min wall")
    serial = sum(r["minutes"] for r in results)
    print(f"[SAFETAIL][RUN] serial equivalent {serial:.1f} min -> "
          f"speedup {serial / max(1e-9, (time.time()-t0)/60):.1f}x")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
