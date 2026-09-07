"""
[SAFETAIL][AUDIT][G8] Gate G8 -- is the trace dataset fit to train on?

Run this the moment a NEW dataset lands, BEFORE spending hours training on it.
It answers one question: does this data actually carry the computation-time
distribution the simulator needs (D-40), or is it another set of averages?

Background (plan.md 4.4b, D-40). The shipped dataset has 363 rows and 363 unique
contention strings with `Iteration == [1]` -- exactly ONE measurement per
scenario, and each of those is itself an average over ~500 files. The spread was
destroyed at collection time. Consequences:

  * computation latency is deterministic -- 8 identical calls return the identical
    value, stddev exactly 0;
  * measured p99/p50 on the shipped run is 2.89x, where real tail-latency
    problems are 5-50x;
  * SafeTail 1.0 had a noise term (`np.random.normal(0, st_dev[...])`) and 2.0
    deleted it, so this is a regression, not an original gap.

There is nothing to optimise for tail latency until this is fixed, and it can
only be fixed with data.

WHAT GOOD DATA LOOKS LIKE
  * several rows per contention string (>= MIN_REPEATS), so a distribution exists
  * a non-trivial spread within each contention string (CV above a floor), so the
    repeats are real measurements and not the same number copied
  * coverage of the contention strings the simulator actually asks for -- every
    non-empty multiset over {s, d, p} up to the server concurrency limit (4)
  * every column the fitted regressors need, per server schema
  * a per-server `Iteration` (or equivalent) column so repeats are identifiable

Usage:
    python tools/verify_dataset.py
    python tools/verify_dataset.py --dataset path/to/new_dataset --min-repeats 10
    python tools/verify_dataset.py --strict     # fail (exit 1) instead of warn

Exit code is 0 while the dataset is merely INADEQUATE (that is today's known
state, and failing by default would just make the gate noise). With --strict it
is 1, which is what you want in CI once the new data is in.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LETTERS = ("s", "d", "p")
MAX_CONCURRENT = 4          # servers.MAX_CONCURRENT_REQUESTS
TIME_COL = "Total Processing Time (sec)"

# Columns the fitted regressor bundles consume, by schema (see src/regressors.py).
SCHEMA_COLS = {
    "gpu": ["Combination", "Scripts Executed", "Peak RAM Usage (MB)",
            "Peak GPU Usage (%)", "Peak GPU Memory (MB)", TIME_COL],
    "cpu": ["Combination", "Scripts Executed", "Peak RAM Usage (MB)",
            "Peak CPU Usage (%)", "Average CPU Clock (MHz)", TIME_COL],
}


def _ok(m): print(f"[SAFETAIL][AUDIT][G8][ok] {m}")
def _warn(m): print(f"[SAFETAIL][AUDIT][G8][warn] {m}")
def _bad(m, strict=False):
    # [SAFETAIL][AUDIT][G8] the severity marker MUST match the exit code.
    # Printing [FAIL] while exiting 0 is how a gate gets chained with && and
    # silently passes unusable data -- the exact failure mode G8 exists to
    # prevent. In advisory mode (the default, because the CURRENT dataset is
    # known-inadequate and a permanently-red gate just gets ignored) the marker
    # is [INADEQUATE] and the exit is 0. With --strict it is [FAIL] and 1.
    print(f"[SAFETAIL][AUDIT][G8][{'FAIL' if strict else 'INADEQUATE'}] {m}")


def expected_combinations() -> set[str]:
    """Every contention string the simulator can produce: non-empty multisets
    over {s,d,p} of size 1..MAX_CONCURRENT, normalised the way the code keys
    them (lowercase, as emitted by the server's active-request ordering)."""
    out = set()
    for n in range(1, MAX_CONCURRENT + 1):
        for combo in itertools.combinations_with_replacement(LETTERS, n):
            out.add("".join(combo))
    return out


def audit_one(path: Path, min_repeats: int, min_cv: float) -> dict:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    name = path.name
    res = {"file": name, "rows": len(df), "problems": [], "warnings": []}

    if "Combination" not in df.columns:
        res["problems"].append("no 'Combination' column")
        return res

    key = df["Combination"].astype(str).str.strip().str.lower()
    counts = key.value_counts()
    res["unique_combos"] = int(counts.size)
    res["repeats_min"] = int(counts.min())
    res["repeats_med"] = float(counts.median())
    res["repeats_max"] = int(counts.max())

    # --- 1. are there repeats at all? -------------------------------------
    if res["repeats_max"] <= 1:
        res["problems"].append(
            f"ONE measurement per contention string ({res['unique_combos']} combos, "
            f"{len(df)} rows). No distribution exists -> computation latency is "
            f"deterministic -> there is no tail to optimise (D-40)."
        )
    elif res["repeats_min"] < min_repeats:
        thin = counts[counts < min_repeats]
        res["warnings"].append(
            f"{len(thin)} contention string(s) have fewer than {min_repeats} repeats "
            f"(min {res['repeats_min']}), e.g. {list(thin.index[:5])}"
        )

    # --- 2. is the spread real? -------------------------------------------
    if TIME_COL in df.columns and res["repeats_max"] > 1:
        g = df.groupby(key)[TIME_COL]
        cv = (g.std() / g.mean()).dropna()
        if len(cv):
            res["cv_median"] = float(cv.median())
            flat = cv[cv < min_cv]
            if len(flat) == len(cv):
                res["problems"].append(
                    f"repeats exist but carry NO spread (median CV {cv.median():.5f} "
                    f"< {min_cv}) -- the same value repeated is not a distribution"
                )
            elif len(flat):
                res["warnings"].append(
                    f"{len(flat)}/{len(cv)} contention strings have CV < {min_cv} "
                    f"(median CV {cv.median():.4f})"
                )
    elif TIME_COL not in df.columns:
        res["problems"].append(f"no '{TIME_COL}' column -- nothing to fit or sample")

    # --- 3. coverage -------------------------------------------------------
    have = set(counts.index)
    want = expected_combinations()
    missing = sorted(want - have)
    res["coverage"] = f"{len(want & have)}/{len(want)}"
    if missing:
        res["warnings"].append(
            f"{len(missing)} contention string(s) the simulator can ask for are "
            f"absent, e.g. {missing[:8]} -- these raise at predict time unless "
            f"SAFETAIL_ALLOW_DEGRADED_PREDICTORS=1"
        )

    # --- 4. schema ---------------------------------------------------------
    cols = set(df.columns)
    schema = "gpu" if "Peak GPU Usage (%)" in cols else "cpu"
    res["schema"] = schema
    missing_cols = [c for c in SCHEMA_COLS[schema] if c not in cols]
    if missing_cols:
        res["problems"].append(f"{schema} schema is missing columns: {missing_cols}")

    # --- 5. repeat identifier ---------------------------------------------
    if "Iteration" in df.columns:
        iters = sorted(pd.unique(df["Iteration"].dropna()))
        res["iterations"] = len(iters)
        if len(iters) <= 1 and res["repeats_max"] > 1:
            res["warnings"].append(
                "repeats exist but 'Iteration' has a single value -- repeats are "
                "not individually identifiable"
            )
    else:
        res["warnings"].append("no 'Iteration' column -- repeats are not labelled")

    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate G8 -- trace dataset fitness (D-40)")
    ap.add_argument("--dataset", default=str(REPO / "dataset"))
    ap.add_argument("--min-repeats", type=int, default=5,
                    help="minimum measurements per contention string (default 5)")
    ap.add_argument("--min-cv", type=float, default=0.01,
                    help="minimum coefficient of variation within a contention "
                         "string for the spread to count as real (default 0.01)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the dataset is inadequate (use once the new "
                         "data is in; without it an inadequate dataset only warns, "
                         "because that is the known current state)")
    args = ap.parse_args()

    root = Path(args.dataset)
    files = sorted(root.glob("server*.csv"))
    if not files:
        print(f"[SAFETAIL][AUDIT][G8] no server*.csv under {root}")
        return 1

    print(f"[SAFETAIL][AUDIT][G8] auditing {len(files)} trace file(s) in {root}\n")
    any_problem = False
    usable = []

    for f in files:
        r = audit_one(f, args.min_repeats, args.min_cv)
        head = (f"{r['file']}: {r['rows']} rows, {r.get('unique_combos', '?')} combos, "
                f"repeats min/med/max = {r.get('repeats_min','?')}/"
                f"{r.get('repeats_med','?')}/{r.get('repeats_max','?')}, "
                f"coverage {r.get('coverage','?')}, schema {r.get('schema','?')}")
        if "cv_median" in r:
            head += f", median CV {r['cv_median']:.4f}"
        print(f"  {head}")
        for p in r["problems"]:
            any_problem = True
            _bad(f"  {r['file']}: {p}", strict=args.strict)
        for w in r["warnings"]:
            _warn(f"  {r['file']}: {w}")
        if not r["problems"]:
            usable.append(r["file"])
        print()

    if usable:
        _ok(f"{len(usable)}/{len(files)} trace file(s) carry a usable distribution: {usable}")

    if any_problem:
        print(f"\n[SAFETAIL][AUDIT][G8] {'FAIL' if args.strict else 'VERDICT (advisory, exit 0)'}: "
              f"the dataset does NOT carry a")
        print("  computation-time distribution. Training on it cannot produce a")
        print("  tail-latency result (D-40). What is needed: repeated measurements")
        print("  per contention string -- several rows for each of s, d, p, ss, sd,")
        print("  sp, ... -- NOT one averaged row per scenario.")
        print("  Once such a dataset exists, src/regressors.py samples among the")
        print("  repeats automatically (constants.TRACE_SAMPLING='sample'), and")
        print("  the regressors should be refitted on all rows (B1b).")
        if not args.strict:
            print("\n  NOTE: advisory mode -- exit code 0 so this gate can sit in a\n  sequence while the current dataset is known-inadequate. Use --strict to\n  make it BLOCK (exit 1), which is what you want once the new data lands.")
        return 1 if args.strict else 0

    print("\n[SAFETAIL][AUDIT][G8] PASS -- every trace carries repeated measurements")
    print("  with real spread. Next: refit the regressors on all rows (B1b), then")
    print("  re-run. Expect p99/p50 to rise well above the 2.89x of the old data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
