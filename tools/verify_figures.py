#!/usr/bin/env python
"""
[SAFETAIL][PLOT][AUDIT] Gate G6 -- figure integrity.

plan.md section 11. Checks:

  1. Every figure in figures/ has a companion .csv of the numbers plotted.
     ("If a number is in a figure, it must be in a CSV.")
  2. tools/make_figures.py contains NO per-mode additive offset. This is the
     D-03 guard: plotting/plot_baselines.ipynb added a hardcoded +0.005 s
     (5 ms) to the MinProp family ONLY, in two cells, before percentiles were
     taken -- an 11% inflation of one baseline family that went unnoticed for
     months because it lived in a notebook.
  3. No notebook under plotting/ is referenced as a publication path, and any
     surviving notebook that still carries the D-03 offset is reported.
  4. table_main.csv exists and its percentiles are internally consistent
     (p50 <= p90 <= p95 <= p99 for every policy).

Usage:  python tools/verify_figures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "figures"
MAKER = REPO / "tools" / "make_figures.py"

# any literal added to a latency series, keyed on a mode/family name
OFFSET_PATTERNS = [
    re.compile(r"v\s*=\s*0?\.0*[1-9]"),                       # load_latencies(mode, v=0.005)
    re.compile(r"\+\s*0?\.00[0-9]+"),                          # + 0.005
    re.compile(r"(minprop|minload|rand|safetail)\w*.{0,40}[+]\s*\d", re.I),
]


class Fail(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"[SAFETAIL][AUDIT][G6][FAIL] {msg}")


def _ok(m: str) -> None:
    print(f"[SAFETAIL][AUDIT][G6][ok] {m}")


def check_companions() -> None:
    if not FIGS.is_dir():
        raise Fail(f"figures/ not found at {FIGS} -- run tools/make_figures.py first")
    imgs = sorted(p for p in FIGS.glob("*.png"))
    if not imgs:
        raise Fail("figures/ contains no .png")
    missing = [p.name for p in imgs if not (FIGS / f"{p.stem}.csv").is_file()]
    if missing:
        raise Fail("figures without a companion .csv: " + ", ".join(missing))
    _ok(f"all {len(imgs)} figures have a companion .csv")


def check_no_offsets() -> None:
    if not MAKER.is_file():
        raise Fail(f"{MAKER} not found")
    hits = []
    for i, line in enumerate(MAKER.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("#", 1)[0]
        if not code.strip():
            continue
        for pat in OFFSET_PATTERNS:
            if pat.search(code):
                hits.append(f"{MAKER.name}:{i}: {line.strip()}")
                break
    if hits:
        raise Fail("possible per-mode additive offset in the figure script (D-03):\n  "
                   + "\n  ".join(hits))
    _ok("tools/make_figures.py carries no per-mode additive offset (D-03 guard)")


def check_notebooks() -> None:
    nbs = list((REPO / "plotting").glob("*.ipynb")) if (REPO / "plotting").is_dir() else []
    flagged = []
    for nb in nbs:
        try:
            src = json.dumps(json.loads(nb.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            continue
        if re.search(r"v\s*=\s*0\.005|\+\s*0\.005", src):
            flagged.append(nb.name)
    if flagged:
        print(f"[SAFETAIL][AUDIT][G6][warn] notebook(s) still contain the D-03 "
              f"+0.005 MinProp offset: {flagged}. They are NOT the publication path "
              f"(tools/make_figures.py is); do not use them for paper numbers.")
    else:
        _ok("no notebook carries the D-03 offset")


def check_table() -> None:
    t = FIGS / "table_main.csv"
    if not t.is_file():
        raise Fail("figures/table_main.csv missing")
    df = pd.read_csv(t)
    # the aggregate table uses p<NN>_med (median across seeds); a single-run
    # table uses bare p<NN>. Accept either.
    suffix = "_med" if "p50_med" in df.columns else ""
    cols = [f"p{p}{suffix}" for p in (50, 90, 95, 99)]
    need = {"policy", *cols}
    if not need <= set(df.columns):
        raise Fail(f"table_main.csv missing columns {need - set(df.columns)}")
    bad = []
    for _, r in df.iterrows():
        vals = [r[c] for c in cols]
        if not all(a <= b for a, b in zip(vals, vals[1:])):
            bad.append(f"{r.policy}: " + "/".join(f"{v:.2f}" for v in vals))
    if bad:
        raise Fail("non-monotonic percentiles: " + "; ".join(bad))
    if df["n"].nunique() != 1:
        raise Fail(f"policies compared at different n: {df[['policy','n']].to_dict('records')}")

    # per-seed table, when present, must show >1 seed for at least one policy
    ps = FIGS / "table_per_seed.csv"
    if ps.is_file():
        d2 = pd.read_csv(ps)
        multi = d2.groupby("policy")["seed"].nunique()
        if (multi > 1).sum() == 0:
            print("[SAFETAIL][AUDIT][G6][warn] every policy has a single seed -- "
                  "plan.md B7 wants >=3 before publishing")
        else:
            _ok(f"per-seed table: {int((multi > 1).sum())} policies with >1 seed "
                f"(max {int(multi.max())})")
    _ok(f"table_main.csv: {len(df)} policies, percentiles monotonic, common n={int(df.n.iloc[0])}")


def main() -> int:
    print("[SAFETAIL][AUDIT][G6] verifying figure integrity ...")
    check_companions()
    check_no_offsets()
    check_notebooks()
    check_table()
    print("\n[SAFETAIL][AUDIT][G6] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
