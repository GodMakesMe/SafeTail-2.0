#!/usr/bin/env python
"""
[SAFETAIL][AUDIT][S-14] Gate G7 -- request-type mapping, asserted against the DATASET.

plan.md section 11 + S-14 + decision 13.1(6). This gate exists for exactly one
reason: BTP section 3.3 and section 4.6 prose, and HED item 9, all misname the
request types, and they read far more authoritatively than a CSV column does. An
agent "fixing" correct code to match that prose would silently break the type
system. G7 makes the dataset the arbiter.

Checks (exit non-zero on the first failure):

  1. In every dataset/server{1..5}.csv, the single-letter Combination rows map
         s -> Speech,  d -> Detect,  p -> Predict
     via the 'Scripts Executed' column.
  2. Magnitude sanity: on every server, the 's' (Speech) row has the LARGEST
     'Total Processing Time (sec)' of the three single-letter rows (HED section IV-D:
     speech-to-text 60-300 ms vs vision 5-20 ms).
  3. src/constants.ORIGINAL_DEADLINES pairs s -> (100, 400) and d/p -> (30, 200).
  4. If src/regressors.py exists (post-B1), its TASK_FOR_LETTER maps
     s->speech, d->detect, p->predict. (Skipped with a note until B1 lands.)
  5. The legacy per-server regressor wrappers, if still present, resolve
     scripts.index('speech'|'detect'|'predict') for s|d|p respectively.

Usage:  python tools/verify_types.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SRC = REPO / "src"

LETTER_TO_SCRIPT = {"s": "speech", "d": "detect", "p": "predict"}
EXPECTED_DEADLINES = {"s": (100, 400), "d": (30, 200), "p": (30, 200)}


class Fail(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"[SAFETAIL][AUDIT][G7][FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"[SAFETAIL][AUDIT][G7][ok] {msg}")


def check_dataset_mapping() -> None:
    for s in range(1, 6):
        csv = DATA / f"server{s}.csv"
        if not csv.is_file():
            raise Fail(f"missing {csv}")
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        singles = df[df["Combination"].astype(str).str.strip().str.len() == 1]
        seen = {}
        for _, row in singles.iterrows():
            letter = str(row["Combination"]).strip().lower()
            script = str(row["Scripts Executed"]).strip().lower()
            seen.setdefault(letter, set()).add(script)
        for letter, want in LETTER_TO_SCRIPT.items():
            got = seen.get(letter)
            if not got:
                raise Fail(f"server{s}.csv has no single-letter row for {letter!r}")
            if got != {want}:
                raise Fail(
                    f"server{s}.csv: letter {letter!r} maps to {sorted(got)}, expected [{want!r}]. "
                    f"S-14: the dataset is authoritative -- do NOT 'fix' this to match BTP prose."
                )
        # magnitude: Speech is the slowest single-letter task
        tpt = {}
        for _, row in singles.iterrows():
            letter = str(row["Combination"]).strip().lower()
            tpt.setdefault(letter, []).append(float(row["Total Processing Time (sec)"]))
        s_t = min(tpt["s"])  # smallest speech still expected to beat largest of d/p
        dp_max = max(max(tpt["d"]), max(tpt["p"]))
        if not s_t > dp_max:
            raise Fail(
                f"server{s}.csv: Speech time {s_t:.4f}s is not > max(Detect,Predict) {dp_max:.4f}s; "
                f"the s<->Speech identification is magnitude-inconsistent (S-14)."
            )
    _ok("dataset: s->Speech, d->Detect, p->Predict on all 5 servers; Speech is slowest (S-14)")


def check_deadlines() -> None:
    sys.path.insert(0, str(SRC))
    import constants  # noqa: E402
    od = [list(pair) for pair in constants.ORIGINAL_DEADLINES]
    if od != [[100, 400], [30, 200]]:
        raise Fail(f"constants.ORIGINAL_DEADLINES = {od}, expected [[100,400],[30,200]]")
    # constants stores [s_pair, dp_pair]; verify the request_factory pairing intent
    _ok("constants.ORIGINAL_DEADLINES: s->(100,400), d/p->(30,200)")


def check_regressors_module() -> None:
    reg = SRC / "regressors.py"
    if not reg.is_file():
        print("[SAFETAIL][AUDIT][G7][skip] src/regressors.py not present yet (B1 not landed)")
        return
    sys.path.insert(0, str(SRC))
    import regressors  # noqa: E402
    mapping = getattr(regressors, "TASK_FOR_LETTER", None)
    if mapping is None:
        raise Fail("src/regressors.py has no TASK_FOR_LETTER mapping to assert against")
    norm = {k: str(v).lower() for k, v in mapping.items()}
    if norm != LETTER_TO_SCRIPT:
        raise Fail(f"regressors.TASK_FOR_LETTER = {norm}, expected {LETTER_TO_SCRIPT}")
    _ok("src/regressors.py TASK_FOR_LETTER: s->speech, d->detect, p->predict")


def check_legacy_wrappers() -> None:
    hits = 0
    for s in range(1, 6):
        folder = SRC / f"server{s}_regressor"
        if not folder.is_dir():
            continue
        for letter, script in LETTER_TO_SCRIPT.items():
            wrapper = folder / f"{script}_predictor.py"
            if not wrapper.is_file():
                raise Fail(f"{wrapper} missing but {folder} exists")
            text = wrapper.read_text(encoding="utf-8", errors="replace").lower()
            if f'index("{script}")' not in text and f"index('{script}')" not in text:
                raise Fail(
                    f"{wrapper}: expected scripts.index({script!r}) for letter {letter!r}"
                )
            hits += 1
    if hits:
        _ok(f"legacy per-server wrappers: {hits} resolve scripts.index(<matching task>)")
    else:
        print("[SAFETAIL][AUDIT][G7][skip] legacy server*_regressor wrappers already removed")


def main() -> int:
    print("[SAFETAIL][AUDIT][G7] verifying request-type mapping against the dataset ...")
    check_dataset_mapping()
    check_deadlines()
    check_regressors_module()
    check_legacy_wrappers()
    print("\n[SAFETAIL][AUDIT][G7] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
