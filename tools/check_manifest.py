"""
[SAFETAIL][AUDIT][G5] Gate G5 -- run provenance.

plan.md section 11:

    G5 | check_manifest.py | every results/*/manifest.json present, complete,
                             zero [DEGRADED] counts, git clean | blocks D

and section 10.3:

    A run whose manifest reports any [DEGRADED] count > 0 is NOT PUBLISHABLE.
    Make tools/check_manifest.py enforce it.

What this checks, per run directory under `results/`:

  1. `manifest.json` exists and parses.
  2. Required keys are present and non-empty (schema, git.sha, seed, constants,
     packages, inputs_sha256, degraded).
  3. The `degraded` ledger is EMPTY -- any [DEGRADED] event (e.g. D-02c
     regressor fallback) makes the run unpublishable.
  4. `git.dirty` is false -- a SHA recorded from a dirty tree does not describe
     the code that ran.
  5. Input hashes agree with the CURRENT dataset/ and models/ on disk, so a run
     cannot silently be attributed to inputs that have since changed.

Runs that predate the manifest (everything produced before G5 landed) are
reported as UNPROVENANCED. They are not failures -- they cannot retroactively
grow a manifest -- but they are listed explicitly so no one quotes them as if
they were gated. Pass `--strict` to make them fail too, and `--require <dir>` to
insist that specific runs carry a manifest.

Usage:
    python tools/check_manifest.py
    python tools/check_manifest.py --strict
    python tools/check_manifest.py --require native_legacy_s0 --require oracle_legacy_s0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
MANIFEST_NAME = "manifest.json"

REQUIRED_KEYS = ("schema", "git", "seed", "constants", "packages",
                 "inputs_sha256", "degraded")

# Directories under results/ that are not runs.
_NOT_RUNS = {"_superseded", "_validate", "_matrix_logs"}


class Fail(Exception):
    pass


def _ok(msg: str) -> None:
    print(f"[SAFETAIL][AUDIT][G5][ok] {msg}")


def _warn(msg: str) -> None:
    print(f"[SAFETAIL][AUDIT][G5][warn] {msg}")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _current_input_hashes() -> dict:
    out = {}
    for pattern in ("dataset/*.csv", "models/*/*.pkl"):
        for p in sorted(REPO.glob(pattern)):
            try:
                out[p.relative_to(REPO).as_posix()] = _sha256(p)
            except Exception:
                pass
    return out


def _run_dirs() -> list[Path]:
    if not RESULTS.is_dir():
        raise Fail(f"{RESULTS} does not exist")
    return sorted(
        d for d in RESULTS.iterdir()
        if d.is_dir() and d.name not in _NOT_RUNS and not d.name.startswith(".")
    )


def check_one(d: Path, current_inputs: dict) -> tuple[str, list[str]]:
    """Return (status, problems). status in {ok, unprovenanced, bad}."""
    mpath = d / MANIFEST_NAME
    if not mpath.exists():
        return "unprovenanced", []

    problems: list[str] = []
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "bad", [f"manifest does not parse: {type(exc).__name__}: {exc}"]

    for key in REQUIRED_KEYS:
        if key not in m:
            problems.append(f"missing key {key!r}")
        elif key not in ("seed", "degraded") and not m[key]:
            problems.append(f"empty key {key!r}")

    git = m.get("git") or {}
    if not git.get("sha"):
        problems.append("git.sha missing -- run is not attributable to any commit")
    if git.get("dirty"):
        files = ", ".join((git.get("dirty_files") or [])[:5])
        problems.append(f"git tree was DIRTY at run time ({files}...) -- the recorded "
                        f"SHA does not describe the code that ran")

    degraded = m.get("degraded") or {}
    if degraded:
        problems.append(f"[DEGRADED] ledger is non-empty: {degraded} -- plan.md 10.3: "
                        f"this run is NOT publishable")

    recorded = m.get("inputs_sha256") or {}
    if recorded and current_inputs:
        changed = [k for k, v in recorded.items()
                   if k in current_inputs and current_inputs[k] != v]
        missing = [k for k in recorded if k not in current_inputs]
        if changed:
            problems.append(f"{len(changed)} input file(s) changed since the run "
                            f"(e.g. {changed[:3]})")
        if missing:
            problems.append(f"{len(missing)} recorded input file(s) no longer exist "
                            f"(e.g. {missing[:3]})")

    return ("bad" if problems else "ok"), problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate G5 -- run manifests")
    ap.add_argument("--strict", action="store_true",
                    help="treat runs with no manifest as failures")
    ap.add_argument("--require", action="append", default=[],
                    help="run directory name that MUST carry a valid manifest "
                         "(repeatable)")
    ap.add_argument("--dir", action="append", default=[],
                    help="check a run directory OUTSIDE results/ (repeatable). "
                         "`--smoke` writes to tools/out/smoke_logs, which the "
                         "results/* scan below cannot see; this is how you check "
                         "a smoke run. Found by the Claude Code verification "
                         "session, 8 Sep 2026.")
    args = ap.parse_args()

    print("[SAFETAIL][AUDIT][G5] verifying run provenance ...")

    # [SAFETAIL][AUDIT][G5] out-of-tree runs first -- these are named explicitly,
    # so a missing manifest here IS a failure (unlike the results/* scan, where
    # pre-gate runs are merely unprovenanced).
    # [SAFETAIL][AUDIT][G5] count as well as flag: the tally line below adds
    # this in, otherwise a --dir failure prints "0 failing" while exiting 1.
    extra_failed = False
    extra_failed_n = 0
    if args.dir:
        _cur = _current_input_hashes()
        for d in args.dir:
            p = Path(d)
            if not p.is_absolute():
                p = REPO / p
            if not p.is_dir():
                print(f"[SAFETAIL][AUDIT][G5][FAIL] {d}: not a directory")
                extra_failed = True; extra_failed_n += 1
                continue
            status, problems = check_one(p, _cur)
            if status == "unprovenanced":
                print(f"[SAFETAIL][AUDIT][G5][FAIL] {d}: no {MANIFEST_NAME}")
                extra_failed = True; extra_failed_n += 1
            elif status == "bad":
                print(f"[SAFETAIL][AUDIT][G5][FAIL] {d}:")
                for pr in problems:
                    print(f"    - {pr}")
                extra_failed = True; extra_failed_n += 1
            else:
                _ok(f"{d}: manifest complete, clean tree, no degradation")
    try:
        dirs = _run_dirs()
    except Fail as exc:
        print(f"\n[SAFETAIL][AUDIT][G5] FAIL: {exc}")
        return 1

    current_inputs = _current_input_hashes()
    _ok(f"hashed {len(current_inputs)} current input files (dataset/*.csv, models/*/*.pkl)")

    good, unprov, bad = [], [], {}
    for d in dirs:
        status, problems = check_one(d, current_inputs)
        if status == "ok":
            good.append(d.name)
        elif status == "unprovenanced":
            unprov.append(d.name)
        else:
            bad[d.name] = problems

    for name in good:
        _ok(f"{name}: manifest complete, clean tree, no degradation")

    if unprov:
        _warn(f"{len(unprov)} run(s) predate gate G5 and carry no manifest "
              f"(UNPROVENANCED, not gated): {unprov}")

    for name, problems in bad.items():
        print(f"[SAFETAIL][AUDIT][G5][FAIL] {name}:")
        for p in problems:
            print(f"    - {p}")

    for name in args.require:
        if name in unprov:
            bad.setdefault(name, []).append("required run has no manifest")
        elif name not in good and name not in bad:
            bad.setdefault(name, []).append("required run directory not found")

    failed = bool(bad) or extra_failed or (args.strict and bool(unprov))
    print(f"\n[SAFETAIL][AUDIT][G5] {len(good)} ok, {len(unprov)} unprovenanced, "
          f"{len(bad) + extra_failed_n} failing")
    print(f"[SAFETAIL][AUDIT][G5] {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
