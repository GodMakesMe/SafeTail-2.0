#!/usr/bin/env python
"""
[SAFETAIL][REPLAY][FIX] Gate G2 -- the RL training signal (D-04, D-05, D-06).

plan.md section 7 B2. Runs a short --smoke training run with SAFETAIL_AUDIT_REPLAY=1
so the controller records, for every stored transition, the realised per-server
delays of that same step, then asserts:

  D-04  no stored state array contains a value equal to a realised
        total_processing_delay of the SAME transition (beyond the trivial
        sentinels 0.0 / -1.0).
  D-05  state and next_state arrays are not element-wise identical
        (> 95% of sampled transitions).
  D-06  per-step rewards within an episode are not all identical
        (step_rewards.csv), i.e. the episodic reward did not overwrite them.

Usage:  python tools/audit_replay.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
os.environ["SAFETAIL_SMOKE"] = "1"
os.environ["SAFETAIL_AUDIT_REPLAY"] = "1"
os.environ.setdefault("SAFETAIL_SEED", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("SAFETAIL_SMOKE_CHUNKS", "90")     # ~30 episodes -> plenty of transitions
os.environ.setdefault("SAFETAIL_SMOKE_EPISODES", "30")

import numpy as np  # noqa: E402

TRIVIAL = {0.0, -1.0, 1.0}
SENTINEL_TOL = 1e-9


class Fail(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"[SAFETAIL][AUDIT][G2][FAIL] {msg}")


def _ok(m: str) -> None:
    print(f"[SAFETAIL][AUDIT][G2][ok] {m}")


def main() -> int:
    print("[SAFETAIL][AUDIT][G2] running instrumented smoke training ...")
    import controller as C
    import main as M

    C.AUDIT_TRANSITIONS.clear()
    M.run_smoke(seed=0, log_folder=str(REPO / "tools" / "out" / "g2_logs"))

    trans = list(C.AUDIT_TRANSITIONS)
    if len(trans) < 50:
        raise Fail(f"only {len(trans)} transitions captured; expected >= 50. "
                   f"Is the safetail DQN path storing arrays?")
    _ok(f"captured {len(trans)} instrumented transitions")

    sample = trans if len(trans) <= 400 else [trans[i] for i in
             np.linspace(0, len(trans) - 1, 400).astype(int)]

    # --- D-05 : state != next_state ---
    diff = 0
    for t in sample:
        a, b = t["s_t"], t["s_tp1"]
        if b is None:
            continue
        n = min(a.size, b.size)
        if not np.array_equal(a[:n], b[:n]) or a.size != b.size:
            diff += 1
    frac = diff / len(sample)
    if frac < 0.95:
        raise Fail(f"D-05: only {frac:.1%} of transitions have state != next_state (need > 95%)")
    _ok(f"D-05: state != next_state for {frac:.1%} of sampled transitions")

    # --- D-04 : no realised delay leaked into the stored state ---
    leaks = []
    for k, t in enumerate(sample):
        realised = t["realised_delay"]
        realised = realised[np.isfinite(realised)]
        realised = np.array([r for r in realised if abs(r) > 1e-6 and round(r, 6) not in TRIVIAL])
        if realised.size == 0:
            continue
        st = t["s_t"]
        for r in realised:
            hit = np.isclose(st, r, rtol=0, atol=1e-6)
            if hit.any():
                leaks.append((k, float(r)))
                break
    if leaks:
        raise Fail(f"D-04: {len(leaks)} sampled states contain a realised delay of their own step, "
                   f"e.g. {leaks[:5]}")
    _ok(f"D-04: no realised delay found in any sampled stored state ({len(sample)} checked)")

    # --- D-06 : per-step rewards vary within an episode ---
    srcsv = REPO / "tools" / "out" / "g2_logs" / "step_rewards.csv"
    if not srcsv.is_file():
        raise Fail(f"missing {srcsv}")
    import csv
    per_ep: dict[str, set] = {}
    with srcsv.open() as fh:
        for row in csv.DictReader(fh):
            per_ep.setdefault(row["episode"], set()).add(round(float(row["reward"]), 6))
    multi = [ep for ep, vals in per_ep.items() if len(vals) > 1]
    if not multi:
        raise Fail("D-06: every episode has a single distinct step-reward value "
                   "(episodic reward still overwriting per-step credit?)")
    _ok(f"D-06: {len(multi)}/{len(per_ep)} episodes have >1 distinct step reward")

    # bonus: the stored replay rewards themselves must vary
    rr = {round(t["reward"], 6) for t in trans}
    if len(rr) < 2:
        raise Fail("stored transition rewards are all identical")
    _ok(f"stored transition rewards take {len(rr)} distinct values")

    print("\n[SAFETAIL][AUDIT][G2] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
