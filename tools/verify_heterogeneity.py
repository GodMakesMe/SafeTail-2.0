#!/usr/bin/env python
"""
[SAFETAIL][REGRESSOR][FIX][D-02] Gate G1 -- computation latency is actually heterogeneous.

plan.md section 11. Blocks B1 sign-off.

Before B1 every one of the 15 wrappers loaded models/server1/ + dataset/server1.csv,
so Server(i).compute_request_time() returned server 1's computation delay for
every i. This gate asserts that is no longer true.

Checks (exit non-zero on failure):
  1. For a fixed contention string, the per-server COMPUTATION delay differs
     materially across the distinct hardware profiles.
  2. server 1 vs server 3 differ (GPU vs CPU-only, ~20x).           HARD
  3. server 3 vs server 4 differ.                                   HARD
  4. server 1 vs server 5 differ.                                   HARD
  5. server 1 == server 2 within tolerance -- reported as an ACKNOWLEDGED
     WARNING (D-15: byte-identical CSV, no distinct profile), NOT a failure.
  6. Every TracePredictor loads its OWN server index (except the documented
     D-15 alias 2 -> 1).                                            HARD

Usage:  python tools/verify_heterogeneity.py
"""
from __future__ import annotations

import statistics
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

REL_TOL = 0.05          # 5% -> "materially different"
CONTENTION_STRINGS = ["s", "d", "p", "sd", "dp", "sdp", "sdpd"]


class Fail(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"[SAFETAIL][AUDIT][G1][FAIL] {msg}")


def _ok(m: str) -> None:
    print(f"[SAFETAIL][AUDIT][G1][ok] {m}")


def _warn(m: str) -> None:
    print(f"[SAFETAIL][AUDIT][G1][warn] {m}")


def main() -> int:
    from regressors import TracePredictor, resolve_server

    print("[SAFETAIL][AUDIT][G1] verifying computation-path heterogeneity ...")

    # (6) each predictor loads its own server index (2 -> 1 by D-15)
    for s in range(1, 6):
        tp = TracePredictor(s, "speech")
        want = resolve_server(s)
        if tp.server_index != want:
            raise Fail(f"server {s}: TracePredictor loaded server{tp.server_index}, expected server{want}")
    _ok("each server loads its own model/CSV (2->1 is the documented D-15 alias)")

    # gather computation-delay estimates per server for several contention strings
    comp = {s: [] for s in range(1, 6)}
    for cs in CONTENTION_STRINGS:
        for s in range(1, 6):
            task = "speech" if cs[0] == "s" else ("detect" if cs[0] == "d" else "predict")
            comp[s].append(TracePredictor(s, task).predict_from_combination(cs))
    means = {s: statistics.fmean(v) for s, v in comp.items()}
    print("\n[SAFETAIL][AUDIT][G1] mean computation delay (ms) over "
          f"{CONTENTION_STRINGS}:")
    for s in range(1, 6):
        print(f"    server{s}: {means[s] * 1000:8.2f} ms   (loads server{resolve_server(s)})")

    def differ(a: int, b: int) -> bool:
        m = max(abs(means[a]), abs(means[b]), 1e-12)
        return abs(means[a] - means[b]) / m > REL_TOL

    # (2)(3)(4) hard: the distinct profiles must be distinct
    for a, b in [(1, 3), (3, 4), (1, 5), (1, 4), (3, 5)]:
        if not differ(a, b):
            raise Fail(f"server{a} and server{b} computation delays are within {REL_TOL:.0%} "
                       f"({means[a]*1000:.2f} vs {means[b]*1000:.2f} ms) -- D-02 not fixed")
    _ok("servers 1/3/4/5 give materially different computation delays (D-02 fixed)")

    # per-contention-string check that servers 1 and 3 always diverge
    for i, cs in enumerate(CONTENTION_STRINGS):
        m = max(abs(comp[1][i]), abs(comp[3][i]), 1e-12)
        if abs(comp[1][i] - comp[3][i]) / m <= REL_TOL:
            raise Fail(f"contention {cs!r}: server1 {comp[1][i]*1000:.2f} ms ~= "
                       f"server3 {comp[3][i]*1000:.2f} ms")
    _ok("server 1 vs server 3 differ for every tested contention string")

    # (5) D-15 acknowledged warning
    if differ(1, 2):
        raise Fail(f"server1 and server2 diverged ({means[1]*1000:.2f} vs {means[2]*1000:.2f} ms) -- "
                   f"unexpected: server2.csv is byte-identical to server1.csv (D-15)")
    _warn("server 1 == server 2 (D-15: byte-identical CSV, server2 has no distinct "
          "computation profile; it aliases server 1 -- this is acknowledged, not a bug)")

    print("\n[SAFETAIL][AUDIT][G1] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
