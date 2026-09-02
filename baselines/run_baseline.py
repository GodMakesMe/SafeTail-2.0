"""
[SAFETAIL][SEAM] CLI entry point for running SafeTail 2.0 under an external policy.

Usage (from the repo root):

    python baselines/run_baseline.py --policy oracle [--smoke]
    python baselines/run_baseline.py --policy safetail_v1 --seed 1

What it does, in order:
  1. put src/ and baselines/ on sys.path (both INSIDE heterogenous/ -- G4d ok),
  2. set env POLICY=<name> so src/constants picks it up,
  3. import baselines.register  -> registers every baseline into the seam,
  4. hand off to src/main.main().

src/ is never imported by baselines/ except through policy_registry; this file
only *invokes* src/main as a program.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
BASELINES = REPO / "baselines"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SafeTail 2.0 env under a registered baseline policy")
    ap.add_argument("--policy", required=True, help="registered policy name, e.g. oracle | safetail_v1")
    ap.add_argument("--smoke", action="store_true", help="short deterministic run (plan.md 9.4)")
    ap.add_argument("--seed", type=int, default=None, help="override constants.SEED")
    args, passthrough = ap.parse_known_args()

    # 1. sys.path -- both entries are inside heterogenous/ (G4d)
    for p in (str(SRC), str(BASELINES)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 2. env for src/constants
    os.environ["POLICY"] = args.policy
    if args.smoke:
        os.environ["SAFETAIL_SMOKE"] = "1"
    if args.seed is not None:
        os.environ["SAFETAIL_SEED"] = str(args.seed)

    # 3. register baselines into the seam
    import register  # noqa: F401  (side effect: register_all())
    import policy_registry
    if args.policy not in policy_registry.available():
        raise SystemExit(
            f"[SAFETAIL][SEAM] policy {args.policy!r} not registered. "
            f"Known: {policy_registry.available()}"
        )

    # 4. hand off
    if args.smoke:
        sys.argv = [sys.argv[0], "--smoke", *passthrough]
    else:
        sys.argv = [sys.argv[0], *passthrough]
    import main as src_main
    src_main.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
