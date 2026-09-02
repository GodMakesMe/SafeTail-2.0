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
    controller = src_main.main()

    # [SAFETAIL][POLICY][SNAPSHOT] persist the trained policy + its schedule.
    try:
        pol = getattr(controller, "policy", None)
        out = getattr(src_main.constants, "training_log_folder", ".")
        if pol is not None and hasattr(pol, "save_snapshot"):
            pol.save_snapshot(out, tag="final")
        if pol is not None and hasattr(pol, "report"):
            import json
            from pathlib import Path
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "policy_report.json").write_text(
                json.dumps(pol.report(), indent=2, default=str), encoding="utf-8")
            print(f"[SAFETAIL][POLICY] report -> {out}/policy_report.json")
    except Exception as e:  # noqa: BLE001
        print(f"[SAFETAIL][POLICY][SNAPSHOT] post-run save failed: {type(e).__name__} - {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
