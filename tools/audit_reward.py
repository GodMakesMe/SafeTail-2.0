#!/usr/bin/env python
"""
[SAFETAIL][REWARD][FIX] Gate G3 -- redundancy is priced (D-07, M-04, S-02) and the
step reward is well-behaved (D-16, D-17).

plan.md section 11. Blocks B3 / B8 sign-off.

Checks (exit non-zero on failure):

  D-07 / M-04
    * With c_red == 0, collapsing an EQUAL per-server headroom vector over
      |A| = 1..beta is FLAT -- adding a server no longer raises R_step (the old
      `np.mean` over 6 slots made it strictly increasing).
    * With c_red > 0, R_step is STRICTLY DECREASING in |A|: R_step(beta) < R_step(1).
      => the reward gradient now penalises redundancy.
  D-16
    * compute_step_reward is finite for a CPU-only server dict (no GPU keys) and
      for a GPU dict; equal headroom fractions give equal reward (geo-mean
      renormalisation), busier server strictly lower.
  D-17
    * a forced failure on server 0 zeroes rewards[0] (not rewards[-1]); all
      finite.

Usage:  python tools/audit_reward.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("SAFETAIL_SMOKE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np  # noqa: E402


class Fail(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"[SAFETAIL][AUDIT][G3][FAIL] {msg}")


def _ok(m: str) -> None:
    print(f"[SAFETAIL][AUDIT][G3][ok] {m}")


def _mk_controller():
    import constants
    constants.SMOKE = True
    constants.training_log_folder = str(REPO / "tools" / "out" / "g3_logs")
    Path(constants.training_log_folder).mkdir(parents=True, exist_ok=True)
    from controller import Controller
    return Controller(num_servers=constants.beta), constants


def _gpu_dict(cm=0.2, cu=0.2, gm=0.2, gu=0.2):
    cpu = np.array([cu * 100 * 8])
    return {"ram_usage": cm * 100 * 1024, "total_ram": 100.0,
            "cpu_usage": cpu, "cpu_core_usage": cpu, "total_cpu_cores": 8.0,
            "gpu_memory": gm * 8 * 1024, "total_gpu_memory": 8.0, "gpu_usage": gu * 100}


def _cpu_dict(cm=0.2, cu=0.2):
    cpu = np.array([cu * 100 * 8])
    return {"ram_usage": cm * 100 * 1024, "total_ram": 100.0,
            "cpu_usage": cpu, "cpu_core_usage": cpu, "total_cpu_cores": 8.0,
            "gpu_memory": 0.0, "total_gpu_memory": 0.0, "gpu_usage": 0}


class _Req:
    def __init__(self):
        self.request_id = 1
        self.combination = "d"
        self.contention_str = ""
        self.deadline = np.array([30.0, 200.0])
        self.queue_waiting_time = 1.0
        self.total_processing_delay = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        self.server_dicts = [{} for _ in range(6)]


def main() -> int:
    print("[SAFETAIL][AUDIT][G3] verifying reward shape + redundancy pricing ...")
    ctrl, constants = _mk_controller()
    beta = ctrl.num_servers

    # ---- D-07 / M-04 : redundancy pricing ---------------------------------
    equal = np.array([0.5] * beta + [0.0])          # 6-long, phantom slot 0
    constants.C_RED = 0.0
    flat = [ctrl._collapse_step_reward(equal, list(range(k))) for k in range(1, beta + 1)]
    if max(flat) - min(flat) > 1e-9:
        raise Fail(f"c_red=0: collapse over |A|=1..{beta} is not flat for equal headroom: {flat}")
    _ok(f"c_red=0: R_step flat in |A| for equal headroom ({flat[0]:.4f}); no pro-redundancy gradient")

    constants.C_RED = 0.10
    priced = [ctrl._collapse_step_reward(equal, list(range(k))) for k in range(1, beta + 1)]
    if not all(priced[i] > priced[i + 1] + 1e-9 for i in range(len(priced) - 1)):
        raise Fail(f"c_red=0.10: R_step is not strictly decreasing in |A|: {priced}")
    if not priced[-1] < priced[0]:
        raise Fail(f"c_red>0: R_step(|A|={beta}) not < R_step(|A|=1): {priced}")
    _ok(f"c_red=0.10: R_step strictly decreasing in |A|  {[round(x,3) for x in priced]}  (D-07/M-04 fixed)")
    constants.C_RED = 0.0

    # ---- D-16 : CPU-only server, geo-mean renormalisation ----------------
    r = _Req()
    r.server_dicts[0] = _gpu_dict(0.2, 0.2, 0.2, 0.2)
    r.server_dicts[1] = _cpu_dict(0.2, 0.2)
    rew = ctrl.compute_step_reward(r, action_subset=[0, 1])
    if not np.isfinite(rew).all():
        raise Fail(f"non-finite reward: {rew}")
    if abs(rew[0] - rew[1]) > 1e-6:
        raise Fail(f"D-16: equal headroom fractions give unequal reward gpu={rew[0]} cpu={rew[1]}")
    r.server_dicts[1] = _cpu_dict(0.9, 0.9)
    rew2 = ctrl.compute_step_reward(r, action_subset=[0, 1])
    if not rew2[1] < rew2[0]:
        raise Fail(f"D-16: busier CPU-only server did not score lower: {rew2}")
    idle = _Req(); idle.server_dicts[0] = _gpu_dict(0, 0, 0, 0)
    full = _Req(); full.server_dicts[0] = _gpu_dict(1, 1, 1, 1)
    hi = ctrl.compute_step_reward(idle, action_subset=[0])[0]
    lo = ctrl.compute_step_reward(full, action_subset=[0])[0]
    if not (abs(hi - np.log(2)) < 1e-6 and abs(lo) < 1e-6):
        raise Fail(f"D-16: reward range not [0, log2]: idle={hi} full={lo}")
    _ok("D-16: CPU-only servers not inflated; reward range [0, log2]; finite")

    # ---- D-17 : failure path zeroes the right slot ----------------------
    r = _Req()
    r.server_dicts[0] = {"ram_usage": 1.0}           # non-empty -> KeyError inside
    r.server_dicts[1] = _gpu_dict()
    rew = ctrl.compute_step_reward(r, action_subset=[0, 1])
    if rew[0] != 0.0 or not np.isfinite(rew).all() or rew[1] <= 0.0:
        raise Fail(f"D-17: failure path wrong: {rew}")
    _ok("D-17: reward failure zeroes the failing server's own slot")

    print("\n[SAFETAIL][AUDIT][G3] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
