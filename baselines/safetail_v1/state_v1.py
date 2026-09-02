"""
[SAFETAIL][POLICY][M-02] SafeTail 1.0 state adapter: 2.0 PolicyContext -> 1.0 flat vector.

plan.md section 8.5 (A-1..A-4) + section 14.1 "State".

get_state_input(state) in 1.0 concatenates, in this order:
    LOAD[0..b-1], MESSAGE_SIZE, [RESOLUTION -- omitted, A-4], BANDWIDTH, PROPOGATION[0..b-1]
=> length 2*beta + 2.
"""
from __future__ import annotations

import numpy as np

from . import config_v1 as cfg


def build_state(ctx) -> np.ndarray:
    beta = ctx.beta

    # A-1: homogeneous integer LOAD, clipped to [1, MAX_LOAD]. Derived from
    # ctx.free_slots (free-slot count; -1 == full). active = MAX_LOAD - free.
    load = []
    for f in ctx.free_slots:
        active = cfg.MAX_LOAD if f == -1 else max(0, cfg.MAX_LOAD - int(f))
        load.append(min(cfg.MAX_LOAD, max(1, active)))

    # A-3: constant today (D-34); becomes meaningful after B5.
    msg = float(ctx.message_size)
    bw = float(ctx.bandwidth)

    # A-2: the SAME single propagation draw the environment used (D-18).
    prop = [float(c[1]) for c in ctx.est_components]

    vec = np.array([*load, msg, bw, *prop], dtype=float)
    assert vec.size == cfg.NS, (vec.size, cfg.NS)
    return vec
