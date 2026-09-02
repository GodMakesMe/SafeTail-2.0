"""
[SAFETAIL][POLICY][REWARD][M-02] SafeTail 1.0's tau-referenced 5-case reward.

Thin re-export of src/rewards.tau_reward_5case -- the same primitive B4 uses in
the 2.0 reward path. src -> baselines is the allowed import direction.

Full spec: plan.md 14.1. V1-BUG-01: 1.0 returned None when abs(lam) >= 1000;
here it is 0.0 + the out_of_band flag.
"""
from __future__ import annotations

from rewards import TAU_BAND as BAND  # noqa: F401
from rewards import tau_reward_5case as reward_v1  # noqa: F401
