"""
[SAFETAIL][POLICY][REWARD][M-02][M-03] SafeTail 1.0's tau-referenced 5-case reward.

plan.md section 14.1 "Reward":

    lam   = obs_latency - tau
    gamma = |A| - 1                       # number of redundant servers
    delta = alpha * exp(-lam)  if lam < 0
            alpha * exp(+lam)  if lam >= 0

    lam == 0                      ->  R = 0
    lam  > 0 and (beta - gamma) == 1  ->  R = 0
    lam  > 0 and (beta - gamma)  > 1  ->  R = -exp(beta - gamma - 1) * delta
    lam  < 0                          ->  R = -exp(gamma) * delta

R <= 0 always. Late  => penalty scaled by how FEW servers were used (should have
replicated more); early => penalty scaled by how MANY were used (wasted compute).
This is the redundancy pricing SafeTail 2.0 lacks (M-04, D-07) -- porting it is
both a baseline AND the reference implementation for B4.

V1-BUG-01 (plan.md 8.6): 1.0 returns None when abs(lam) >= 1000, and that None
reaches the replay buffer. Ported as 0.0 + a counter.
"""
from __future__ import annotations

import math

BAND = 1000.0  # 1.0's abs(obs_latency - tau) < 1000 guard


def reward_v1(obs_latency: float, tau: float, action_size: int, beta: int, alpha: float):
    """Return (reward, out_of_band) where out_of_band is the V1-BUG-01 flag."""
    lam = float(obs_latency) - float(tau)
    if abs(lam) >= BAND:
        return 0.0, True  # V1-BUG-01: was `return None`

    gamma = action_size - 1
    if lam < 0:
        delta = alpha * math.exp(-lam)
    else:
        delta = alpha * math.exp(lam)

    if lam == 0:
        r = 0.0
    elif lam > 0 and (beta - gamma) == 1:
        r = 0.0
    elif lam > 0 and (beta - gamma) > 1:
        r = -math.exp(beta - gamma - 1) * delta
    elif lam < 0:
        r = -math.exp(gamma) * delta
    else:
        r = 0.0
    return float(r), False
