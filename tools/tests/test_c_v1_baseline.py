"""
[SAFETAIL][POLICY] Regression tests for workstream C -- seam + oracle + SafeTail v1.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "baselines"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")


def _ctx(beta=5):
    from policy_registry import PolicyContext
    return PolicyContext(
        request_id=1, request_type="s", deadline=(100.0, 400.0),
        message_size=1024.0, bandwidth=20.0, beta=beta,
        free_slots=[3, 3, 0, 3, 3],
        est_delay=[0.05, 0.06, 0.9, 0.07, 0.04],
        est_components=[(0.02, 0.02, 0.01)] * beta,
        server_static=[{}] * beta, server_dynamic=[{}] * beta,
        arrival_time=0.0,
    )


def test_seam_subset_index_roundtrip():
    from policy_registry import index_to_subset, subset_to_index
    for beta in (3, 5):
        for idx in range(2 ** beta - 1):
            assert subset_to_index(index_to_subset(idx, beta), beta) == idx


def test_M01_oracle_selects_all_servers():
    from oracle.policy_oracle import OraclePolicy
    p = OraclePolicy()
    assert sorted(p.select(_ctx())) == [0, 1, 2, 3, 4]


def test_M02_v1_state_vector_shape_and_bounds():
    from safetail_v1 import config_v1 as cfg
    from safetail_v1.state_v1 import build_state
    v = build_state(_ctx())
    assert v.shape == (cfg.NS,) == (2 * 5 + 2,)
    load = v[:5]
    assert load.min() >= 1 and load.max() <= cfg.MAX_LOAD
    assert load[2] == cfg.MAX_LOAD               # free_slots[2] == 0 -> full -> MAX_LOAD


def test_M02_v1_reward_five_cases():
    from safetail_v1.reward_v1 import reward_v1
    beta, alpha = 5, 0.001
    # late (lam>0) with only 1 server -> penalty (should have replicated more)
    r_late_1, _ = reward_v1(0.20, 0.05, 1, beta, alpha)
    r_late_3, _ = reward_v1(0.20, 0.05, 3, beta, alpha)
    assert r_late_1 < 0 and r_late_3 < 0
    assert r_late_1 < r_late_3                    # fewer servers -> larger penalty when late
    # early (lam<0) -> penalty grows with |A| (wasted compute)
    r_early_1, _ = reward_v1(0.01, 0.05, 1, beta, alpha)
    r_early_4, _ = reward_v1(0.01, 0.05, 4, beta, alpha)
    assert r_early_4 < r_early_1
    # V1-BUG-01: out of band -> 0.0 + flag, never None
    r_oob, oob = reward_v1(5000.0, 0.05, 2, beta, alpha)
    assert r_oob == 0.0 and oob is True
    # all rewards <= 0
    for lat in (0.001, 0.05, 0.5):
        for k in range(1, 6):
            rr, _ = reward_v1(lat, 0.05, k, beta, alpha)
            assert rr <= 0.0


def test_M02_v1_policy_select_returns_valid_subset():
    from safetail_v1.policy_v1 import SafeTailV1Policy
    p = SafeTailV1Policy()
    sub = p.select(_ctx())
    assert 1 <= len(sub) <= 5 and all(0 <= i < 5 for i in sub)
    p.observe(_ctx(), sub, 0.0)                   # must not raise; ignores the passed reward
    assert p.report()["policy"] == "safetail_v1"
