"""
[SAFETAIL][REGRESSOR] Regression tests for workstream B1 (D-02 / D-02b / D-02c).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")


def test_D02_each_server_loads_its_own_model():
    from regressors import TracePredictor, resolve_server
    for s in range(1, 6):
        tp = TracePredictor(s, "detect")
        assert tp.server_index == resolve_server(s)
        assert str(tp.df["Combination"].iloc[0])  # csv actually loaded
    # 2 -> 1 is the documented D-15 alias; everyone else is identity
    assert resolve_server(2) == 1
    for s in (1, 3, 4, 5):
        assert resolve_server(s) == s


def test_D02_computation_delay_is_heterogeneous():
    """Server(i).compute_request_time must give materially different comp delays."""
    from servers import Server
    from user import Request
    servers = [Server(i) for i in (1, 3, 4, 5)]
    req = Request(request_id=1, process_id=1, combination="s",
                  message_size=1024, bandwidth=20,
                  load=np.zeros(5, dtype=int), deadline=np.array([100, 400]))
    comps = []
    for srv in servers:
        # compute_request_time returns (total, combined_str, comp, prop, trans)
        _, _, comp, _, _ = srv.compute_request_time(req)
        comps.append(comp)
    # server1 (GPU) vs server3 (CPU-only) must differ by a lot
    assert comps[1] > comps[0] * 5, comps          # server3 >> server1
    assert comps[2] > comps[0] * 5, comps          # server4 >> server1
    assert abs(comps[1] - comps[2]) / max(comps[1], comps[2]) > 0.05  # 3 != 4
    assert abs(comps[0] - comps[3]) / max(comps[0], comps[3]) > 0.05  # 1 != 5


def test_D02b_no_sys_modules_collision():
    """Two TracePredictors for different servers are independent objects/data."""
    from regressors import TracePredictor
    a = TracePredictor(1, "speech")
    c = TracePredictor(3, "speech")
    assert a.model is not c.model
    va = a.predict_from_combination("s")
    vc = c.predict_from_combination("s")
    assert vc > va * 5, (va, vc)


def test_D02c_missing_contention_row_raises_not_silent():
    """A contention string absent from the trace must raise, not fall back to
    the contention-free single-letter latency."""
    from regressors import TracePredictor
    tp = TracePredictor(1, "speech")
    with pytest.raises(KeyError):
        tp.predict_from_combination("ssssss")   # length 6 > max concurrency, no such row


def test_D02c_degraded_path_is_off_by_default():
    import constants
    assert constants.ALLOW_DEGRADED_PREDICTORS is False
