"""
[SAFETAIL][AUDIT] Regression tests for workstream B8 -- structural correctness.

One test per fixed defect, named test_<ID>_<slug> (plan.md 10.1). Run:

    .venv/Scripts/python.exe -m pytest tools/tests/test_b8_structural.py -q

These construct a real Controller (socket-free; __init__ only builds the agent +
servers, it does not bind any port).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("SAFETAIL_SMOKE", "1")          # keep it quiet + plot-free
os.environ.setdefault("TRAINING_LOG_FOLDER", str(REPO / "tools" / "out" / "pytest_logs"))


@pytest.fixture(scope="module")
def controller():
    import constants
    constants.SMOKE = True
    constants.training_log_folder = str(REPO / "tools" / "out" / "pytest_logs")
    Path(constants.training_log_folder).mkdir(parents=True, exist_ok=True)
    from controller import Controller
    return Controller(num_servers=constants.beta)


def _fake_request(combination="d", n_servers=6):
    """Minimal object with the attributes compute_step_reward touches."""
    class R:  # noqa: D401
        pass
    r = R()
    r.request_id = 123
    r.combination = combination
    r.contention_str = ""
    r.deadline = np.array([30.0, 200.0])
    r.queue_waiting_time = 5.0
    r.total_processing_delay = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
    r.server_dicts = [{} for _ in range(n_servers)]
    return r


def _gpu_dict(cm=0.2, cu=0.2, gm=0.2, gu=0.2):
    cpu = np.array([cu * 100 * 8])  # sum/100/cores == cu
    return {
        "ram_usage": cm * 100 * 1024, "total_ram": 100.0,
        "cpu_usage": cpu, "cpu_core_usage": cpu, "total_cpu_cores": 8.0,
        "gpu_memory": gm * 8 * 1024, "total_gpu_memory": 8.0,
        "gpu_usage": gu * 100,
    }


def _cpu_only_dict(cm=0.2, cu=0.2):
    cpu = np.array([cu * 100 * 8])
    return {
        "ram_usage": cm * 100 * 1024, "total_ram": 100.0,
        "cpu_usage": cpu, "cpu_core_usage": cpu, "total_cpu_cores": 8.0,
        "gpu_memory": 0.0, "total_gpu_memory": 0.0, "gpu_usage": 0,
    }


def test_D17_failure_path_zeroes_the_right_slot(controller):
    """A reward failure on server 0 must zero rewards[0], not rewards[-1]."""
    r = _fake_request()
    # server 0 dict is malformed (missing keys after the 'if not d' guard passes)
    r.server_dicts[0] = {"ram_usage": 1.0}            # non-empty -> passes 'if not d', then KeyError
    r.server_dicts[1] = _gpu_dict()
    rewards = controller.compute_step_reward(r, action_subset=[0, 1])
    assert rewards[0] == 0.0                          # THIS slot zeroed
    assert np.isfinite(rewards).all()
    assert rewards[1] > 0.0                           # neighbour untouched


def test_D16_cpu_only_server_not_inflated(controller):
    """CPU-only server must not out-score a GPU server at identical cm/cu."""
    r = _fake_request()
    r.server_dicts[0] = _gpu_dict(cm=0.2, cu=0.2, gm=0.2, gu=0.2)
    r.server_dicts[1] = _cpu_only_dict(cm=0.2, cu=0.2)
    rewards = controller.compute_step_reward(r, action_subset=[0, 1])
    assert np.isfinite(rewards[0]) and np.isfinite(rewards[1])
    # geometric-mean renormalisation => within ~1e-6 for equal headroom fractions
    assert abs(rewards[0] - rewards[1]) < 1e-6, (rewards[0], rewards[1])
    # and a busier CPU-only server must score strictly lower
    r.server_dicts[1] = _cpu_only_dict(cm=0.9, cu=0.9)
    rewards2 = controller.compute_step_reward(r, action_subset=[0, 1])
    assert rewards2[1] < rewards2[0]


def test_D16_reward_range_is_0_to_log2(controller):
    r = _fake_request()
    r.server_dicts[0] = _gpu_dict(cm=0.0, cu=0.0, gm=0.0, gu=0.0)   # fully idle
    r.server_dicts[1] = _gpu_dict(cm=1.0, cu=1.0, gm=1.0, gu=1.0)   # fully loaded
    rewards = controller.compute_step_reward(r, action_subset=[0, 1])
    assert rewards[0] == pytest.approx(np.log(2.0), abs=1e-6)
    assert rewards[1] == pytest.approx(0.0, abs=1e-6)


def test_D19_combination_not_mutated_by_schedule():
    """Server.schedule_request writes contention_str, never combination."""
    import constants
    from servers import Server
    from user import Request
    srv = Server(1)
    req = Request(request_id=1, process_id=1, combination="d",
                  message_size=1024, bandwidth=20,
                  load=np.zeros(5, dtype=int), deadline=np.array([30, 200]))
    srv.schedule_request(req, current_time=0.0)
    assert req.combination == "d", "combination must remain the raw type letter (D-19)"
    assert isinstance(req.contention_str, str) and req.contention_str != ""


def test_D21_saturation_is_bounded_and_counted(controller, monkeypatch):
    """All-full servers => bounded retries, one drop, no RecursionError."""
    monkeypatch.setattr(controller, "find_free_servers",
                        lambda: np.array([-1] * controller.num_servers))
    monkeypatch.setattr(controller, "SATURATION_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(controller, "SATURATION_BACKOFF_CAP", 0.0)
    before = controller.dropped_requests
    r = _fake_request()
    controller.process_step(r)                        # must return, not recurse
    assert controller.dropped_requests == before + 1


def test_D20_done_counter_is_deadline_conditional(controller):
    """With an unmeetable deadline, request_*_done stays below request_*_total."""
    controller.request_d_done = controller.request_d_total = 0
    controller.average_P_T_values = []
    r = _fake_request(combination="d")
    r.total_processing_delay = np.array([9999.0, 9999.0, 9999.0, 9999.0, 9999.0])  # far past D2=200
    r.queue_waiting_time = 0.0
    for sd in range(5):
        r.server_dicts[sd] = _gpu_dict()
    controller.compute_step_reward(r, action_subset=[0, 1, 2])
    assert controller.request_d_total == 1
    assert controller.request_d_done == 0            # deadline missed => not counted done
