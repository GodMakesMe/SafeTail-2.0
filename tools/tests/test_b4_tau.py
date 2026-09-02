"""
[SAFETAIL][REWARD] Regression tests for workstream B4 (M-03 -- tau tail-latency term).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")


def test_M03_tau_reward_is_nonpositive_and_asymmetric():
    from rewards import tau_reward_5case
    beta, alpha = 5, 0.005
    late_few, _ = tau_reward_5case(0.20, 0.05, 1, beta, alpha)
    late_many, _ = tau_reward_5case(0.20, 0.05, 4, beta, alpha)
    early_few, _ = tau_reward_5case(0.01, 0.05, 1, beta, alpha)
    early_many, _ = tau_reward_5case(0.01, 0.05, 4, beta, alpha)
    assert late_few < late_many <= 0        # late: fewer servers -> worse
    assert early_many < early_few <= 0      # early: more servers -> worse
    for lat in (0.001, 0.02, 0.05, 0.4):
        for k in range(1, beta + 1):
            r, _ = tau_reward_5case(lat, 0.05, k, beta, alpha)
            assert r <= 1e-12


def test_V1BUG01_out_of_band_returns_zero_not_none():
    from rewards import tau_reward_5case
    r, oob = tau_reward_5case(9999.0, 0.05, 2, 5, 0.005)
    assert r == 0.0 and oob is True
    r2, oob2 = tau_reward_5case(0.05, 0.05, 2, 5, 0.005)
    assert oob2 is False


def test_reward_mode_switch_changes_the_signal(tmp_path):
    """headroom vs tau vs headroom+tau produce different combined_step_reward."""
    import constants
    constants.SMOKE = True
    constants.training_log_folder = str(tmp_path)
    Path(constants.training_log_folder).mkdir(parents=True, exist_ok=True)
    from controller import Controller
    import numpy as np

    class R:
        pass
    def mk_req():
        r = R()
        r.request_id = 1
        r.combination = "s"
        r.contention_str = ""
        r.deadline = np.array([100.0, 400.0])
        r.queue_waiting_time = 1.0
        r.total_processing_delay = np.array([200.0, 210.0, 900.0, 220.0, 190.0])  # ms
        r.server_dicts = [{} for _ in range(6)]
        for i in range(5):
            cpu = np.array([20.0 * 8])
            r.server_dicts[i] = {"ram_usage": 20 * 1024, "total_ram": 100.0,
                                 "cpu_usage": cpu, "cpu_core_usage": cpu, "total_cpu_cores": 8.0,
                                 "gpu_memory": 1600.0, "total_gpu_memory": 8.0, "gpu_usage": 20}
        return r

    ctrl = Controller(num_servers=constants.beta)
    A = [0, 1, 3]
    hr = ctrl._collapse_step_reward(ctrl.compute_step_reward(mk_req(), A), A)
    t_only = ctrl._apply_tau_term(hr, mk_req(), A, 190.0, "tau")
    t_plus = ctrl._apply_tau_term(hr, mk_req(), A, 190.0, "headroom+tau")
    assert abs(t_only - hr) > 1e-6
    assert abs(t_plus - (hr + t_only)) < 1e-6


def test_constants_reward_mode_defaults_to_headroom():
    import importlib, constants
    importlib.reload(constants)
    assert constants.REWARD_MODE == "headroom"
    assert set("sdp") <= set(constants.TAU_BY_TYPE)
