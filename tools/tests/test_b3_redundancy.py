"""
[SAFETAIL][REWARD] Regression tests for workstream B3 (D-07, M-04, S-02).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")


def _controller():
    import constants
    constants.SMOKE = True
    constants.training_log_folder = str(REPO / "tools" / "out" / "pytest_logs")
    Path(constants.training_log_folder).mkdir(parents=True, exist_ok=True)
    from controller import Controller
    return Controller(num_servers=constants.beta), constants


def test_D07_cred_zero_is_flat_in_A():
    ctrl, constants = _controller()
    constants.C_RED = 0.0
    equal = np.array([0.5] * ctrl.num_servers + [0.0])
    vals = [ctrl._collapse_step_reward(equal, list(range(k))) for k in range(1, ctrl.num_servers + 1)]
    assert max(vals) - min(vals) < 1e-9, vals            # no pro-redundancy gradient


def test_D07_cred_positive_penalises_redundancy():
    ctrl, constants = _controller()
    constants.C_RED = 0.1
    equal = np.array([0.5] * ctrl.num_servers + [0.0])
    vals = [ctrl._collapse_step_reward(equal, list(range(k))) for k in range(1, ctrl.num_servers + 1)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), vals
    assert vals[-1] < vals[0]
    constants.C_RED = 0.0


def test_M04_collapse_is_mean_over_A_not_constant_6():
    """Selecting one good server must not be diluted by /6."""
    ctrl, constants = _controller()
    constants.C_RED = 0.0
    rewards = np.zeros(6)
    rewards[2] = 0.6
    # old code: np.mean(rewards) == 0.6/6 == 0.1 ; new: mean over {2} == 0.6
    assert abs(ctrl._collapse_step_reward(rewards, [2]) - 0.6) < 1e-9


def test_S02_constants_expose_c_red_and_sweep():
    import constants
    assert hasattr(constants, "C_RED") and isinstance(constants.C_RED, float)
    assert isinstance(constants.C_RED_SWEEP, list) and 0.0 in constants.C_RED_SWEEP
