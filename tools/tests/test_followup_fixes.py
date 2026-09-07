"""
[SAFETAIL][AUDIT] Regression tests for the follow-up repairs (plan.md 11:
"one test per fixed defect, named test_<ID>_<slug>").

Covers the defects that were still open after session 1:

    D-03   the rigged +5 ms MinProp offset in plotting/plot_baselines.ipynb
    D-32   access_rate_log.csv was a cumulative mean; baselines logged nothing
    M-14   generate_testing_plots() was `pass` and was never called
    S-16   the epsilon schedule had no single, correctly-named statement
    G5     no run carried a manifest.json

Several of these live inside `Controller`, which imports TensorFlow. Where an
assertion would otherwise need a full TF import just to read one line of logic,
the test asserts against the SOURCE instead -- a source-level guard is worth more
than a skipped test, because the failure mode being guarded against is somebody
reintroducing the old line.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
CONTROLLER = (SRC / "controller.py").read_text(encoding="utf-8")
CONSTANTS = (SRC / "constants.py").read_text(encoding="utf-8")
AGENT = (SRC / "agent.py").read_text(encoding="utf-8")


# --------------------------------------------------------------- D-03 --------

def test_D03_notebook_carries_no_minprop_offset():
    """
    plotting/plot_baselines.ipynb added a hardcoded +0.005 s to the MinProp
    family ONLY, in two cells, before percentiles were taken -- an ~11%
    inflation of one baseline family. plan.md B10.
    """
    nb_path = REPO / "plotting" / "plot_baselines.ipynb"
    if not nb_path.exists():
        pytest.skip("plotting/plot_baselines.ipynb absent")
    raw = json.dumps(json.loads(nb_path.read_text(encoding="utf-8", errors="replace")))
    assert not re.search(r"v\s*=\s*0\.005|\+\s*0\.005", raw), (
        "the D-03 MinProp offset is back in plot_baselines.ipynb"
    )


def test_D03_figure_script_has_no_per_mode_offset():
    """The publication path must never grow one either (gate G6)."""
    src = (REPO / "tools" / "make_figures.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"\+\s*0\.005|\bv\s*=\s*0\.005", code)


# --------------------------------------------------------------- D-32 --------

def test_D32_episode_access_rate_is_reset_each_episode():
    """
    The per-episode series must be cleared at the episode boundary, otherwise
    `np.mean(...)` over it is a cumulative (expanding) mean -- monotone by
    construction and therefore structurally unable to show the "adaptation"
    BTP 6.2 reads off the figure.
    """
    assert "self.episode_access_rates = []" in CONTROLLER
    tree = ast.parse(CONTROLLER)
    fin = next((n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "finalize_episode"), None)
    assert fin is not None, "finalize_episode not found"
    body = ast.get_source_segment(CONTROLLER, fin) or ""
    assert "self.episode_access_rates = []" in body, (
        "episode_access_rates is never reset inside finalize_episode -- D-32 is back"
    )


def test_D32_access_rate_column_is_not_the_cumulative_mean():
    """The headline value must come from the per-episode list, not the agent's
    never-cleared run-long list."""
    m = re.search(r"if len\(self\.episode_access_rates\) > 0:\s*\n\s*avg_access_rate = "
                  r"float\(np\.mean\(self\.episode_access_rates\)\)", CONTROLLER)
    assert m, "avg_access_rate is no longer computed from the per-episode series"


def test_D32_every_selection_branch_logs_access_rate():
    """
    External-policy, native SafeTail and fixed-K baseline branches must all log.
    Previously only the first two did, so every baseline run wrote an empty
    request_wise_access_log.csv and an all-zero access_rate_log.csv.
    """
    calls = CONTROLLER.count("self.log_request_access_rate_with_type(")
    assert calls >= 3, (
        f"expected >=3 access-rate logging call sites (policy / safetail / fixed-K), "
        f"found {calls}"
    )


# --------------------------------------------------------------- M-14 --------

def test_M14_generate_testing_plots_is_implemented_and_called():
    tree = ast.parse(CONTROLLER)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "generate_testing_plots"), None)
    assert fn is not None, "generate_testing_plots disappeared"
    body = [s for s in fn.body if not isinstance(s, ast.Expr)
            or not isinstance(getattr(s, "value", None), ast.Constant)]
    assert body and not all(isinstance(s, ast.Pass) for s in body), (
        "generate_testing_plots is still an empty stub (M-14)"
    )
    assert CONTROLLER.count("self.generate_testing_plots()") >= 1, (
        "generate_testing_plots is implemented but still never called"
    )


# --------------------------------------------------------------- S-16 --------

def test_S16_epsilon_schedule_has_one_correctly_named_constant():
    assert "epsilon_decay_step" in CONSTANTS, "the correctly-named constant is gone"
    assert re.search(r"^gamma_decay\s*=\s*epsilon_decay_step", CONSTANTS, re.M), (
        "gamma_decay must remain a pure alias of epsilon_decay_step"
    )
    assert "constants.epsilon_decay_step" in CONTROLLER, (
        "the controller should build the agent from the correctly-named constant"
    )
    # the decay itself stays SUBTRACTIVE -- it is what produced every run in
    # results/. Switching to BTP 4.4's multiplicative form would invalidate them.
    assert "self.epsilon -= self.epsilon_decay" in AGENT, (
        "the epsilon schedule changed shape; that invalidates every run in results/"
    )


def test_S16_constants_import_and_alias_agree():
    sys.path.insert(0, str(SRC))
    import constants as c  # noqa: PLC0415
    assert c.gamma_decay == c.epsilon_decay_step


# ----------------------------------------------------------------- G5 --------

def test_G5_manifest_is_written_and_complete(tmp_path):
    sys.path.insert(0, str(SRC))
    import constants as c  # noqa: PLC0415
    from _manifest import write_manifest  # noqa: PLC0415

    path = write_manifest(tmp_path, c, controller=None, started_at=None,
                          extra={"label": "pytest"})
    assert path is not None and Path(path).exists()
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("schema", "git", "seed", "constants", "packages",
                "inputs_sha256", "degraded", "publishable"):
        assert key in m, f"manifest is missing {key!r}"
    assert m["constants"], "constants dump is empty (plan.md 10.5)"
    assert m["inputs_sha256"], "no input file hashes recorded"
    assert isinstance(m["degraded"], dict)


def test_G5_gate_script_exists_and_is_importable():
    gate = REPO / "tools" / "check_manifest.py"
    assert gate.exists(), "gate G5 script is missing"
    ast.parse(gate.read_text(encoding="utf-8"))


def test_G5_run_directory_is_written_by_the_runner():
    """The runner must call write_manifest, or no run will ever have one."""
    main_src = (SRC / "main.py").read_text(encoding="utf-8")
    assert "write_manifest(" in main_src, (
        "src/main.py no longer writes a manifest -- every future run would be "
        "unprovenanced (plan.md 9.1)"
    )


# ----------------------------------------------------------------- D-11 ------

def test_D11_latency_metric_constant_is_actually_consumed():
    """
    constants.LATENCY_METRIC was declared as the B5 metric decision but read
    nowhere, so the documented knob did nothing.
    """
    figs = (REPO / "tools" / "make_figures.py").read_text(encoding="utf-8")
    assert "LATENCY_METRIC" in figs, (
        "constants.LATENCY_METRIC is dead again -- nothing consumes it"
    )


# --------------------------------------------------------------- D-38 --------
# Raised by the external defect dossier (Uttam, 3 Sep 2026), reproduced here.

def test_D38_P_of_T_has_the_leading_one_minus():
    """
    `P_T = (T - D1) / (D2 - D1)` -- the leading `1 -` was missing, so
    satisfaction INCREASED with lateness. BTP documentation.pdf 3.7 states
    `1 - (T-D1)/(D2-D1)`, and it is the one mechanism BTP genuinely got right
    (ERRATA S-01).
    """
    assert re.search(r"P_T\s*=\s*1\.?0?\s*-\s*\(T\s*-\s*D1\)\s*/\s*\(D2\s*-\s*D1\)",
                     CONTROLLER), "the P(T) sign inversion (D-38) is back"
    assert not re.search(r"P_T\s*=\s*\(T\s*-\s*D1\)\s*/\s*\(D2\s*-\s*D1\)", CONTROLLER)


def test_D38_P_of_T_is_continuous_and_decreasing():
    """
    The real guard: the three branches must agree at the boundaries. The broken
    form jumped 1 -> 0 at D1 and 1 -> 0 at D2, i.e. it was discontinuous at BOTH
    ends -- which is how you can tell it is a bug and not a design choice,
    without reading the report at all.
    """
    D1, D2 = 30.0, 200.0

    def P(T):
        if T <= D1:
            return 1.0
        if T <= D2:
            return 1.0 - (T - D1) / (D2 - D1)
        return 0.0

    assert P(D1) == pytest.approx(P(D1 + 1e-6), abs=1e-4), "discontinuous at D1"
    assert P(D2) == pytest.approx(P(D2 - 1e-6), abs=1e-4), "discontinuous at D2"
    xs = [0, 10, 30, 40, 115, 199, 200, 400]
    assert all(P(a) >= P(b) for a, b in zip(xs, xs[1:])), "P(T) must be non-increasing"
    assert P(40) == pytest.approx(0.9412, abs=1e-3), "the dossier's worked example"


def test_D39_regressor_applies_the_fitted_scaler():
    """
    Four shipped wrappers (server3/detect, server3/speech, server4/speech,
    server4/predict) fed UNSCALED features to a LinearRegression that was fitted
    on scaled ones -- RMSE ~3e10 s instead of ~0.6 s. Unifying the wrappers into
    one TracePredictor fixed it incidentally; this test keeps it fixed.
    """
    src = (SRC / "regressors.py").read_text(encoding="utf-8")
    assert "self.scaler.transform(" in src, "the fitted scaler is no longer applied"
    assert re.search(r"if\s+self\.scaler\s+is\s+not\s+None\s*:", src), (
        "scaler application is no longer guarded/uniform across servers"
    )


def test_D40_repeated_measurements_are_not_silently_discarded():
    """
    `_row()` used to be an unconditional `.iloc[0]`. With a distributional
    dataset it would have used the first measurement and thrown the rest away --
    so the new data collection would have produced no change at all, and looked
    like it had achieved nothing. This is the guard on that.
    """
    src = (SRC / "regressors.py").read_text(encoding="utf-8")
    assert "rows_per_combo_max" in src, "the repeat detector is gone"
    assert not re.search(r"if isinstance\(row, pd\.DataFrame\):\s*\n\s*row = row\.iloc\[0\]", src), (
        "regressors._row silently collapses repeated measurements to row 0 again (D-40)"
    )
    sys.path.insert(0, str(SRC))
    import constants as c  # noqa: PLC0415
    assert c.TRACE_SAMPLING in ("sample", "first", "mean")


def test_D40_legacy_env_pins_deterministic_row_choice():
    """SAFETAIL_LEGACY_ENV=1 must still reproduce results/reference_v0 exactly,
    so it cannot sample -- server5 has 6 contention strings with 2 rows each."""
    src = (SRC / "constants.py").read_text(encoding="utf-8")
    assert re.search(r'TRACE_SAMPLING\s*=\s*\(\s*"first"\s+if\s+LEGACY_ENV', src), (
        "legacy mode no longer pins row selection -- reference_v0 reproduction breaks"
    )


def test_D40_empirical_source_exists_and_is_not_the_default():
    """
    The regressors are POINT predictors: they return E[y|x]. Demonstrated on the
    only repeats the current data contains (server5 'sd': measured 33.80 and
    23.56 ms, model returns 20.91 for both). So a distributional dataset fed to
    the model still yields no distribution. `COMPUTATION_SOURCE="empirical"` is
    the inference path that does not average -- present, but NOT default, so
    today's numbers are unchanged.
    """
    sys.path.insert(0, str(SRC))
    import constants as c  # noqa: PLC0415
    assert hasattr(c, "COMPUTATION_SOURCE")
    assert c.COMPUTATION_SOURCE == "model", (
        "COMPUTATION_SOURCE default changed -- that silently changes every "
        "computation latency in the project. Deliberate? Then update D-40 and re-run."
    )
    src = (SRC / "regressors.py").read_text(encoding="utf-8")
    assert "def measured_values" in src and 'source == "empirical"' in src


def test_D40_computation_prediction_is_still_deterministic():
    """
    NOT a fix -- a WITNESS. Computation latency has no stochastic component at
    all (SafeTail 1.0 added `np.random.normal(0, st_dev[...])`; 2.0 dropped the
    line), and the traces cannot supply one: 363 rows, 363 unique contention
    strings, Iteration == [1]. There is no tail to optimise.

    This test asserts the CURRENT (broken) state deliberately, so that whoever
    restores the noise term is forced to come here, read this, and update the
    defect register rather than changing the physics silently.
    """
    import pandas as pd  # noqa: PLC0415
    d = pd.read_csv(REPO / "dataset" / "server1.csv")
    d.columns = [c.strip() for c in d.columns]
    assert d["Combination"].nunique() == len(d), (
        "server1.csv now has repeated contention strings -- if real repeats were "
        "collected, D-40 can finally be fixed properly. Update plan.md 4."
    )
    assert sorted(d["Iteration"].unique()) == [1], (
        "server1.csv now has more than one Iteration -- see the note above"
    )
    src = (SRC / "regressors.py").read_text(encoding="utf-8")
    assert "np.random.normal" not in src, (
        "a noise term appeared in regressors.py -- that CHANGES THE SIMULATOR. "
        "Fine, but it needs a defect-register entry and a full re-run; see D-40."
    )


# ------------------------------------------------- D-42 / D-43 / D-44 / D-45 --
# Second batch from the dossier, from pages 9-20 (the first read was truncated).

def test_D42_socket_run_cannot_block_forever():
    main_src = (SRC / "main.py").read_text(encoding="utf-8")
    assert "training_done.wait(timeout=" in main_src, (
        "the socket entry point waits without a timeout again (D-42) -- and the "
        "event it waits on may be unreachable, so the process hangs with partial "
        "results on disk"
    )
    assert "SAFETAIL_DRAIN_TIMEOUT" in main_src


def test_D43_episode_target_is_reachable():
    """
    no_of_episodes used to be no_of_chunk/episode_size = 25,000 while the sender
    could deliver at most no_of_burst*max_burst = 4,000 chunks, i.e. 1,333
    episodes at 3 chunks each. Unreachable by ~19x -- the root cause of D-42.
    """
    sys.path.insert(0, str(SRC))
    import constants as c  # noqa: PLC0415
    deliverable = min(c.no_of_chunk, c.no_of_burst * c.max_burst)
    reachable = deliverable // max(1, c.CHUNKS_PER_EPISODE)
    assert c.no_of_episodes <= reachable, (
        f"episode target {c.no_of_episodes} exceeds the {reachable} episodes the "
        f"sender can actually produce -- D-42 will hang again"
    )


def test_D43_episode_length_and_count_agree():
    """One constant drives both, so they cannot drift apart again."""
    sys.path.insert(0, str(SRC))
    import constants as c  # noqa: PLC0415
    assert hasattr(c, "CHUNKS_PER_EPISODE")
    assert "CHUNKS_PER_EPISODE" in CONTROLLER, (
        "Controller no longer defaults its episode length from the shared constant"
    )


def test_D44_request_stores_its_deadline():
    sys.path.insert(0, str(SRC))
    import numpy as np  # noqa: PLC0415
    from user import Request  # noqa: PLC0415
    r = Request(1, 0, "d", 1024, 20, np.zeros(5), np.array([30.0, 200.0]))
    assert len(r.deadline) == 2 and float(r.deadline[0]) == 30.0, (
        "the `deadline` constructor argument is being discarded again (D-44)"
    )
    # must stay None-safe: request_factory may not always supply one
    assert len(Request(2, 0, "s", 1024, 20, np.zeros(5), None).deadline) == 0


def test_D45_unit_conversion_is_explicit_not_a_magic_number():
    """
    latency_log.csv mixes seconds (components) and ms (totals) with no unit in
    the header. The figure script must convert through a named map, not a bare
    `* 1000.0` -- that is the D-03 failure mode.
    """
    figs = (REPO / "tools" / "make_figures.py").read_text(encoding="utf-8")
    assert "_TO_MS" in figs, "the explicit unit map is gone (D-45)"
    assert "D-45" in CONTROLLER, (
        "the mixed-units warning on the latency_log.csv header is gone (D-45)"
    )
