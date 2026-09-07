"""
[SAFETAIL][REGRESSOR][FIX][D-02][D-02b][D-02c] One parameterised trace predictor.

Replaces the 15 copy-pasted `src/server{1..5}_regressor/*.py` wrappers, every one
of which hardcoded `models/server1/` + `dataset/server1.csv` (D-02), collided on
identical module names so `sys.modules` handed back server 1's class regardless
of `sys.path` (D-02b), and swallowed every load failure into a contention-free
CSV fallback (D-02c).

Now the server index is an argument, so:
  * server i loads models/server{i}/{task}_regressor_model.pkl and
    dataset/server{i}.csv  -- computation latency is finally heterogeneous;
  * there is one module, so no sys.modules collision is possible;
  * a load / lookup failure RAISES unless constants.ALLOW_DEGRADED_PREDICTORS.

### The three feature schemas (W-01)

The shipped per-server models do NOT share a feature schema:

  * servers 1, 5  -> "gpu" schema:  num_speech/num_detect/num_predict, total_ops,
                     position(0-based), is_first/is_last, peak_ram, peak_gpu,
                     peak_gpu_memory, total_processing_time   (the last is a
                     TARGET LEAK -- D-25 -- dropped only by a B1b retrain).
  * servers 3, 4  -> "cpu" schema:  num_tasks, position(1-based),
                     total_combination_length, has_{speech,detect,predict},
                     is_first/is_last, num_{speech,detect,predict}_tasks,
                     peak_ram, peak_cpu, avg_cpu_clock, num_files(=500).
  * server 2      -> its shipped model uses the "cpu" schema, but server 2's CSV
                     is byte-identical to server 1's (D-15) and carries GPU
                     columns, NOT the peak_cpu / avg_cpu_clock the model needs.
                     server 2 therefore ALIASES server 1 (documented, not hidden;
                     gate G1 reports the 1==2 duplicate as an acknowledged
                     warning).

`_build_features_*` here are faithful ports of the original wrappers'
`_build_features`; the schema is chosen from the model bundle's own
`feature_columns`, and features are selected by that list, so server2/speech's
`has_detect` vs server2/detect's `has_speech` etc. all resolve correctly.

B1b (not yet done): retrain to drop `total_processing_time` (D-25) and adopt
HED SS IV-D live-utilisation features (M-15); report held-out R2 before/after.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import pandas as pd

import constants

_REPO = Path(__file__).resolve().parent.parent
_MODELS = _REPO / "models"
_DATA = _REPO / "dataset"

TASK_FOR_LETTER = {"s": "speech", "d": "detect", "p": "predict"}
LETTER_FOR_TASK = {v: k for k, v in TASK_FOR_LETTER.items()}

# [SAFETAIL][REGRESSOR][D-15] server 2 has no distinct computation profile: its
# CSV is byte-identical to server 1's and its shipped model needs columns that
# CSV does not contain. Alias it to server 1.
SERVER_ALIAS = {2: 1}

TIME_COL = "Total Processing Time (sec)"   # the measured target in every trace

_GPU_COLS = {"peak_gpu", "peak_gpu_memory", "total_processing_time"}
_CPU_COLS = {"num_tasks", "peak_cpu", "avg_cpu_clock", "num_files"}


def resolve_server(server_index: int) -> int:
    """
    Map a logical server index (1-based) to the index whose model/CSV to load.

    [SAFETAIL][LEGACY][D-02] With constants.LEGACY_REGRESSORS, every server
    resolves to server 1 -- exactly the pre-fix behaviour that produced
    results/reference_v0/. Used to evaluate a baseline under the same physics as
    the already-published heterogeneous results.
    """
    try:
        import constants
        if getattr(constants, "LEGACY_REGRESSORS", False):
            return 1
    except Exception:
        pass
    return SERVER_ALIAS.get(int(server_index), int(server_index))


class TracePredictor:
    """Predicts computation delay (seconds) for one (server, task) from the trace CSV."""

    def __init__(self, server_index: int, task: str):
        task = task.strip().lower()
        if task not in LETTER_FOR_TASK:
            raise ValueError(f"[SAFETAIL][REGRESSOR] unknown task {task!r}; expected one of {list(LETTER_FOR_TASK)}")
        self.logical_index = int(server_index)
        self.server_index = resolve_server(server_index)   # after D-15 alias
        self.task = task
        self.letter = LETTER_FOR_TASK[task]

        model_path = _MODELS / f"server{self.server_index}" / f"{task}_regressor_model.pkl"
        csv_path = _DATA / f"server{self.server_index}.csv"
        if not model_path.is_file():
            raise FileNotFoundError(f"[SAFETAIL][REGRESSOR][D-02] model not found: {model_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"[SAFETAIL][REGRESSOR][D-02] csv not found: {csv_path}")

        with model_path.open("rb") as fh:
            bundle = pickle.load(fh)
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise TypeError(f"[SAFETAIL][REGRESSOR] {model_path} is not a (model, scaler, feature_columns) bundle")
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.feature_columns = list(bundle["feature_columns"])
        self.model_name = bundle.get("model_name", type(self.model).__name__)

        self.schema = "gpu" if (_GPU_COLS & set(self.feature_columns)) else "cpu"

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "Combination" not in df.columns:
            raise ValueError(f"[SAFETAIL][REGRESSOR] {csv_path} has no 'Combination' column")
        df["_combo_key"] = df["Combination"].astype(str).str.strip().str.lower()
        self.df = df.set_index("_combo_key", drop=False)

        # [SAFETAIL][REGRESSOR][D-40] How many measurements does this trace carry
        # per contention string? 1 == the averaged-at-collection-time dataset
        # that has no recoverable variance. >1 == repeated measurements, i.e. a
        # real distribution the simulator can sample from.
        _counts = self.df.index.value_counts()
        self.rows_per_combo_max = int(_counts.max()) if len(_counts) else 0
        self.rows_per_combo_mean = float(_counts.mean()) if len(_counts) else 0.0
        self.has_repeated_measurements = self.rows_per_combo_max > 1

    # ------------------------------------------------------------------ #
    def _row(self, combined_str: str) -> pd.Series:
        """
        [SAFETAIL][REGRESSOR][FIX][D-40] Pick ONE measurement for this contention
        string, SAMPLING when the trace has more than one.

        This used to be an unconditional `row.iloc[0]`: with repeated
        measurements it would silently use the first and discard the rest, so a
        newly-collected dataset with a real distribution would produce exactly
        the same zero-variance behaviour as the averaged one, and the data
        collection would look like it had achieved nothing.

        Modes (`constants.TRACE_SAMPLING`, env `SAFETAIL_TRACE_SAMPLING`):
          "sample" (default) -- draw uniformly among the matching rows. On a
                     single-row-per-combination trace this is IDENTICAL to the
                     old behaviour, so today's numbers are unchanged; the moment
                     repeated measurements exist it becomes the measured
                     computation-time distribution, which is what D-40 needs.
          "first"  -- always row 0. Exactly the pre-fix behaviour; use it to
                     reproduce an old run against a new dataset.
          "mean"   -- average the numeric columns across the repeats, i.e.
                     reproduce what the CURRENT dataset already is.

        Sampling goes through `random`, which `src/_seeding.py` seeds, so a run
        stays reproducible for a given SAFETAIL_SEED.
        """
        key = str(combined_str).strip().lower()
        if key not in self.df.index:
            raise KeyError(
                f"[SAFETAIL][REGRESSOR][D-02c] no trace row for contention string "
                f"{combined_str!r} in server{self.server_index}.csv (task={self.task}). "
                f"The contention-free single-letter fallback is DISABLED by design."
            )
        row = self.df.loc[key]
        if not isinstance(row, pd.DataFrame):
            return row                      # single measurement -- nothing to choose

        mode = str(getattr(constants, "TRACE_SAMPLING", "sample")).strip().lower()
        if mode == "first":
            return row.iloc[0]
        if mode == "mean":
            num = row.select_dtypes(include="number").mean()
            base = row.iloc[0].copy()       # keep the non-numeric columns
            for c in num.index:
                base[c] = num[c]
            return base
        return row.iloc[random.randrange(len(row))]

    def _build_features_gpu(self, row: pd.Series) -> dict:
        combination = str(row["Combination"]).lower()
        scripts = [s.strip().lower() for s in str(row["Scripts Executed"]).split(",")]
        idx = scripts.index(self.task)
        return {
            "num_speech": combination.count("s"),
            "num_detect": combination.count("d"),
            "num_predict": combination.count("p"),
            "total_ops": len(combination),
            "position": idx,                       # 0-based (original server1 wrapper)
            "is_first": 1 if idx == 0 else 0,
            "is_last": 1 if idx == len(scripts) - 1 else 0,
            "peak_ram": float(row.get("Peak RAM Usage (MB)", 0.0)),
            "peak_gpu": float(row.get("Peak GPU Usage (%)", 0.0)),
            "peak_gpu_memory": float(row.get("Peak GPU Memory (MB)", 0.0)),
            "total_processing_time": float(row.get("Total Processing Time (sec)", 0.0)),  # D-25 leak
        }

    def _build_features_cpu(self, row: pd.Series) -> dict:
        combination = str(row["Combination"]).lower()
        scripts = [s.strip().lower() for s in str(row["Scripts Executed"]).split(",")]
        idx = scripts.index(self.task)
        return {
            "num_tasks": len(scripts),
            "position": idx + 1,                   # 1-based (original server3 wrapper)
            "total_combination_length": len(combination),
            "has_speech": int("s" in combination),
            "has_detect": int("d" in combination),
            "has_predict": int("p" in combination),
            "is_first": int(idx == 0),
            "is_last": int(idx == len(scripts) - 1),
            "num_speech_tasks": combination.count("s"),
            "num_detect_tasks": combination.count("d"),
            "num_predict_tasks": combination.count("p"),
            "peak_ram": float(row.get("Peak RAM Usage (MB)", 0.0)),
            "peak_cpu": float(row.get("Peak CPU Usage (%)", 0.0)),
            "avg_cpu_clock": float(row.get("Average CPU Clock (MHz)", 0.0)),
            "num_files": 500,                       # original server3 wrapper hardcodes 500
        }

    def measured_values(self, combined_str: str) -> list[float]:
        """
        [SAFETAIL][REGRESSOR][D-40] Every MEASURED processing time for this
        contention string, straight from the trace. Empty if the string is
        absent or the column is missing.
        """
        key = str(combined_str).strip().lower()
        if key not in self.df.index or TIME_COL not in self.df.columns:
            return []
        sel = self.df.loc[key]
        vals = (sel[TIME_COL] if isinstance(sel, pd.DataFrame) else pd.Series([sel[TIME_COL]]))
        return [float(v) for v in vals.dropna().tolist()]

    def predict_from_combination(self, combined_str: str) -> float:
        """
        Computation time for one task under one contention string.

        [SAFETAIL][REGRESSOR][D-40] `constants.COMPUTATION_SOURCE` selects where
        the number comes from:

          "model" (DEFAULT, today's behaviour)
              Ask the fitted regressor. ⚠️ These are POINT predictors trained on
              squared error, so they output E[y|x] -- the conditional MEAN. That
              is why they cannot produce a tail even when the trace can: on
              server5's 'sd', the two measured values are 33.80 ms and 23.56 ms
              and the RandomForest returns 20.91 ms for BOTH. Handing such a
              model a distributional dataset does not give you a distribution;
              it gives you a better-estimated mean.

          "empirical"
              Sample uniformly among the MEASURED values for this contention
              string, falling back to the model only for strings the trace does
              not contain. This is the mode to use once repeated measurements
              exist: the variance is then measured, not assumed, and no
              retraining is required to obtain it.

        So D-40 has two halves and BOTH are needed:
          (1) data with repeats            -- being collected;
          (2) an inference path that does not average them away -- this.

        Default stays "model" so today's numbers are unchanged. Run
        `python tools/verify_dataset.py` (gate G8) when the new data lands; it
        tells you whether "empirical" is worth switching on.
        """
        source = str(getattr(constants, "COMPUTATION_SOURCE", "model")).strip().lower()
        if source == "empirical":
            vals = self.measured_values(combined_str)
            if vals:
                return vals[random.randrange(len(vals))] if len(vals) > 1 else vals[0]
            # unmeasured contention string -> fall through to the model
        row = self._row(combined_str)
        feats = self._build_features_gpu(row) if self.schema == "gpu" else self._build_features_cpu(row)
        missing = [c for c in self.feature_columns if c not in feats]
        if missing:
            raise KeyError(
                f"[SAFETAIL][REGRESSOR] server{self.server_index}/{self.task}: model expects "
                f"features {missing} that this builder ({self.schema}) does not produce"
            )
        X = pd.DataFrame([[feats[c] for c in self.feature_columns]], columns=self.feature_columns).astype(float)
        if self.scaler is not None:
            Xs = self.scaler.transform(X)
            X = pd.DataFrame(Xs, columns=self.feature_columns)  # keep names -> silences sklearn warning
        return float(self.model.predict(X)[0])


def load_all(server_index: int, allow_degraded: bool = False) -> dict:
    """Return {letter: TracePredictor|None} for one server. Raises unless allow_degraded."""
    out: dict[str, TracePredictor | None] = {}
    for letter, task in TASK_FOR_LETTER.items():
        try:
            out[letter] = TracePredictor(server_index, task)
        except Exception as exc:  # noqa: BLE001
            if not allow_degraded:
                raise RuntimeError(
                    f"[SAFETAIL][REGRESSOR][DEGRADED][D-02c] server={server_index} task={task} "
                    f"predictor failed to load"
                ) from exc
            try:
                from _safetail_log import get_logger, tag, note_degraded
                note_degraded("D-02c")
                get_logger("REGRESSOR").warning(
                    tag("DEGRADED", "D-02c", "server=%d task=%s load failed: %s"),
                    server_index, task, exc,
                )
            except Exception:
                print(f"[SAFETAIL][REGRESSOR][DEGRADED][D-02c] server={server_index} task={task}: {exc}")
            out[letter] = None
    return out
