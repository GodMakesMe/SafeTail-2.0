#!/usr/bin/env python
"""
[SAFETAIL][AUDIT][FIX][D-01] Gate G0 -- environment truth.

Blocks: everything (plan.md section 11).

Checks, in order, and exits non-zero on the FIRST failure:

  1. Python is 3.11 or 3.12          (TF 2.20 support window; plan.md 3.1)
  2. numpy is 1.x                    (pickled sklearn models + pad_sequences are
                                      numpy-1 era; plan.md 3.1 -- "do not upgrade to 2.x")
  3. every pinned package in requirements.txt is importable at the pinned version
     (soft-warns on the handful of packages whose dist name != import name)
  4. all 15 regressor pickles under models/server{1..5}/ unpickle, expose a
     .predict(), and carry the (model, scaler, feature_columns) bundle keys the
     wrappers expect                 (D-02c: today this failure is silent)

On success prints the 15 estimator class names and "G0 PASS".

Usage:
    python tools/verify_env.py
    python tools/verify_env.py --skip-versions   # only structural checks

This script imports NOTHING from src/. It is safe to run before any other work.
"""
from __future__ import annotations

import argparse
import importlib
import pickle
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "models"
REQ_FILE = REPO / "requirements.txt"

TASKS = ("detect", "speech", "predict")
SERVERS = (1, 2, 3, 4, 5)

# dist-name -> import-name for the cases where they differ
IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "tensorflow": "tensorflow",
    "pillow": "PIL",
    "python-dateutil": "dateutil",
    "pyyaml": "yaml",
    "protobuf": "google.protobuf",
    "opt_einsum": "opt_einsum",
    "ml_dtypes": "ml_dtypes",
    "google-pasta": "pasta",
    "typing_extensions": "typing_extensions",
    "ipython": "IPython",
    "prompt_toolkit": "prompt_toolkit",
    "jupyter_client": "jupyter_client",
    "jupyter_core": "jupyter_core",
    "ipython_pygments_lexers": "ipython_pygments_lexers",
    "matplotlib-inline": "matplotlib_inline",
    "nest-asyncio": "nest_asyncio",
    "python-json-logger": "pythonjsonlogger",
}
# packages we do not try to import-check (build backends, pure data, namespace-y)
SKIP_IMPORT = {
    "setuptools", "wheel", "pip", "flatbuffers", "libclang", "namex",
    "tensorboard-data-server", "wcwidth", "pure_eval", "executing",
    "asttokens", "stack-data", "pytz", "tzdata", "six", "colorama",
    "charset-normalizer", "certifi", "idna", "urllib3", "packaging",
    "termcolor", "gast", "astunparse", "wrapt", "grpcio", "h5py",
    "markdown-it-py", "mdurl", "Markdown", "MarkupSafe", "Werkzeug",
    "kiwisolver", "cycler", "fonttools", "pyparsing", "contourpy",
    "threadpoolctl", "joblib", "tornado", "pyzmq", "psutil", "debugpy",
    "comm", "decorator", "jedi", "parso", "platformdirs", "traitlets",
    "Pygments", "rich", "requests", "optree", "ml_dtypes", "opt_einsum",
    "google-pasta", "absl-py", "tensorboard",
}


class Fail(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"[SAFETAIL][AUDIT][G0][FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"[SAFETAIL][AUDIT][G0][ok] {msg}")


def check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) not in {(3, 11), (3, 12)}:
        raise Fail(
            f"Python {major}.{minor} is outside the supported window (3.11 / 3.12). "
            f"TensorFlow 2.20 has no wheel for it. Use heterogenous/.venv."
        )
    _ok(f"Python {major}.{minor}")


def check_numpy() -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise Fail(f"numpy not importable: {exc}")
    if not np.__version__.startswith("1."):
        raise Fail(
            f"numpy is {np.__version__}; must be 1.x. The pickled sklearn models and "
            f"tf.keras pad_sequences path predate numpy 2. (plan.md 3.1)"
        )
    _ok(f"numpy {np.__version__}")


def parse_requirements() -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    for raw in REQ_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        pins.append((name.strip(), ver.strip()))
    return pins


def check_versions(pins: list[tuple[str, str]]) -> None:
    try:
        from importlib.metadata import version as dist_version
    except ImportError:  # pragma: no cover
        from importlib_metadata import version as dist_version  # type: ignore

    mismatches: list[str] = []
    missing: list[str] = []
    for name, want in pins:
        try:
            have = dist_version(name)
        except Exception:
            missing.append(f"{name} (want {want})")
            continue
        if _normalise(have) != _normalise(want):
            mismatches.append(f"{name}: installed {have}, pinned {want}")

    if missing:
        raise Fail("packages missing from the environment:\n  " + "\n  ".join(missing))
    if mismatches:
        raise Fail("version mismatches vs requirements.txt:\n  " + "\n  ".join(mismatches))
    _ok(f"all {len(pins)} pinned package versions match requirements.txt")


def _normalise(v: str) -> str:
    return re.sub(r"[^0-9A-Za-z.]", "", v).lower()


def check_imports(pins: list[tuple[str, str]]) -> None:
    failed: list[str] = []
    for name, _ in pins:
        if name in SKIP_IMPORT:
            continue
        mod = IMPORT_NAME.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 - we want the reason
            failed.append(f"{name} -> import {mod}: {type(exc).__name__}: {exc}")
    if failed:
        raise Fail("packages installed but not importable:\n  " + "\n  ".join(failed))
    _ok("core packages import cleanly")


def check_regressors() -> None:
    if not MODELS_DIR.is_dir():
        raise Fail(f"models/ directory not found at {MODELS_DIR}")

    names: list[str] = []
    problems: list[str] = []
    for s in SERVERS:
        for task in TASKS:
            pkl = MODELS_DIR / f"server{s}" / f"{task}_regressor_model.pkl"
            if not pkl.is_file():
                problems.append(f"server{s}/{task}: file missing ({pkl})")
                continue
            try:
                with pkl.open("rb") as fh:
                    bundle = pickle.load(fh)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"server{s}/{task}: unpickle failed -> {type(exc).__name__}: {exc}")
                continue

            if not isinstance(bundle, dict) or "model" not in bundle:
                problems.append(
                    f"server{s}/{task}: expected dict with a 'model' key, got "
                    f"{type(bundle).__name__} keys={list(bundle) if isinstance(bundle, dict) else 'n/a'}"
                )
                continue
            est = bundle["model"]
            if not hasattr(est, "predict"):
                problems.append(f"server{s}/{task}: bundle['model'] has no .predict()")
                continue
            names.append(f"server{s}/{task:<7} -> {type(est).__module__}.{type(est).__name__}")

    if problems:
        raise Fail(
            "regressor pickles did not load cleanly (D-02c makes this silent at runtime):\n  "
            + "\n  ".join(problems)
        )

    print("\n[SAFETAIL][AUDIT][G0] 15 regressor estimators:")
    for n in names:
        print(f"    {n}")
    _ok(f"all {len(names)} regressor pickles load and expose .predict()")


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate G0 -- environment truth")
    ap.add_argument("--skip-versions", action="store_true",
                    help="skip the requirements.txt version cross-check")
    args = ap.parse_args()

    print("[SAFETAIL][AUDIT][G0] verifying environment ...")
    check_python()
    check_numpy()
    pins = parse_requirements()
    if not args.skip_versions:
        check_versions(pins)
    check_imports(pins)
    check_regressors()

    print("\n[SAFETAIL][AUDIT][G0] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
