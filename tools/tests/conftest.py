"""
[SAFETAIL][AUDIT] Shared pytest setup for the regression suite.

Windows-only hazard: scikit-learn/scipy and TensorFlow both ship an OpenMP
runtime (libiomp5md.dll). When one test module imports sklearn (via
src/regressors) and another imports tensorflow (via src/controller) in the SAME
process, TF's native runtime hits "DLL initialization routine failed". Setting
KMP_DUPLICATE_LIB_OK before either is imported is the standard workaround, and
importing tensorflow first makes it deterministic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("SAFETAIL_SMOKE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Win the DLL race: load TF before any test module pulls in sklearn.
try:  # pragma: no cover
    import tensorflow  # noqa: F401
except Exception:
    pass
