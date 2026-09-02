"""
[SAFETAIL][POLICY] Percentile / CDF helpers shared by baselines.

Pure numpy. No import from src/. plan.md section 8.2.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

PERCENTILES = (50, 90, 95, 99)


def percentiles(values: Iterable[float], ps: Sequence[int] = PERCENTILES) -> dict[int, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(arr, p)) for p in ps}


def ccdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_values, P[X > x]) for a complementary CDF / tail plot."""
    arr = np.sort(np.asarray(list(values), dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, arr
    n = arr.size
    surv = 1.0 - (np.arange(1, n + 1) / n)
    return arr, surv


def summary_row(name: str, values: Iterable[float]) -> dict[str, float | str]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    row: dict[str, float | str] = {"policy": name, "n": int(arr.size)}
    row.update({f"p{p}": v for p, v in percentiles(arr).items()})
    row["mean"] = float(np.mean(arr)) if arr.size else float("nan")
    return row
