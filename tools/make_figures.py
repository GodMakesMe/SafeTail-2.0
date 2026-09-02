#!/usr/bin/env python
"""
[SAFETAIL][PLOT] Comparison figures: SafeTail 1.0 baseline vs the heterogeneous
SafeTail 2.0 results, plus the K-baselines and the oracle.

plan.md section 9.2. Rules enforced here:
  * ONE script, no notebooks in the publication path (a notebook hid D-03 --
    a hardcoded +5 ms added to the MinProp family only -- for months).
  * NO per-mode additive offsets. Anywhere. Gate G6 greps this file for them.
  * Every figure writes a companion .csv of the exact numbers plotted.
    If a number is in a figure, it must be in a CSV.
  * All runs are truncated to a COMMON request count before percentiles, so a
    run that happened to process more requests cannot win on sample size.

Usage:
    python tools/make_figures.py                      # auto-discovers runs
    python tools/make_figures.py --out figures/       # choose output dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "results" / "reference_v0"
PCTS = (50, 90, 95, 99)

# Order + colour is stable across every figure so panels can be read together.
STYLE = {
    "SafeTail-2.0 (het)": "#0072B2",
    "SafeTail-1.0":       "#D55E00",
    "Oracle":             "#000000",
    "MinProp-1": "#009E73", "MinProp-2": "#00A67E", "MinProp-3": "#00B88A",
    "MinLoad-1": "#CC79A7", "MinLoad-2": "#D98CB6", "MinLoad-3": "#E6A0C5",
    "Rand-1": "#999999", "Rand-2": "#AAAAAA", "Rand-3": "#BBBBBB",
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_latency(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "total_latency" not in df.columns:
        return None
    # request_type in reference_v0 holds the MUTATED contention string (D-19 was
    # not fixed then); post-fix runs hold the bare letter. First char works for both.
    if "request_type" in df.columns:
        df["type"] = df["request_type"].astype(str).str[0].str.lower()
    else:
        df["type"] = "?"
    if "end_to_end_latency" not in df.columns and "queueing_delay" in df.columns:
        df["end_to_end_latency"] = df["total_latency"] + df["queueing_delay"]
    return df


def discover_runs(extra: dict[str, Path] | None = None) -> dict[str, pd.DataFrame]:
    runs: dict[str, pd.DataFrame] = {}

    def add(label: str, p: Path):
        d = _read_latency(p)
        if d is not None and len(d):
            runs[label] = d

    # the already-published heterogeneous result set
    add("SafeTail-2.0 (het)", REF / "safetail_training_logs" / "latency_log.csv")
    # the shipped K-baselines
    for fam, pretty in (("minload", "MinLoad"), ("minprop", "MinProp"), ("rand", "Rand")):
        for k in (1, 2, 3):
            add(f"{pretty}-{k}",
                REF / "baselines" / f"training_logs_{fam}_{k}" / f"{fam}_{k}_latency_log.csv")
    # our new runs
    add("SafeTail-1.0", REPO / "results" / "v1_legacy_s0" / "latency_log.csv")
    add("Oracle", REPO / "results" / "oracle_legacy_s0" / "latency_log.csv")

    for label, p in (extra or {}).items():
        add(label, p)
    return runs


def truncate_common(runs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], int]:
    n = min(len(d) for d in runs.values())
    return {k: d.iloc[:n].copy() for k, d in runs.items()}, n


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _save(fig, out: Path, name: str, table: pd.DataFrame):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    table.to_csv(out / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"[SAFETAIL][PLOT] wrote {name}.png/.pdf + {name}.csv")


def fig_tail_bars(runs, out: Path, metric="total_latency", name="F1_tail_latency"):
    labels = list(runs)
    rows = []
    for lab in labels:
        v = runs[lab][metric].to_numpy(float)
        rows.append({"policy": lab, "n": len(v), **{f"p{p}": float(np.percentile(v, p)) for p in PCTS},
                     "mean": float(v.mean())})
    tbl = pd.DataFrame(rows)

    x = np.arange(len(PCTS)); w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, lab in enumerate(labels):
        vals = [tbl.loc[tbl.policy == lab, f"p{p}"].iloc[0] for p in PCTS]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=lab,
               color=STYLE.get(lab, None), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"p{p}" for p in PCTS])
    ax.set_ylabel(f"{metric.replace('_', ' ')} (ms)")
    ax.set_title(f"Tail latency by percentile — {metric} (n={len(next(iter(runs.values())))} per policy)")
    ax.legend(ncol=3, fontsize=8); ax.grid(axis="y", alpha=.3)
    _save(fig, out, name, tbl)
    return tbl


def fig_ccdf(runs, out: Path, metric="total_latency", name="F2_ccdf"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rows = []
    for lab, d in runs.items():
        v = np.sort(d[metric].to_numpy(float)); v = v[np.isfinite(v)]
        surv = 1.0 - np.arange(1, v.size + 1) / v.size
        ax.plot(v, surv, label=lab, color=STYLE.get(lab, None), lw=1.6)
        step = max(1, v.size // 400)
        for xx, ss in zip(v[::step], surv[::step]):
            rows.append({"policy": lab, "latency_ms": xx, "P_gt": ss})
    ax.set_yscale("log"); ax.set_xlabel(f"{metric.replace('_',' ')} (ms)")
    ax.set_ylabel("P(latency > x)"); ax.set_title("Latency CCDF (tail on the right, log-y)")
    ax.set_ylim(1e-4, 1); ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=8)
    _save(fig, out, name, pd.DataFrame(rows))


def fig_by_type(runs, out: Path, metric="total_latency", name="F6_by_request_type"):
    types = [("s", "Speech"), ("d", "Detect"), ("p", "Predict")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    rows = []
    for ax, (t, pretty) in zip(axes, types):
        labs, vals = [], []
        for lab, d in runs.items():
            sub = d[d["type"] == t][metric].to_numpy(float)
            if sub.size == 0:
                continue
            labs.append(lab); v = float(np.percentile(sub, 95)); vals.append(v)
            rows.append({"policy": lab, "type": pretty, "n": int(sub.size), "p95": v})
        ax.bar(range(len(labs)), vals, color=[STYLE.get(l) for l in labs],
               edgecolor="black", linewidth=.4)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{pretty} ({t})"); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel(f"p95 {metric.replace('_',' ')} (ms)")
    fig.suptitle("p95 latency per request type (S-14: s=Speech, d=Detect, p=Predict)")
    _save(fig, out, name, pd.DataFrame(rows))


def fig_decomposition(runs, out: Path, name="F7_latency_decomposition"):
    comps = [("computation_delay", "computation"), ("propagation_delay", "propagation"),
             ("transmission_delay", "transmission"), ("queueing_delay", "queueing")]
    labs = list(runs); rows = []
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(labs))
    for col, pretty in comps:
        vals = []
        for lab in labs:
            d = runs[lab]
            # component columns are in SECONDS except queueing_delay (ms)
            if col not in d.columns:
                vals.append(0.0); continue
            m = float(d[col].mean())
            vals.append(m if col == "queueing_delay" else m * 1000.0)
        ax.bar(labs, vals, bottom=bottom, label=pretty, edgecolor="black", linewidth=.3)
        for lab, v in zip(labs, vals):
            rows.append({"policy": lab, "component": pretty, "mean_ms": v})
        bottom += np.array(vals)
    ax.set_ylabel("mean contribution (ms)")
    ax.set_title("Latency decomposition — what is actually being optimised (D-12)")
    ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
    ax.legend(); ax.grid(axis="y", alpha=.3)
    _save(fig, out, name, pd.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeTail comparison figures")
    ap.add_argument("--out", default=str(REPO / "figures"))
    ap.add_argument("--metric", default="total_latency",
                    choices=["total_latency", "end_to_end_latency"],
                    help="decision 13.1(2): headline = service latency (total_latency)")
    args = ap.parse_args()
    out = Path(args.out)

    runs = discover_runs()
    if not runs:
        print("[SAFETAIL][PLOT] no runs found"); return 1
    print(f"[SAFETAIL][PLOT] runs: {list(runs)}")
    runs, n = truncate_common(runs)
    print(f"[SAFETAIL][PLOT] truncated all runs to the common n={n}")

    tbl = fig_tail_bars(runs, out, args.metric)
    fig_ccdf(runs, out, args.metric)
    fig_by_type(runs, out, args.metric)
    fig_decomposition(runs, out)

    tbl.insert(1, "metric", args.metric)
    tbl.to_csv(out / "table_main.csv", index=False)
    print("\n" + tbl.to_string(index=False))
    print(f"\n[SAFETAIL][PLOT] figures + CSVs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
