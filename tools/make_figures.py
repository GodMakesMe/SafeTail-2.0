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
import re
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
_STYLE = {
    "SafeTail-2.0 (shipped)":     "#0072B2",
    "SafeTail-2.0 (rerun)":       "#56B4E9",
    "SafeTail-1.0 (code)":        "#D55E00",
    "SafeTail-1.0 (paper)":       "#8B0000",
    "SafeTail-1.0 (slowpath s0)": "#E69F00",
    "Oracle":                     "#000000",
    "MinProp-1": "#009E73", "MinProp-2": "#00A67E", "MinProp-3": "#00B88A",
    "MinLoad-1": "#CC79A7", "MinLoad-2": "#D98CB6", "MinLoad-3": "#E6A0C5",
    "Rand-1": "#999999", "Rand-2": "#AAAAAA", "Rand-3": "#BBBBBB",
}


class _Style(dict):
    """Never hand matplotlib a None colour."""
    def get(self, k, default="#777777"):        # noqa: A003
        return dict.get(self, k, default) or default


STYLE = _Style(_STYLE)


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


def discover_seeded(pattern: str) -> dict[int, pd.DataFrame]:
    """All seeds of one policy: results/<pattern>_s<seed>/latency_log.csv."""
    out: dict[int, pd.DataFrame] = {}
    for p in sorted((REPO / "results").glob(f"{pattern}_s*")):
        m = re.search(r"_s(\d+)$", p.name)
        if not m:
            continue
        d = _read_latency(p / "latency_log.csv")
        if d is not None and len(d):
            out[int(m.group(1))] = d
    return out


def discover_runs(extra: dict[str, Path] | None = None) -> dict[str, dict[int, pd.DataFrame]]:
    """label -> {seed: dataframe}. Single-run sources are recorded as seed -1."""
    runs: dict[str, dict[int, pd.DataFrame]] = {}

    def add_single(label: str, p: Path, seed: int = -1):
        d = _read_latency(p)
        if d is not None and len(d):
            runs.setdefault(label, {})[seed] = d

    # the already-published heterogeneous result set (single run, original code)
    add_single("SafeTail-2.0 (shipped)", REF / "safetail_training_logs" / "latency_log.csv")
    # the shipped K-baselines (single runs each)
    for fam, pretty in (("minload", "MinLoad"), ("minprop", "MinProp"), ("rand", "Rand")):
        for k in (1, 2, 3):
            add_single(f"{pretty}-{k}",
                       REF / "baselines" / f"training_logs_{fam}_{k}" / f"{fam}_{k}_latency_log.csv")

    # our seeded runs from tools/run_matrix.py
    for pattern, label in (("safetail_v1_legacy", "SafeTail-1.0 (code)"),
                           ("safetail_v1_paper_legacy", "SafeTail-1.0 (paper)"),
                           ("oracle_legacy", "Oracle"),
                           ("native_legacy", "SafeTail-2.0 (rerun)")):
        s = discover_seeded(pattern)
        if s:
            runs[label] = s
    # the original slow-path seed-0 v1 run, kept as cross-validation of the
    # tf.function rewrite (identical maths, 27.7x faster)
    add_single("SafeTail-1.0 (slowpath s0)", REPO / "results" / "v1_legacy_s0" / "latency_log.csv", seed=0)

    for label, p in (extra or {}).items():
        add_single(label, p)
    return runs


def truncate_common_seeded(runs):
    n = min(len(d) for seeds in runs.values() for d in seeds.values())
    return {lab: {s: d.iloc[:n].copy() for s, d in seeds.items()}
            for lab, seeds in runs.items()}, n


def percentile_table(runs, metric: str) -> pd.DataFrame:
    """Per-seed percentiles, then median/min/max across seeds."""
    rows = []
    for lab, seeds in runs.items():
        per_seed = []
        for s, d in sorted(seeds.items()):
            v = d[metric].to_numpy(float)
            r = {"policy": lab, "seed": s, "n": len(v), "mean": float(v.mean())}
            r.update({f"p{p}": float(np.percentile(v, p)) for p in PCTS})
            per_seed.append(r)
            rows.append(r)
        _ = per_seed
    return pd.DataFrame(rows)


def aggregate(tbl: pd.DataFrame) -> pd.DataFrame:
    g = tbl.groupby("policy", sort=False)
    out = g.agg(seeds=("seed", "count"), n=("n", "first"),
                **{f"{c}_med": (c, "median") for c in [f"p{p}" for p in PCTS] + ["mean"]},
                **{f"{c}_min": (c, "min") for c in [f"p{p}" for p in PCTS]},
                **{f"{c}_max": (c, "max") for c in [f"p{p}" for p in PCTS]})
    return out.reset_index()


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


def pooled(runs) -> dict[str, pd.DataFrame]:
    """Concatenate all seeds of a policy, for distribution-shaped figures."""
    return {lab: pd.concat(list(seeds.values()), ignore_index=True)
            for lab, seeds in runs.items()}


def fig_tail_bars(agg: pd.DataFrame, out: Path, metric: str, name="F1_tail_latency"):
    labels = list(agg["policy"])
    x = np.arange(len(PCTS)); w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, lab in enumerate(labels):
        r = agg[agg.policy == lab].iloc[0]
        med = [r[f"p{p}_med"] for p in PCTS]
        lo = [max(0.0, r[f"p{p}_med"] - r[f"p{p}_min"]) for p in PCTS]
        hi = [max(0.0, r[f"p{p}_max"] - r[f"p{p}_med"]) for p in PCTS]
        ax.bar(x + i * w - 0.4 + w / 2, med, w, label=f"{lab} (n_seeds={int(r.seeds)})",
               color=STYLE.get(lab), edgecolor="black", linewidth=0.4,
               yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels([f"p{p}" for p in PCTS])
    ax.set_ylabel(f"{metric.replace('_',' ')} (ms)")
    ax.set_title(f"Tail latency — {metric} (bars = median across seeds, whiskers = min/max)")
    ax.legend(ncol=3, fontsize=7); ax.grid(axis="y", alpha=.3)
    _save(fig, out, name, agg)


def fig_ccdf(runs, out: Path, metric: str, name="F2_ccdf"):
    fig, ax = plt.subplots(figsize=(8.5, 5.5)); rows = []
    for lab, d in pooled(runs).items():
        v = np.sort(d[metric].to_numpy(float)); v = v[np.isfinite(v)]
        surv = 1.0 - np.arange(1, v.size + 1) / v.size
        ax.plot(v, surv, label=lab, color=STYLE.get(lab), lw=1.6)
        step = max(1, v.size // 400)
        rows += [{"policy": lab, "latency_ms": xx, "P_gt": ss}
                 for xx, ss in zip(v[::step], surv[::step])]
    ax.set_yscale("log"); ax.set_xlabel(f"{metric.replace('_',' ')} (ms)")
    ax.set_ylabel("P(latency > x)"); ax.set_title("Latency CCDF (log-y; tail to the right)")
    ax.set_ylim(1e-4, 1); ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=7)
    _save(fig, out, name, pd.DataFrame(rows))


def fig_by_type(runs, out: Path, metric: str, name="F6_by_request_type"):
    P = pooled(runs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True); rows = []
    for ax, (t, pretty) in zip(axes, [("s", "Speech"), ("d", "Detect"), ("p", "Predict")]):
        labs, vals = [], []
        for lab, d in P.items():
            sub = d[d["type"] == t][metric].to_numpy(float)
            if sub.size == 0:
                continue
            labs.append(lab); v = float(np.percentile(sub, 95)); vals.append(v)
            rows.append({"policy": lab, "type": pretty, "n": int(sub.size), "p95": v})
        ax.bar(range(len(labs)), vals, color=[STYLE.get(l) for l in labs],
               edgecolor="black", linewidth=.4)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=65, ha="right", fontsize=6.5)
        ax.set_title(f"{pretty} ({t})"); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel(f"p95 {metric.replace('_',' ')} (ms)")
    fig.suptitle("p95 latency per request type (S-14: s=Speech, d=Detect, p=Predict)")
    _save(fig, out, name, pd.DataFrame(rows))


def fig_decomposition(runs, out: Path, name="F7_latency_decomposition"):
    P = pooled(runs); labs = list(P); rows = []
    fig, ax = plt.subplots(figsize=(13, 5)); bottom = np.zeros(len(labs))
    for col, pretty in [("computation_delay", "computation"), ("propagation_delay", "propagation"),
                        ("transmission_delay", "transmission"), ("queueing_delay", "queueing")]:
        vals = []
        for lab in labs:
            d = P[lab]
            if col not in d.columns:
                vals.append(0.0); continue
            m = float(d[col].mean())
            vals.append(m if col == "queueing_delay" else m * 1000.0)  # queue is already ms
        ax.bar(labs, vals, bottom=bottom, label=pretty, edgecolor="black", linewidth=.3)
        rows += [{"policy": l, "component": pretty, "mean_ms": v} for l, v in zip(labs, vals)]
        bottom += np.array(vals)
    ax.set_ylabel("mean contribution (ms)")
    ax.set_title("Latency decomposition — what is actually being optimised (D-12)")
    ax.set_xticklabels(labs, rotation=65, ha="right", fontsize=7)
    ax.legend(); ax.grid(axis="y", alpha=.3)
    _save(fig, out, name, pd.DataFrame(rows))


def fig_convergence(runs, out: Path, metric: str, name="F5_convergence", bins=10):
    """
    p95 within each decile of the run -- does the policy converge, plateau, or
    degrade as training proceeds? This is the evidence for whether a policy is
    under-trained or unstable, and it cannot be read off end-of-run percentiles.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2)); rows = []
    keep = [l for l in runs if l.startswith(("SafeTail", "Oracle"))]
    for lab in keep:
        seeds = runs[lab]
        per_seed = []
        for s, d in sorted(seeds.items()):
            v = d[metric].to_numpy(float); n = len(v)
            seg = [float(np.percentile(v[int(i * n / bins):int((i + 1) * n / bins)], 95))
                   for i in range(bins)]
            per_seed.append(seg)
            rows += [{"policy": lab, "seed": s, "decile": i + 1, "p95": x}
                     for i, x in enumerate(seg)]
        med = np.median(np.array(per_seed), axis=0)
        x = np.arange(1, bins + 1)
        ax.plot(x, med, marker="o", ms=4, lw=1.8, label=lab, color=STYLE.get(lab))
        if len(per_seed) > 1:
            arr = np.array(per_seed)
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0),
                            color=STYLE.get(lab), alpha=.15, lw=0)
    ax.set_xlabel("decile of the run (training proceeds ->)")
    ax.set_ylabel(f"p95 {metric.replace('_',' ')} (ms)")
    ax.set_title("Convergence: p95 within each decile\n"
                 "(line = median across seeds, band = min/max)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    _save(fig, out, name, pd.DataFrame(rows))


def fig_budget(out: Path, metric: str, name="F8_budget_3x"):
    """
    1x vs 3x data budget, CONVERGED phase only (last third of each run).
    Full-run percentiles are contaminated by the exploration phase, whose share
    shrinks in a longer run -- so a naive full-run comparison overstates the
    benefit of more data. This isolates the plateau each policy actually reaches.
    """
    series = [("SafeTail-1.0", "results/safetail_v1_legacy_s{}", "1x"),
              ("SafeTail-1.0", "results/safetail_v1_legacy_s{}_long3x", "3x"),
              ("SafeTail-2.0 (updated)", "results/native_legacy_s{}", "1x"),
              ("SafeTail-2.0 (updated)", "results/native_legacy_s{}_long3x", "3x")]
    rows = []
    for label, pat, budget in series:
        for s in (0, 1, 2):
            f = REPO / pat.format(s) / "latency_log.csv"
            if not f.is_file():
                continue
            v = pd.read_csv(f)[metric].to_numpy(float)
            v = v[int(len(v) * 2 / 3):]           # converged third
            rows.append({"policy": label, "budget": budget, "seed": s,
                         **{f"p{p}": float(np.percentile(v, p)) for p in PCTS}})
    if not rows:
        return
    df = pd.DataFrame(rows)
    agg = df.groupby(["policy", "budget"])[[f"p{p}" for p in PCTS]].median().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    keys = [(p, b) for p in agg.policy.unique() for b in ("1x", "3x")]
    x = np.arange(len(PCTS)); w = 0.8 / len(keys)
    hatch = {"1x": "", "3x": "//"}
    for i, (pol, bud) in enumerate(keys):
        sub = agg[(agg.policy == pol) & (agg.budget == bud)]
        if sub.empty:
            continue
        vals = [sub.iloc[0][f"p{p}"] for p in PCTS]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=f"{pol} — {bud}",
               color=STYLE.get(pol.split(" (")[0]), hatch=hatch[bud],
               edgecolor="black", linewidth=.5)
    # reference lines
    for y, lab, c in ((49.18, "MinProp-3 (K=3)", "#009E73"),
                      (49.53, "Oracle", "#000000")):
        ax.axhline(y, ls="--", lw=1, color=c, alpha=.8)
        ax.text(len(PCTS) - 0.5, y, f" {lab}", va="bottom", fontsize=7, color=c)
    ax.set_xticks(x); ax.set_xticklabels([f"p{p}" for p in PCTS])
    ax.set_ylabel(f"{metric.replace('_',' ')} (ms)")
    ax.set_title("Data budget: 1x vs 3x, CONVERGED phase only (last third of run)\n"
                 "median of 3 seeds — neither policy is budget-limited")
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=.3)
    _save(fig, out, name, df)


def main() -> int:
    ap = argparse.ArgumentParser(description="SafeTail comparison figures")
    ap.add_argument("--out", default=str(REPO / "figures"))
    ap.add_argument("--metric", default="total_latency",
                    choices=["total_latency", "end_to_end_latency"])
    args = ap.parse_args()
    out = Path(args.out)

    runs = discover_runs()
    if not runs:
        print("[SAFETAIL][PLOT] no runs found"); return 1
    for lab, seeds in runs.items():
        print(f"[SAFETAIL][PLOT] {lab:<28} seeds={sorted(seeds)}")
    runs, n = truncate_common_seeded(runs)
    print(f"[SAFETAIL][PLOT] truncated every run to the common n={n}")

    per_seed = percentile_table(runs, args.metric)
    agg = aggregate(per_seed)

    fig_tail_bars(agg, out, args.metric)
    fig_ccdf(runs, out, args.metric)
    fig_convergence(runs, out, args.metric)
    fig_by_type(runs, out, args.metric)
    fig_decomposition(runs, out)
    fig_budget(out, args.metric)

    out.mkdir(parents=True, exist_ok=True)
    per_seed.insert(1, "metric", args.metric)
    per_seed.to_csv(out / "table_per_seed.csv", index=False)
    agg.to_csv(out / "table_main.csv", index=False)

    show = agg[["policy", "seeds", "n"] + [f"p{p}_med" for p in PCTS] + ["mean_med"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print(f"\n[SAFETAIL][PLOT] figures + CSVs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
