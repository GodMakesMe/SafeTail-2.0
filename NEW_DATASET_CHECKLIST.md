# When the new dataset arrives — do this, in this order

The current traces carry **one averaged measurement per contention string**
(363 rows, 363 unique combinations, `Iteration == [1]`, each already an average
over ~500 files). That is defect **D-40**: computation latency is deterministic,
so there is no tail to optimise and no tail-latency claim is admissible.

You are replacing that with repeated measurements — several rows for each of
`s`, `d`, `p`, `ss`, `sd`, `sp`, … **The code is now ready for it.** This file is
the sequence.

---

## 0. Two halves, and you need both

> **Collecting repeats is necessary but not sufficient.**

The shipped regressors are **point predictors fitted on squared error**, so they
return `E[y|x]` — the conditional **mean**. Proof, from the only repeats the
current data happens to contain (`server5.csv`, contention `'sd'`):

| | value |
|---|---|
| measured, row 0 | **33.7991 ms** |
| measured, row 1 | **23.5592 ms** |
| RandomForest predicts, for **both** | **20.9059 ms** |

Feed a distributional dataset to that model and you do not get a distribution —
you get a better-estimated mean. So:

* **half 1 — the data**: repeats with real spread → *you are doing this*
* **half 2 — the inference path**: stop averaging them → *done, see §3*

---

## 1. Format the rows so the loader can use them

The loader keys on the **`Combination`** column, lowercased and stripped. Repeats
are simply **multiple rows with the same `Combination` value**:

```csv
Combination,Scripts Executed,Iteration,Peak RAM Usage (MB),...,Total Processing Time (sec)
sd,"Speech, Detect",1,10407.16,...,0.033799
sd,"Speech, Detect",2,8599.03,...,0.023559
sd,"Speech, Detect",3,9233.41,...,0.028104
```

Requirements:

| | |
|---|---|
| **`Combination`** | the contention string, e.g. `s`, `sd`, `sdp`, `sdpd`. Repeats share it. |
| **`Iteration`** | 1, 2, 3, … per repeat. Currently constant 1 everywhere; make it the repeat index so repeats are identifiable. |
| **`Scripts Executed`** | must list the task names in order (`Speech`, `Detect`, `Predict`) — the feature builders call `scripts.index(task)`, so a mismatch raises. |
| **`Total Processing Time (sec)`** | the measured target, **in seconds**. |
| **telemetry columns** | keep the per-server schema unchanged: servers 1/2/5 carry the GPU columns, servers 3/4 the CPU ones. Adding columns is fine; **renaming or dropping them breaks the fitted bundles.** |
| **coverage** | every non-empty multiset over `{s,d,p}` of size 1–4 — 34 strings. The current traces already cover all 34; keep it that way or unmeasured strings raise at predict time. |
| **repeats** | aim for **≥ 10 per contention string**; ≥ 5 is the gate's floor. |

Do **not** pre-average. That is what produced D-40.

---

## 2. Check the data before training on it

```bash
python tools/verify_dataset.py                       # gate G8
python tools/verify_dataset.py --min-repeats 10 --strict
```

It reports, per file: rows, unique combinations, repeats min/median/max,
coverage of the 34 contention strings, schema, and the median coefficient of
variation within each contention string. It fails the dataset if there is one
row per scenario, or if repeats exist but carry no spread (the same number
copied is not a distribution).

On today's data it correctly reports servers 1–4 as inadequate — and flags that
`server5.csv` already has 369 rows for 363 combinations, i.e. six genuine
accidental repeats with a median CV of 0.057.

---

## 3. Turn on the inference path that does not average

```bash
export SAFETAIL_COMPUTATION_SOURCE=empirical     # Windows: $env:SAFETAIL_COMPUTATION_SOURCE="empirical"
```

Samples uniformly among the **measured** values for that contention string, and
falls back to the model only for strings the trace does not contain. **The
variance is then measured, not assumed, and no retraining is needed to get it.**

Two related switches:

| constant | default | what it does |
|---|---|---|
| `COMPUTATION_SOURCE` | `model` | `model` = fitted regressor (today). `empirical` = sample measured values. |
| `TRACE_SAMPLING` | `sample` | which repeat feeds the *model's* features: `sample` / `first` / `mean`. `SAFETAIL_LEGACY_ENV=1` pins `first` so `results/reference_v0` still reproduces exactly. |

Sampling goes through `random`, which `src/_seeding.py` seeds — a run stays
reproducible for a given `SAFETAIL_SEED`.

---

## 4. Sanity-check that a tail actually appeared

```bash
python - <<'PY'
import sys, warnings, random; warnings.filterwarnings('ignore'); sys.path.insert(0,'src')
from regressors import TracePredictor
random.seed(0)
p = TracePredictor(1, 'detect')
v = [p.predict_from_combination('sdp')*1000 for _ in range(200)]
import statistics as st
print(f"n=200  mean={st.mean(v):.3f} ms  stdev={st.pstdev(v):.3f} ms  "
      f"min={min(v):.3f}  max={max(v):.3f}  distinct={len(set(v))}")
PY
```

**Before:** `stdev = 0.000, distinct = 1`.
**After:** stdev > 0 and distinct > 1, or something upstream is still averaging.

Then run and check the ratio that matters:

```bash
SAFETAIL_SEED=0 python src/main.py --smoke
python tools/make_figures.py
```

The number to watch is **p99/p50**. On the old data it is **2.89×**; real
tail-latency problems are **5–50×**. If it has not moved, the data has not
changed anything and you need to find out why before running the full matrix.

---

## 5. Then retrain (B1b) — and only then

`empirical` mode gives you measured variance for **measured** contention
strings. The regressors are still needed to generalise to strings the trace does
not cover, and they still have two known problems:

* **D-25** — `total_processing_time` is a *feature* of a model predicting
  processing time. Drop it. The honest held-out R² **will fall**; that fall is
  the finding, not a regression.
* **M-15** — adopt HED §IV-D's feature design (live CPU/GPU/RAM utilisation +
  concurrency), **not** BTP §4.6's description of it. S-12: HED is the better
  specification here and BTP misdescribes what the code actually does.

With repeats available, consider fitting something distribution-aware rather
than another point predictor — quantile regression, or a mean model plus a
fitted residual distribution per contention string. Otherwise you are back to
predicting `E[y|x]`, which is exactly what §0 shows does not work.

Report held-out R² before and after, per server and task.

---

## 6. Re-run everything, and re-baseline

New physics means every number moves. In order:

1. `python tools/verify_env.py` (G0) and the other gates
2. re-run the SafeTail 2.0 side — **also required independently** by the D-38
   `P(T)` repair and the D-44 state change made on 8 Sep 2026
3. re-run SafeTail 1.0 (both variants), the Oracle and the K-baselines: the
   *environment* changed, so the old baseline rows are no longer comparable
4. `python tools/make_figures.py`, then `python tools/verify_figures.py` (G6)
5. `python tools/check_manifest.py` (G5) — the manifests will record the new
   dataset's SHA-256s, so runs before and after can never be confused

**`results/reference_v0/` stays frozen.** It is the published pre-fix run and is
byte-identical to what the advisor has plots for.

---

## 7. What this does and does not settle

Fixing D-40 gives the system a tail to optimise. It does **not** fix **D-41**:
propagation and compute speed are rank-correlated across the servers (Spearman
ρ = 0.95 on servers 1–4), so "pick the nearest" is still very nearly optimal and
MinProp-2 still lands within ~2 % of an all-5-server Oracle at p99.

To make the scheduling problem non-trivial you also need **at least one server
whose network distance disagrees with its compute speed**. The ping-trace →
machine assignment lives in `dataset/propagation_delays.pkl` and is arbitrary —
reassigning it is a small change with a large effect on whether there is
anything to learn.

Worth deciding at the same time as the data collection, since both are about the
testbed rather than the code.

---

*Written 8 September 2026. Companion to `FIXES_REPORT_SafeTail2.0_to_heterogenous.md`
§4b.3 (D-40) and `plan.md` §4.4b.*
