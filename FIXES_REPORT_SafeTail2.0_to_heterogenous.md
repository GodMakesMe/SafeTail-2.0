# SafeTail 2.0 → `heterogenous`: what was fixed, and what is still open

**Scope.** Everything that changed between `SafeTail-2.0-main/` (the fork of
`github.com/shrutya22487/SafeTail-2.0`, preserved verbatim as commit `c93a859`)
and the current `heterogenous/` working tree — plus an honest register of what
remains unimplemented.

**Date:** 7 September 2026 · **Author:** Krishna Shukla · **Advisor:** Arani Bhattacharya

---

## 0. Ground rules for reading this

Three things constrain what this document may claim, and they are worth stating
before the tables:

1. **The BTP report (`documentation.pdf`) is the primary specification** — but it
   is not correct. It disagrees with the code in ~20 places, contradicts itself
   in 5, and its §6.2 headline claim is false on its own shipped data. Those
   disagreements are catalogued in `audit/BTP_REPORT_ERRORS.md`; where code and
   report differ, this report says which one was changed and why.
2. **The HED draft (`Heterogeneous_Edge_Devices____Tail_Latency.pdf`) is not
   authoritative.** It is an unfinished draft — its abstract is truncated
   mid-word ("*The reward consists of the su*", S-09). It is cited here only for
   *intent*, never as a requirement. Several BTP changes are genuine repairs of
   HED bugs and must **not** be reverted (provenance code `BTP-FIX`, rule R9).
3. **Baselines are out of scope for this document.** MinLoad/MinProp/Rand,
   SafeTail 1.0 and the Oracle appear only where a repair touched them. The
   subject here is the heterogeneous branch itself.

**Framing.** Krishna is not an author of the BTP report; he inherited the
codebase. The correct framing throughout is *"inherited codebase, corrected and
extended"* — not self-criticism.

**Scale of the change.** 74 files, **+5,386 / −277** lines against the seed
commit, ignoring line-ending noise. That splits into ~1,700 lines of repair and
new runtime modules in `src/`, ~2,500 lines of audit gates and tooling in
`tools/`, and ~1,200 lines of the isolated baseline unit in `baselines/`.

---

## 1. Executive summary

| | |
|---|---|
| Entries in the defect register (`plan.md` §4) | **47** — `D-01`…`D-45`, plus the sub-rows `D-02b`, `D-02c` |
| **Repaired in code** | **37** |
| **Resolved by retraction** (claim deleted rather than implemented) | **4** — D-13, D-22, D-24, D-26(a) |
| **Documented but not repaired** | **6** — D-25, D-26(b), D-37, **D-40**, **D-41**, **D-45** |
| **Symptom-level, closed by fixing a cause** | **2** — D-08 (via D-07), D-10 (via the run matrix) |
| Raised by the **external dossier** (Uttam), reproduced here | **8** — D-38, D-42, D-43, D-44 (fixed), D-39 (already fixed), D-40, D-41, D-45 (open/documented) — see §4b |
| Missing features (`M-01`…`M-15`) implemented | **7** — M-01, M-02, M-03, M-04, M-05, M-06, M-14 |
| Missing features deliberately deferred, each with a recorded disposition | **7** — M-07…M-13 |
| Missing features blocked on a decision | **1** — M-15 (with D-25) |
| Specification errata registered | **20** (`S-01`…`S-20`) |
| Audit gates | **8 of 8 green** — G5 and G8 both added this session (G8 advisory by design) |
| Regression tests | **48 collected, 48 passing** on the real machine (verified 8 Sep) |

**Two outcomes matter more than any individual bug fix.**

**First, the headline claim reverses.** BTP §6.2 asserts SafeTail 2.0 beats every
heuristic at every percentile. On the repository's own shipped data it loses to
MinProp at every percentile (p99 **115.6 ms vs 53.4 ms**), and once SafeTail 1.0
is implemented as its *paper* specifies rather than as its GitHub code does, 1.0
beats 2.0 too (p95 **68.66 ms vs 75.66 ms**).

**Second — and this arrived with the external dossier — the testbed cannot
demonstrate what the project set out to demonstrate.** Computation latency has
**zero variance** (the traces measured every scenario exactly once, and the
spread was averaged away at collection time), so p99/p50 is **2.89×** where real
tail-latency problems are 5–50×: there is no tail to optimise (**D-40**). And
network distance is rank-correlated with compute speed across the servers
(Spearman **ρ = 0.95** on servers 1–4), so "pick the nearest" is already very
nearly optimal and there is little for a scheduler to learn (**D-41**). Neither is
fixable by reward engineering; both are properties of the traces and the hardware
selection. See §4b.

Everything else in this report is secondary to those two.

---

## 2. The five defects that invalidated the published results

These are the S1 rows — the ones where the pre-fix code could not have produced
a valid result, regardless of how the experiment was run.

### 2.1 D-02 — the "heterogeneous" system was not heterogeneous

All **15** regressor wrappers (`src/server{1..5}_regressor/{speech,detect,predict}_predictor.py`)
resolved `models/server1/` and `dataset/server1.csv`. Every server predicted its
computation latency with server 1's model on server 1's trace.

Two mechanisms hid it:

* **D-02b** — the loader did `sys.path.insert(0, server{i}_regressor)` then
  `importlib.import_module("detect_predictor")`. The module *names* are identical
  across the five folders, so `sys.modules` returned server 1's already-imported
  module for servers 2–5 no matter what `sys.path` said. Fixing the paths alone
  would not have fixed the bug.
* **D-02c** — each import sat inside `except Exception: predictors['x'] = None`,
  and the fallback was a CSV lookup keyed on the bare letter (`Combination == "d"`),
  which returns the **contention-free** latency. With `sklearn` absent (D-01),
  all 15 loads failed, and the run completed normally producing plausible numbers.

**Fix.** `src/regressors.py` (new, 213 lines): one `TracePredictor(server_index, task)`
that loads the right model and CSV. Ports all three shipped feature schemas
("gpu" for servers 1/5, "cpu" for 3/4, and server 2 aliased to 1 because its CSV
is byte-identical — D-15). Failure now **raises** unless
`SAFETAIL_ALLOW_DEGRADED_PREDICTORS=1`, and the fallback is keyed on the full
contention string. Gate **G1** (`tools/verify_heterogeneity.py`) proves it:

```
server1:  12.45 ms   server2:  12.45 ms  (documented D-15 alias)
server3: 388.38 ms   server4: 266.42 ms   server5:  13.18 ms
```

Before the fix, all five read ≈12 ms. Servers 3 and 4 are genuinely ~20–30×
slower — they have no GPU at all — and the pre-fix code was using the fast GPU
server's regressor to predict their latency. That is not noise; it is a
systematic ~20× underestimate on **exactly the servers the scheduler most needs
to avoid**, which is the most likely mechanical reason the learned policy has no
usable computation signal and MinProp wins.

### 2.2 D-04 / D-05 / D-06 — the Bellman target was meaningless

Three separate bugs in one code path:

| | Before | Consequence |
|---|---|---|
| **D-04** | replay stored the **live `Request` object**; it was flattened only at `finalize_episode`, by which time `total_processing_delay`, `combination` and `queue_waiting_time` all held *realised* values | the state vector contained the **outcome of its own action** |
| **D-05** | the same dict literal put that one object in both `state` and `next_state` | the bootstrap was `r + γ·max Q(s)` on the same state |
| **D-06** | `finalize_episode` overwrote every step's reward with the single `episodic_reward` | per-step credit was discarded entirely |

**Fix.** `s_t` is snapshotted as an array at action time (pre-mutation), `s_{t+1}`
post-action, via a new `DQNAgent.store_arrays`; the per-step reward is kept and
the episodic bonus broadcast as `R_ep/N` on top. Gate **G2**
(`tools/audit_replay.py`) instruments a run: **450 transitions, 0 outcome leaks,
100 % `s_t ≠ s_{t+1}`, per-step rewards vary in 30/30 episodes.**

### 2.3 D-07 / D-08 — nothing anywhere priced redundancy

`compute_step_reward` zero-filled unselected slots, filled selected ones with
`log(1+headroom) ∈ [0, log 2]`, and the controller collapsed with `np.mean` over
**all five slots** (in fact six — `server_dicts` had a phantom entry, logged as
W-02). So `R_step = (1/5)·Σ_{i∈A} log(1+headroom_i)`: every extra server adds a
non-negative term to a fixed denominator. More servers was **always weakly
better**, and all three episodic terms pushed the same way.

The agent duly over-replicated (**D-08**): 73.9 % of requests went to exactly 4
servers, and mean K climbed monotonically **3.25 → 3.86** across the run. BTP
§6.2 reads that climb as "deadline adaptation". It is a climb up the reward
gradient.

**Fix (`_collapse_step_reward`).**

```
R_step = (1/|A|)·Σ_{i∈A} log(1 + headroom_i)  −  c_red·(|A|−1)/(β−1)
```

Two changes: the mean is over the **selected** servers, not a constant 6; and
`c_red` explicitly prices redundancy. `c_red = 0` reproduces the old reward
exactly. Gate **G3** sweeps |A| = 1…5 and asserts flat at `c_red=0`, strictly
decreasing at `c_red=0.1`.

**The provenance matters here, and it is a defensible narrative for the paper
(S-02).** HED's step reward `Σ(l_i − l̄) − W_step` was **sign-inverted** for its
own stated goal — maximising it selects the *most*-loaded servers — and HED
admits as much in the following paragraph. BTP correctly replaced the load term
with the headroom product. **But HED's form contained `d` (the redundancy count)
and `−W_step` explicitly, and BTP's contains no |A|-sensitive quantity at all.**
BTP fixed the sign and broke the redundancy pricing. So `c_red` is not a
restoration of HED — it is a new term on top of BTP's correct headroom product.

**Result:** mean K fell from **3.55 to ~2.87** (median over 3 seeds) with
`c_red = 0`. Collapsing over |A| instead of over a constant was sufficient on
its own; the explicit penalty was not even needed.

### 2.4 D-09 — the headline comparison was not budget-controlled

Baselines were hard-capped at K ∈ {1,2,3} by an `if`-chain; SafeTail ran free
over all 31 subsets at mean K ≈ 3.55, **exceeding the K=3 ceiling in 83.3 % of
episodes**. Its tail-latency win over SafeTail 1.0 came with **39 % more
compute**.

BTP §5.2 makes this worse on the report side (**S-11**): it describes the top-K
servers as "passed through the same downstream decision pipeline as SafeTail".
No such pipeline exists — `_select_minload_servers(x)` returns x servers and all
x are dispatched redundantly. The report describes a budget-neutral pre-filter;
the code is a fixed-K redundant dispatcher. A reader of §5.2 would not realise
the comparison is confounded.

**Fix.** Generic `{minload,minprop,rand}_{K}` dispatch with K ∈ 1…β (was
hardcoded 1…3), plus `SAFETAIL_MATCH_K` which caps SafeTail's per-request subset
— keeping the lowest-estimate servers, alternating floor/ceil so the *running
mean* lands on a fractional target. The realised mean K is now reported next to
every result.

### 2.5 D-01 — the documented install produced a broken system

`requirements.txt` was UTF-16LE with CRLF (pip cannot read it) and listed 72
packages, **none of them scikit-learn** — yet every `models/*/*.pkl` is a pickled
sklearn estimator. A clean install per the repo's own README produced a system
where D-02c fired silently on all five servers.

**Fix.** Re-encoded UTF-8/LF; added `scikit-learn==1.5.2`, `scipy==1.13.1`,
`seaborn==0.13.2`, `joblib`, `threadpoolctl`. Gate **G0** (`tools/verify_env.py`)
asserts the Python version, numpy 1.x, every pin, and that all 15 pickles load
*and* expose `.predict()`.

---

## 3. Complete defect ledger

Legend — **✅ fixed** · **📄 retracted** (claim deleted, not implemented) ·
**⚠️ documented only** · **⛔ open**

### 3.1 S1 — invalidates results

| ID | Defect | Status | Where |
|---|---|---|---|
| D-01 | `requirements.txt` UTF-16, no sklearn | ✅ | `requirements.txt`, G0 |
| D-02 | all 15 wrappers load `server1` | ✅ | `src/regressors.py`, G1 |
| D-02b | `sys.modules` name collision masks D-02 | ✅ | one module, no collision possible |
| D-02c | loader failure silent | ✅ | raises unless `ALLOW_DEGRADED_PREDICTORS`; counted in the manifest |
| D-03 | notebook adds +5 ms to MinProp only | ✅ **today** | both cells cleared; `tools/make_figures.py` is the publication path; G6 |
| D-04 | replay stores post-outcome state | ✅ | `controller.py`, `agent.store_arrays`, G2 |
| D-05 | `next_state == state` | ✅ | G2 |
| D-06 | step rewards discarded | ✅ | G2 |
| D-07 | reward monotone in \|A\| | ✅ | `_collapse_step_reward`, G3 |
| D-08 | agent over-replicates | ✅ *(symptom of D-07)* | mean K 3.55 → ~2.87 |
| D-09 | comparison not budget-controlled | ✅ | `{fam}_{K}` for K∈1…β, `SAFETAIL_MATCH_K` |
| D-10 | SafeTail 2.0 loses to MinProp on shipped data | ✅ *(finding, now measured)* | `figures/table_main.csv` |
| D-11 | `total_latency` excludes queueing | ✅ | both columns logged; `LATENCY_METRIC` selects the headline |
| D-12 | ~45 % of the metric is a 5-value coin flip | ✅ | transmission = f(payload, bandwidth) |

### 3.2 S2 — materially distorts results

| ID | Defect | Status | Where |
|---|---|---|---|
| D-15 | `server1.csv` ≡ `server2.csv` | ✅ documented alias | `regressors.SERVER_ALIAS`; G1 warns explicitly |
| D-16 | CPU-only servers get inflated headroom from missing GPU columns | ✅ | GPU factors dropped for CPU-only servers + geometric-mean renormalise; range stays `[0, log 2]` |
| D-17 | off-by-one: failure path writes `rewards[server_idx-1]` | ✅ | one-line fix + a test that forces the failure |
| D-18 | phase 2 and phase 7 each draw fresh delays | ✅ | `compute_request_time(reuse=True)` |
| D-19 | `request.combination` clobbered with the contention string | ✅ | separate `contention_str`; `request_type` back to {s,d,p} from 192 values |
| D-20 | `request_*_done` unconditional ⇒ completion ratio ≡ 1.0 | ✅ | deadline-conditional at the one site `T` is known |
| D-21 | unbounded recursion under saturation | ✅ | 6× bounded retry, exponential backoff, `dropped_requests` counter |
| D-22 | arrivals uniform, not Poisson | 📄 | claim deleted from README; S-06 in ERRATA. Implementing exponential inter-arrivals is a separate project. |
| D-23 | `queue_waiting_time` is wall-clock Python time | ✅ | simulated: saturation backoff + fixed dispatch cost. **This one mattered** — reported deadline satisfaction was a function of how fast the test machine is, and SafeTail paid a TF forward pass the heuristics did not. |

### 3.3 S3/S4 — correctness and hygiene

| ID | Defect | Status | Note |
|---|---|---|---|
| D-13 | README/BTP §3.5 assert M/M/1; no queue exists | 📄 | README rewritten with an Erlang-B correction banner |
| D-14 | `run_all_baselines.sh` hardcodes `/home/jyoti/miniconda3/bin/python`, wrong output paths | ✅ | venv-relative `$SAFETAIL_PY`, K 1…β, seed loop, `results/<mode>_k<K>_s<seed>_<sha>/` |
| D-24 | no queues anywhere — Erlang-B loss system | 📄 | restated, not faked |
| **D-25** | **regressor feature leaks the target** (`total_processing_time` predicts a component of itself) | ⚠️ **open** | documented in `regressors.py`; needs the B1b retrain |
| D-26 | "MLP regressors" are 6 Linear / 8 RF-DT / 1 GBM | 📄 (a) ⛔ (b) | documentation corrected; training actual MLPs is optional and not done |
| D-27 | docstring formula ≠ code formula | ✅ | one statement, `[0, log 2]` |
| D-28 | `assign_request` unpacks 3 of 7 values | ✅ | deleted (keeping a broken orphan invites someone to "fix" it into the live path) |
| D-29 | 11 orphan methods | ✅ | tagged `[DEAD][D-29]`; `generate_testing_plots` **implemented today** (M-14) |
| D-30 | dead constants `nS`, `episode_size`, `max_load` | ✅ | tagged with the reality each one misstates |
| D-31 | `np.load(allow_pickle=True)` on socket payloads = RCE | ✅ | loopback-only bind guard + 8 MiB payload cap |
| **D-32** | **access-rate log is a cumulative mean; baselines log all-zeros** | ✅ **today** | see §4.2 |
| D-33 | run processes ~15,225 requests, not 500,000 (BTP overstates ~33×) | ✅ | annotated at the constant |
| D-34 | requests carry no workload variation | ✅ | per-type payload sizes: s 16–64 KB, d/p 4–24 KB |
| D-35 | BTP §4.2 misdescribes the state | ✅ | annotated. **Note: the report understates its own system** — the input genuinely is variable-length, so the encoder contribution is legitimate |
| D-36 | emoji prints crash on Windows cp1252 | ✅ | UTF-8 stdout reconfigure |
| **D-37** | **ε floor is a guard, not a clamp** — the last step lands up to `epsilon_decay_step` *below* `epsilon_min` and stays there | ⚠️ **new, deliberately not fixed** | clamping shifts ε by ~1.5 % in every existing run. Fix with the next full re-run, not before. |

### 3.4 Raised by the external dossier — full write-up in §4b

| ID | Defect | Status | Where |
|---|---|---|---|
| **D-38** | **`P(T)` sign inverted** — satisfaction *increased* with lateness; the piecewise was discontinuous at both deadline boundaries | ✅ **fixed today** | `controller.py:486` + the comment block at `:449`. **Changes 2.0's numbers — needs a re-run.** |
| **D-39** | four predictors fed **unscaled** features to scale-sensitive linear models (RMSE ten orders of magnitude out) | ✅ *(fixed incidentally by B1)* | `regressors.py:185-187` applies the scaler uniformly for every server/task |
| **D-40** | **computation latency has zero variance — there is no tail to optimise.** Root cause is the traces (363 rows / 363 unique combos / `Iteration==[1]`; `server5.csv` 369/363); SafeTail 1.0's noise term was deleted in 2.0 | ⛔ **open — design decision** | measured p99/p50 = **2.89×** vs 5–50× for real tail problems |
| **D-41** | **ping is rank-correlated with compute speed** (Spearman ρ = 0.95 on servers 1–4) — MinProp is near-optimal by construction | ⛔ **open — property of the testbed** | reframes D-10: fixing D-02 does *not* make the environment learnable |

---

## 4. Fixes applied in this session (7 Sep 2026)

Six items that were still open, all with an unambiguous disposition already
recorded in `plan.md`. Every one has a regression test in
`tools/tests/test_followup_fixes.py` (12 tests, all green).

### 4.1 D-03 — the rigged plot offset · `plan.md` B10

`plotting/plot_baselines.ipynb` added a hardcoded **+0.005 s to the MinProp
family only**, in two cells, **before percentiles were taken**. At a mean total
latency of ~44 ms that is an **11 % inflation of one baseline family**.

It does not reverse D-10 — MinProp still wins by ~20 ms at p95 — but it is a
rigged comparison, and it survived for months precisely because it lived in a
notebook. Both offsets are now gone, with the removal recorded in place. Gate
**G6** previously emitted a standing warning about this file; it is now clean.

### 4.2 D-32 — the access-rate figure could not show what BTP claims

Two distinct bugs behind one column:

* `access_rate_log.csv` was `np.mean(self.agent.episode_access_rate)` where
  `agent.episode_access_rate` is **appended to for the whole run and never
  cleared**. That is a cumulative (expanding) mean: its final value is
  bit-identical to the overall request-wise mean, and **a cumulative average
  cannot be non-monotonic**. It is therefore *structurally incapable* of showing
  the adaptation-over-time that BTP §6.2 reads off the figure.
* The per-request logging call sat inside the SafeTail branch only, so every
  fixed-K baseline run wrote an **empty** `request_wise_access_log.csv` and an
  **all-zero** `access_rate_log.csv`.

**Fix.** A new `Controller.episode_access_rates`, appended by every selection
branch (external policy / native SafeTail / fixed-K baseline) and **reset in
`finalize_episode`**. The column is now a real per-episode series that can move
in both directions, and baseline runs populate it.

*No published figure changes:* `tools/make_figures.py` contains no reference to
the access-rate logs at all (verified by grep), so F1–F8 and `table_main.csv`
are untouched. The K statistics in `KNOWME.md` §3 came from
`request_wise_access_log.csv` — the per-request file, which was always correct
for SafeTail runs. Only the *episodic* column was wrong, and only baselines
wrote nothing.

### 4.3 M-14 — the train/test split produced no evidence

`generate_testing_plots()` was `pass` — **and was never called.** So the testing
segment, the only part run with ε = 0 and a frozen policy, was never shown on
its own; every plot averaged it into the training run.

Implemented: per-metric panels over the testing slice (episodic reward, latency,
access rate), sliced at the `testing_start_*_index` markers, written to
`<log_folder>/plots/testing/` **plus** a `testing_metrics.csv` — because the
plan.md §9.2 rule that gate G6 enforces for publication figures ("if a number is
in a figure, it must be in a CSV") should hold here too. Wired into
`finalize_episode` at run end.

### 4.4 S-16 — one ε schedule, correctly named

BTP §4.4 states a **multiplicative** decay `ε ← max(ε_min, ε(1−γ_ε))`; the code
has always done a **subtractive** step `ε -= gamma_decay`; and the constant's
name matched neither.

**Decision: keep the subtractive step.** It produced every run in `results/`;
switching schedules for cosmetic agreement with the report would silently
invalidate the whole comparison. BTP §4.4 is what is wrong. `epsilon_decay_step`
is now the name that describes it, `gamma_decay` is a deprecated alias so
`baselines/safetail_v1` and external scripts keep working.

While reading this path I found **D-37** (above) and deliberately left it alone.

### 4.5 D-11 — the metric knob did nothing

`constants.LATENCY_METRIC` was declared as the B5 metric decision and **consumed
nowhere**; `tools/make_figures.py` had its own independent default. The
documented knob was inert. `make_figures.py` now takes its default from it and
prints which column it used; `--metric` still overrides for a one-off.

### 4.6 G5 — run provenance, the last gate

`plan.md` §9.1: *"a result without a manifest is not a result."* Nothing wrote
one. `_safetail_log.degraded_report()` was written specifically for the manifest
and had no consumer, and `print_constants()` dumped the configuration to stdout
where it died with the terminal scrollback. Every directory under `results/` was
therefore un-provenanced: no machine-checkable record of which code, seed or
environment flags produced it.

**`src/_manifest.py`** (new) writes `manifest.json` per run: git SHA + **dirty
flag**, seed, full `constants` dump, tracked package versions, host, wall clock,
**SHA-256 of every `dataset/*.csv` and `models/*/*.pkl`**, and the `[DEGRADED]`
ledger. Stdlib-only, and every internal failure is caught — a bookkeeping
failure must never destroy a finished run.

**`tools/check_manifest.py`** (new, gate G5) enforces: manifest present and
parsing, required keys non-empty, git SHA recorded, **tree clean at run time**,
**`[DEGRADED]` ledger empty** (plan.md §10.3 — any degradation makes a run
unpublishable), and **input hashes still matching disk**, so a run cannot
silently be attributed to inputs that have since changed.

The 20 existing run directories are reported as **UNPROVENANCED** — not
failures, since they cannot retroactively grow a manifest, but listed explicitly
so nobody quotes them as gated. `--strict` fails on them.

### 4.7 Two stale ledgers, corrected

`CHANGELOG.md` stopped at `B9` and under-reported the work by **ten commits** —
B5, the figure pipeline, the run matrix, the packaging work and the S-18
retraction were all missing. `plan.md` §6.1 still listed B5, B7, D, G5 and G6 as
"still to do" when all five had closed. Both now reflect reality, and §6.1
distinguishes *closed* from *genuinely open*.

---

## 4b. The external defect dossier — four findings this register did not have

A second workstream (Uttam) audited the same inherited codebase independently and
produced *SafeTail Defect Dossier*, 3 September 2026. It is a good document: every
claim carries a file, a line number, a command and an expected output.

**It independently confirmed** D-02/02b/02c, D-03, D-04, D-05, D-06, D-16, D-17
and D-34 from its own reading — two people, two methods, the same defects. It also
raised **four findings this register did not have.** All four were reproduced
against this repository before being written down.

> **All 20 pages read.** The first copy was truncated to exactly 3 MiB with no
> `%%EOF`; the complete 8 MB file was recovered from the workspace folder. Both,
> plus page renders and an OCR transcript, are preserved in
> `audit/dossier_uttam/`. The PDF has no text layer (the type is vector
> outlines), so the pages were rendered and read.

> ⚠️ **The dossier uses its own `D-nn` numbering, which does not match ours.**
> Its `D-01` is our D-38, its `D-07` is our D-40, its `D-08` our D-41, its `D-10`
> our D-15, its `D-13`/`D-14` our D-42/D-43, its `D-15` our D-44. Always say
> which register a number belongs to. A merged register is worth an afternoon
> (§9).

### 4b.1 D-38 — `P(T)` had an inverted sign · **fixed today** · S1

The worst of the four, and the one that most deserved to be caught.

```python
# as it stood, src/controller.py:486
P_T = (T - D1) / (D2 - D1)          # the leading `1 -` is missing
```

Satisfaction **increased with lateness**. With `D1=30, D2=200`, a request
finishing at 40 ms scored **0.06 where it should score 0.94**.

You can tell it is a bug without opening a single document: the three branches
must meet at the boundaries, and these did not. The branch above returns **1** at
`T ≤ D1`; the broken middle branch returns **0** at `T = D1`. The branch below
returns **0** at `T > D2`; the broken middle returns **1** at `T = D2`.
**Discontinuous at both ends, in the same direction — inverted.** With the
leading `1 −` restored it is continuous at both and monotonically decreasing.

`P(T)` feeds `ω`, the degree of satisfaction, which is a term of the **episodic**
reward. So for the entire project the episodic signal was rewarding *missed*
deadlines.

The sting: **BTP §3.7 states the correct form explicitly**, and BTP's `P(T)` is
the one mechanism BTP genuinely got right — a correct repair of HED's malformed
ω (ERRATA **S-01**, and §7 of this report says so). The specification was right
and the code did not implement it. Both the formula and the comment block above
it (which called the increasing form "linearly decreasing") are now fixed, with
two regression tests: one greps for the sign, one asserts continuity and
monotonicity from first principles.

**⚠️ This changes numbers.** It alters the episodic reward, so the learned
2.0 policy will differ. Every SafeTail-2.0 result in `results/` predates it. The
baselines, the Oracle and SafeTail 1.0 are unaffected — none of them uses this
reward — so the *comparison* survives, but the 2.0 side needs a re-run.

### 4b.2 D-39 — four predictors fed unscaled features · **already fixed** · S1

`server3/detect`, `server3/speech`, `server4/speech` and `server4/predict` called
`model.predict(X)` without `scaler.transform(X)` — although the fitted scaler
was sitting in the same bundle and the model is a `LinearRegression`, which is
not scale-invariant. The dossier's calibration table:

| server / task | RMSE unscaled | RMSE scaled |
|---|---|---|
| server3 / detect | 3.34 × 10¹⁰ s | 0.604 s |
| server4 / speech | 3.75 × 10¹⁰ s | 0.174 s |

Not approximately wrong — **ten orders of magnitude out.**

**This branch already fixes it**, incidentally: `src/regressors.py:185-187`
applies the scaler uniformly for every server and task, because B1 replaced all
15 hand-written wrappers with one parameterised `TracePredictor`. It was never
registered as its own defect — it disappeared as a side effect of the
unification. It is now registered as **D-39** with a test, so it stays fixed.

### 4b.3 D-40 — there is no tail to optimise · **OPEN** · S1

The deepest finding in the dossier, and it undercuts the project's premise.

**Computation latency is deterministic.** Eight consecutive calls for
`server1/detect` on contention string `'ddd'`:

```
[2.4931, 2.4931, 2.4931, 2.4931, 2.4931, 2.4931, 2.4931, 2.4931]   stddev = 0.0
```

**The root cause is in the data, not the code.** `dataset/server{1,2,3,4}.csv`:
**363 rows, 363 unique contention strings, `Iteration == [1]`.** *(Correction,
8 Sep: `server5.csv` is the exception — 369 rows for 363 combinations, i.e. six
contention strings carry two genuine measurements each. See §4b.12.)* Every scenario
was measured exactly once, and each measurement is itself an average over ~500
files. The spread was averaged away at collection time and **cannot be recovered
from these CSVs.**

**And it is a regression, not an original gap.** SafeTail 1.0 added per-concurrency
Gaussian noise — `_spec_source/v1_agent.py:73`:

```python
return (pred + np.random.normal(0, st_dev[int(inp[0])-1], 1)) / 1000
```

with a 20-element table of *measured* per-concurrency standard deviations.
SafeTail 2.0 dropped the line. One deletion is why there is no tail.

Measured on the shipped run: **p99/p50 = 2.89×**. Real tail-latency problems are
5–50×. A system built to optimise tail latency is being evaluated on a workload
that barely has one.

**Not fixed — this is a design decision, not a bug.** Restoring noise changes
the simulator itself, and there are two honest routes:

* **Collect repeated measurements** (multiple `Iteration` values per contention
  string) and fit a real distribution. Correct, and the only route that makes the
  variance *measured* rather than *assumed*.
* **Restore SafeTail 1.0's noise term** using its measured `st_dev` table.
  Cheap and defensible as a stated modelling assumption — but the table was
  measured on 1.0's homogeneous hardware, not this testbed's.

`test_D40_*` asserts the current broken state deliberately, so whoever restores
the noise has to come read the entry and update the register rather than change
the physics silently.

### 4b.4 D-41 — MinProp is near-optimal by construction · **OPEN** · S1

| server | hardware | median ping | `detect` compute |
|---|---|---|---|
| 1 | Ryzen 9 7950X + RTX 4080 | 5.1 ms | 2.39 ms |
| 2 | *(aliases server 1, D-15)* | 12.6 ms | 2.39 ms |
| 3 | i7-11700, **no GPU** | 51.3 ms | 46.23 ms |
| 4 | Ryzen 5 5600GT, **no GPU** | 67.5 ms | 57.89 ms |
| 5 | i5-12600K + RTX 3090 | 71.2 ms | 2.72 ms |

**Spearman ρ = 0.95 across servers 1–4** (0.67 over all five). The nearest server
is also the fastest server, so "pick the nearest" is very nearly the optimal
policy — and MinProp-2 lands within ~2 % of an all-5-server Oracle at p99 in our
own runs. Server 5 is the single exception (far but fast), and is essentially the
only place a scheduler could beat MinProp.

**This changes the explanation of D-10.** §2.1 of this report attributes
MinProp's win primarily to D-02 — with computation identical across servers,
propagation was the only discriminating signal. D-41 shows that is not the whole
story: **even with D-02 fixed, propagation still ranks the servers almost exactly
as computation does.** Fixing the heterogeneity bug does not make the environment
learnable.

### 4b.5 D-42 / D-43 — the shipped configuration can never terminate · **fixed today**

`src/main.py` waited on `controller.training_done` with **no timeout**, and the
event it waits on cannot fire in the shipped configuration:

| | |
|---|---|
| episodes targeted (`no_of_episodes`) | `no_of_chunk / episode_size` = 100,000 / 4 = **25,000** |
| chunks the sender can deliver | `no_of_burst × max_burst` = 1000 × 4 = **4,000** |
| episodes reachable | 4,000 / `chunks_per_episode`(3) = **1,333** |

**Unreachable by ~19×.** The socket entry point — the one the README documents —
hangs forever with partial results already on disk. The dossier's conclusion is
correct and worth repeating: *the previous batch's full-length runs cannot have
terminated normally.*

**Our own results are not affected**, because every run in `results/` used the
in-process `--run` path added in this branch, which exits on its own chunk loop.
That is luck, not design.

Two repairs: the wait is now bounded (`SAFETAIL_DRAIN_TIMEOUT`, default 120 s)
and finalises rather than blocking (**D-42**); and the episode *count* is now
derived from the same `CHUNKS_PER_EPISODE` as the episode *length*, clamped to
what the sender can actually deliver (**D-43** — the arithmetic root cause).
The socket path also writes a G5 manifest now, like the in-process one.

### 4b.6 D-44 / D-45 — two smaller ones · **fixed / documented**

**D-44 — `Request.__init__` accepted `deadline` and threw it away.**
`self.deadline = np.asarray([], dtype=float)`, unconditionally. The two call
sites that wanted `request.deadline[0]/[1]` are commented out to this day —
because they could never have worked. Now stored (None-safe).

⚠️ **This changes the state vector.** `request_to_state_array` flattens *every*
numeric attribute (D-35), so a 2-element deadline adds 2 elements the network
never saw. Only the native 2.0 policy is affected — SafeTail 1.0 builds its own
flat state and the Oracle uses none — and D-38 already forces that re-run, so it
rides along at no extra cost.

**D-45 — `latency_log.csv` mixes units, and the header does not say so.**
`computation_delay` / `propagation_delay` / `transmission_delay` are in
**seconds**; `queueing_delay` / `total_latency` / `end_to_end_latency` are in
**milliseconds**. Verified on a real run: component means 0.0087 / 0.0532 /
0.0202 against a total of 82.16. Sum the components, compare to the total, and
you are out by 1000×.

`tools/make_figures.py` was compensating with a bare `m * 1000.0` and a
four-word comment — structurally the same failure mode as D-03. The conversion is
now an explicit named `_TO_MS` map and the header-writing site carries a warning.
**The columns are deliberately not renamed**: that would make every run under
`results/` unreadable. Unify at the next format break.

### 4b.7 D-15, extended — server 2 is worse than "a duplicate CSV"

Our register had D-15 as "`server1.csv` and `server2.csv` are byte-identical".
The dossier adds two facts that make it worse:

* **server 2's *models* are trained on the CPU feature schema** (`peak_cpu`,
  `avg_cpu_clock`) while `server2.csv` is a **GPU** trace with no CPU columns.
  The model and the trace describe **different machines.**
* `models/server2/detect_regressor_model.pkl` is **byte-identical to server 4's**.
* The BTP poster lists server 2 as an AMD Ryzen 5 7600X, 6 cores, no GPU.
  **That machine's trace is not in the repository at all.**

The two workstreams resolved this differently, and **the difference must not be
mixed inside one comparison**:

| | this branch | the dossier |
|---|---|---|
| server 2 | **aliased to server 1**, β = 5, documented, G1 warns | **excluded** via `ACTIVE_SERVERS`, β = 4 |

Both are defensible. β = 4 is arguably the more honest configuration, since it
does not invent a fifth machine's data — and it is what the dossier's numbers use.

### 4b.8 What the dossier found on the *repaired* environment

Worth knowing, because it is the experiment this branch has not run. On a fully
repaired environment (β = 4, real per-server compute), 100k requests, analysed
over the converged window:

**SafeTail 2.0's access rate came out at exactly 0.25 — one server out of four,
every request, no redundancy at all.** ε had been driven to 0 when
`_save_and_enter_testing` fired, so the policy froze on a single action. *For a
framework whose stated contribution is redundancy management, the learned policy
used none.*

This is not a contradiction of our mean-K ≈ 2.5–3.5 figures — it is the same
result seen in a different environment, and the dossier says so explicitly:

* **Legacy environment** (ours): all servers compute identically (D-02), so no
  server dominates and the policy spreads across many.
* **Repaired environment** (theirs): server 1 is fastest on *both* ping and
  compute (D-41), so one server is always right and the policy collapses onto it.

**The two workstreams are two halves of one result.** Our own recommended next
step — re-run on the D-02-repaired environment — is exactly the environment they
have already built and gate-tested.

### 4b.9 Where the two workstreams agree

They did not coordinate, and used different environments, different SafeTail 1.0
implementations and different analysis code. Convergent findings are the most
reliable material either of you has:

| Finding | this branch | the dossier |
|---|---|---|
| All 15 regressors load server 1 | D-02, audit + legacy mode | grep + unified loader |
| Transmission ignores payload; 5-value draw | D-12 / D-34 | verified in `servers.py` |
| MinProp beats every learned policy | MinProp-2 within 2 % of Oracle | `minprop_2` ties best at p99 |
| Severe seed instability | 2.0 re-run p50 76 vs 31 | blind/het winner flips by scale |
| Deadline satisfaction cannot discriminate | ω ≡ 1.00 for every policy (D-20) | ω ≠ 1.00 for most policies |
| Workload carries no variation | D-34 | `main.py:102-103` |
| **BTP §6.2's headline claim is false on this data** | S-17 / D-10 | same conclusion |
| **Eq. 6, not Bellman, is SafeTail 1.0's learning rule** | S-18, commit `909c387` | `safetail1_agent.py` |

**The strongest single result in either workstream:** two independent
implementations of SafeTail 1.0's Eq. 6 — written from the paper without
reference to each other — agree *semantically*. One-hot on zero reward;
otherwise a uniform base of `1/(2ⁿ−1)`, every action whose server set is a subset
of the chosen one set to `max(0, base + R)`, remaining mass spread equally.
Compare `baselines/safetail_v1/paper_v1.py` against their `src/safetail1_agent.py`.

That settles S-18 as firmly as it can be settled: **SafeTail 1.0 as published is
not architecturally broken — its GitHub implementation is.** Any comparison
against "SafeTail 1.0" must use Eq. 6.

### 4b.10 One housekeeping warning from the dossier

> *"The `krishna_work/` folder circulated separately is a superseded package,
> built 2 September 18:12 UTC. Its report still carries the withdrawn 'SafeTail
> 2.0 beats 1.0' headline and the retracted architectural claim."*

Correct — and it applies to `dist/` in this repo too. **Anything packaged before
the 3 September retraction carries a headline that has since been withdrawn.**
Re-package or delete those zips before sending anything to anyone.

### 4b.12 D-40b — repeated measurements alone will not fix D-40

Found while preparing the code for the incoming dataset (8 Sep), and it corrects
two things in this report.

**Correction 1 — `server5.csv` is not `363/363`.** It has **369 rows for 363
combinations**: `s`, `d`, `p`, `ss`, `sd`, `sp` each carry **two genuine
measurements** — different processing times *and* different telemetry, not
duplicated rows. E.g. `'sd'` = 0.033799 s vs 0.023559 s, a ~30 % spread, median
CV across the six ≈ 0.057. Servers 1–4 are exactly 363/363. Earlier statements
in this report generalising "363/363" to server 5 were wrong.

That accident is useful: it is direct evidence that repeated measurements on
this hardware *do* carry meaningful spread.

**Correction 2 — and the important one. The regressors would average a
distributional dataset away.** They are point predictors fitted on squared error,
so they return `E[y|x]`. On that same `'sd'` pair:

| | value |
|---|---|
| measured, row 0 | **33.7991 ms** |
| measured, row 1 | **23.5592 ms** |
| RandomForest predicts, for **both** | **20.9059 ms** |

**Collecting repeats is necessary but not sufficient.** Feeding a distribution to
a mean-predictor yields a better-estimated mean, not a distribution. D-40 has two
halves and both are needed.

Two traps were removed so the incoming data is not wasted:

* `TracePredictor._row` was an unconditional `.iloc[0]` — repeats would have been
  read and silently discarded, so the new dataset would have changed nothing and
  looked like it achieved nothing. Now `constants.TRACE_SAMPLING`
  (`sample`/`first`/`mean`). Same `.iloc[0]` fixed in `servers._csv_lookup`.
* `constants.COMPUTATION_SOURCE="empirical"` samples the **measured** values for a
  contention string, falling back to the model only for unmeasured strings — the
  inference path that does not average.

Both defaults preserve current behaviour, and `SAFETAIL_LEGACY_ENV=1` pins
`TRACE_SAMPLING="first"` so `results/reference_v0` still reproduces exactly.

> ⚠️ **One consequence to record.** Because `TRACE_SAMPLING` defaults to
> `"sample"` and server 5 *does* have six multi-row contention strings, this is
> **not** a no-op on today's data — server 5's computation times become
> stochastic. **Runs made after 8 Sep are not comparable to runs made before it
> unless `SAFETAIL_TRACE_SAMPLING=first` is set.** Flagged by the Claude Code
> verification session; recorded here so nobody discovers it from a graph.

New gate **G8** (`tools/verify_dataset.py`) audits a dataset before anyone trains
on it: repeats per contention string, whether the spread is real (CV — copied
numbers do not pass), coverage of all 34 contention strings, schema, and the
`Iteration` column. Full sequence in `NEW_DATASET_CHECKLIST.md`.

### 4b.11 What D-40 and D-41 mean together

Taken together they say something uncomfortable and worth saying plainly:
**this testbed cannot demonstrate what the project set out to demonstrate.**
There is no tail to optimise (D-40) and almost nothing for a scheduler to learn
that a one-line heuristic does not already capture (D-41). No amount of reward
engineering fixes either — both are properties of the traces and the hardware
selection, not of the code.

That is not a reason to abandon the work. It is the finding that tells you what
to do next: **collect repeated measurements, and add at least one server whose
network distance disagrees with its compute speed.** Until then, a negative
result honestly explained is worth more than a positive one that rests on a
5 ms offset and a discontinuous reward.

---

## 5. Gate status — all seven green

| Gate | Script | Checks | State |
|---|---|---|---|
| G0 | `verify_env.py` | 15 regressors unpickle; pins match; numpy 1.x | **PASS** |
| G1 | `verify_heterogeneity.py` | the 5 servers give materially different computation delays | **PASS** (D-15 alias reported as an acknowledged warning) |
| G2 | `audit_replay.py` | D-04/D-05/D-06 over ≥200 transitions | **PASS** (450 transitions, 0 leaks) |
| G3 | `audit_reward.py` | reward not monotone in \|A\| once `c_red>0`; finite for CPU-only servers; failure path writes the right slot | **PASS** |
| G4 | `verify_isolation.sh` | `rm -rf baselines/` leaves a working repo; no `src/`→`baselines/` reference | **PASS** |
| **G5** | **`check_manifest.py`** | manifest present, complete, clean tree, zero degradation, inputs unchanged | **PASS** *(new today)* |
| G6 | `verify_figures.py` | every figure has a companion CSV; no per-mode additive offsets | **PASS** *(clean today — the D-03 warning is gone)* |
| G7 | `verify_types.py` | `s→Speech, d→Detect, p→Predict` **from the dataset**, on all 5 servers | **PASS** |

G7 is worth a sentence of its own. **S-14** is the one defect where a
plausible-sounding sentence in the report would send someone to "fix" correct
code: BTP §3.3/§4.6 prose and HED item 9 misname the request types, while
BTP **Table 3.1 is correct**. The dataset settles it —
`dataset/server{1..5}.csv` column `Scripts Executed` literally contains
`Speech`/`Detect`/`Predict`, and `s` has the longest processing time and the
longest deadline on every server. G7 makes the dataset the arbiter, permanently.
*This reverses a decision recorded on 19 Aug 2026.*

---

## 6. What is still unimplemented

Nothing below is a surprise; each has a recorded disposition. "Not mentioned" is
not a disposition.

### 6.1 Blocked on a decision — wants advisor sign-off

| ID | What | Why it is blocked |
|---|---|---|
| **D-25 / M-15 / B1b** | The regressors are trained on `total_processing_time` — **a component of the very quantity they predict.** The fix is to drop it, adopt HED §IV-D's live-utilisation feature design, and retrain all 15 models, reporting held-out R² before and after. | The honest R² **will fall** — the current "prediction" is partly a ground-truth lookup. That fall is the finding, not a regression. But it changes every latency number in the project, so it should not run without agreement. **S-12: here HED is the better specification and BTP §4.6 misdescribes what the code actually does.** |
| **F8 as specified** | B3's acceptance figure is the K-vs-`c_red` sweep over `C_RED_SWEEP = [0, 0.02, 0.05, 0.1, 0.2, 0.4]`. | Needs 6 full runs (hours). The `F8_*` files currently in `figures/` are the 3× data-budget experiment — a different figure that took the name. Worth renaming to avoid confusion. |
| **D-26(b)** | Train actual MLP regressors and pick by held-out R². | Optional; bundle with B1b. |
| **D-37** | Clamp the ε floor. | Shifts ε by ~1.5 % in every existing run. Do it with the next full re-run. |
| **D-40** | **Give the simulator a tail.** Either collect repeated measurements per contention string (multiple `Iteration` values) and fit a real distribution, or restore SafeTail 1.0's measured per-concurrency `st_dev` noise as a stated modelling assumption. | Changes the simulator itself. The first route is correct but needs new data collection; the second is cheap but imports a `st_dev` table measured on 1.0's *homogeneous* hardware. **This is the decision that determines whether the project has a research question left.** |
| **D-41** | **Add a server whose network distance disagrees with its compute speed.** | A testbed property, not a code change. Until then MinProp is near-optimal by construction and no scheduler can show much gain. |
| **Re-run 2.0 after D-38** | The P(T) repair changes the episodic reward. | Baselines, Oracle and SafeTail 1.0 are unaffected (none uses that reward), so the comparison structure survives — but every SafeTail-2.0 row needs regenerating. |

### 6.2 Deferred by design — each recorded, none silently missing

| ID | What | Disposition |
|---|---|---|
| M-07 | DRL-Linear baseline | §9.5, out of scope for this phase |
| M-08 | TLORA baseline (Weibull fit) | §9.5; a reference implementation exists at `baselines/safetail_v1/_spec_source/v1_baseline_tlora.py` |
| M-09 | Sigmoid multi-label head + argmax fallback | **Record as a deliberate design choice** (31-way discrete over sigmoid multi-label), not an omission. HED's own output dimension is wrong (`nk` where its §II implies `mk`, S-09), so any implementation is a redesign, not a restoration. |
| M-10 | M/M/1 waiting time | ⚠️ **Do not implement HED as written.** Its formula is the correct M/M/1 queue-wait applied to a finite-capacity system that *rejects* rather than buffers. Resolved by restating the system as `M/M/c/c` Erlang-B. Reporting the rejection rate as a first-class metric is the cheap remaining win — `dropped_requests` is already counted but not plotted. |
| M-11 | Local fallback on the user device under saturation | Interim behaviour is D-21's bounded retry + drop counter. Promote if the drop rate turns out non-trivial. |
| M-12 | Job-length-proportional priority | **S-08: HED states this objective and never implements it in its own reward either.** It is an unimplemented aspiration in the source document, not something this codebase lost. Deferring is legitimate; claiming it is not. |
| M-13 | Joint decision over the k requests in a chunk | Code decides sequentially and independently; HED §III-A and BTP §3.6 agree with each other and both disagree with the code. Either implement or state the semantics plainly. |
| D-22 / D-24 | Poisson arrivals / real queues | Resolved by retraction. Building real queues is a separate project. |

### 6.3 Housekeeping

* **Branch divergence.** `master` and `safetail-v1-paper` differ by one file
  (`audit/BTP_REPORT_ERRORS.md`, present on `master` only). Merge or delete the
  branch before handing anything over.
* **`SafeTail-2.0-main/` on disk is no longer pristine** — it has picked up
  `regressors.py`, `policy_registry.py`, `rewards.py`, `_seeding.py` and a fixed
  `requirements.txt`. **The trustworthy pre-fix reference is commit `c93a859`
  in this repository**, not that folder. Every diff in this report was taken
  against the commit.
* **`results/reference_v0/` remains byte-identical** to the shipped
  `SafeTail-2.0-main/results/safetail_training_logs/` — re-verified today,
  md5 match on all 5 CSVs (`access_rate_log`, `episode_rewards`, `latency_log`,
  `request_wise_access_log`, `step_rewards`). It *is* the published
  heterogeneous run your mentor has plots for, and it is frozen read-only.

---

## 7. Where the report and the code disagree, and which one moved

BTP is the primary specification, so every disagreement needed a decision. The
full register is `audit/ERRATA.md` (S-01…S-20) and `audit/BTP_REPORT_ERRORS.md`.
The ones that changed an implementation decision:

| Item | Decision | Rationale |
|---|---|---|
| Request-type identities (**S-14**) | **Report changed, code kept.** Table 3.1 is right; §3.3/§4.6 prose and HED item 9 are wrong. | The dataset is decisive and G7 now enforces it |
| Degree of satisfaction `P(T)` (**S-01**) | **Neither changed — record as a strength.** BTP's piecewise-linear `P(T)` is a correct repair of HED's malformed ω (defined twice, incompatibly, with a *range* as one branch's value). | Do **not** restore HED's version. Citing this makes the rest of the errata read as analysis rather than attack. |
| Step reward (**S-02**) | **Code extended.** `c_red` added on top of BTP's corrected headroom product. | HED's version was sign-inverted; BTP fixed the sign and dropped the redundancy pricing. Neither document is authoritative, so the design decision is ours. |
| ε decay (**S-16**) | **Report is wrong; code kept.** | The subtractive schedule produced every existing result |
| Reward range (**S-03**) | **Report is wrong.** `log(1+1) = log 2 ≈ 0.693`, not 1. | Three inconsistent statements existed; now one |
| Latency equation (**S-04**) | **Code chose; report needs rewriting.** BTP §3.4's display equation omits `T_exec`, its own bullet list includes it, Fig. 3.2 includes it twice, and the code implemented a fourth version. | Now logs **both** `total_latency` (service) and `end_to_end_latency` (+ queue). Never plot a quantity the reward ignores. |
| State description (**S-10**) | **Report is wrong; §3.2 is right and §4.2 is not.** | §4.2's "temporal accumulation" describes something the code never does. Use §3.2's justification — and note the report *understates* its own system (D-35). |
| Baseline description (**S-11**) | **Report is wrong.** | §5.2 describes a budget-neutral pre-filter; the code is a fixed-K redundant dispatcher |
| MLP claim (**S-13**) | **Report is wrong.** 6 Linear / 8 RF-DT / 1 GBM. | HED may be truthful about *its* models; BTP is not truthful about *this repo's* |
| Headline result (**S-17 / D-10**) | **Report is false on its own data.** | The single most important thing to resolve before submission |
| SafeTail 1.0 code vs paper (**S-18/19/20**) | **The published 1.0 is not architecturally broken; its GitHub implementation is.** | Bellman targets fed to softmax+CCE (S-18), a reward differing from Eq. 5 in all three cases (S-19), 2 sigmoid layers instead of 5 ReLU (S-20). Both variants are now implemented and reported. |

---

## 8. Verification performed

| Check | Result |
|---|---|
| `python tools/verify_types.py` (G7) | **PASS** |
| `python tools/verify_heterogeneity.py` (G1) | **PASS** — per-server delays 12.45 / 12.45 / 388.38 / 266.42 / 13.18 ms |
| `python tools/verify_figures.py` (G6) | **PASS**, and now warning-free |
| `python tools/check_manifest.py` (G5) | **PASS** — 0 failing, 20 unprovenanced (pre-gate) |
| `pytest tools/tests/test_followup_fixes.py` | **21 / 21 pass** |
| `pytest` (full suite, in the Cowork sandbox) | **35 pass, 19 blocked** by the absent TensorFlow (the 4 named files hold 19 tests, not 10 — arithmetic corrected 8 Sep) |
| **`pytest` on the real machine** (Claude Code session, 8 Sep) | **45 passed, 0 failed** — ⚠️ gap in §8 CLOSED |
| **All 7 gates on the real machine** | **PASS**, including a clean `--smoke` run (20 episodes, 300 requests, `dropped=0`, manifest written) |
| **Dossier claim — P(T) sign (D-38)** | **reproduced**: `controller.py:486` had no `1 -`; repaired form is continuous at both boundaries and monotone; T=40 ms now scores 0.94, was 0.06 |
| **Dossier claim — unscaled features (D-39)** | **already fixed here**: `regressors.py:185-187` applies the scaler for every server/task |
| **Dossier claim — no computation variance (D-40)** | **reproduced**: 8 calls → `2.4931` ms every time, stddev 0.0; `server{1,2,3,4}.csv` = 363 rows / 363 unique combos / `Iteration==[1]` (`server5.csv` = **369/363**, six real repeats — §4b.12); SafeTail 1.0's `np.random.normal(0, st_dev[...])` present at `_spec_source/v1_agent.py:73` and absent in 2.0 |
| **Dossier claim — ping ≈ compute (D-41)** | **reproduced**: pings `[5.1, 12.6, 51.3, 67.5, 71.2]` ms vs compute `[2.39, 2.39, 46.23, 57.89, 2.72]` ms; Spearman ρ = 0.95 (servers 1–4), 0.67 (all five) |
| p99/p50 on the shipped run | **2.89×** (real tail-latency problems: 5–50×) |
| Manifest writer, end to end | writes 15 top-level keys, 20 input hashes, 59 constants; correctly reports `NOT PUBLISHABLE` on a dirty tree |
| Notebook integrity after the D-03 edit | valid JSON, 12 cells, zero `0.005` occurrences |

> ### ✅ The gap below is CLOSED (8 September 2026)
>
> A Claude Code session ran the suite on the real machine: **45 passed, 0 failed**,
> all seven gates PASS, and a clean `--smoke` run (20 episodes, 300 requests,
> `dropped=0`, manifest written). **There was no TensorFlow problem** — it was
> purely the Cowork sandbox lacking the package. It also confirmed the two facts
> only a real run can establish: the D-32 access-rate series is genuinely
> per-episode (non-monotone, matches the per-request mean, ≠ the cumulative mean)
> and D-38's `P(T)` is not inverted at runtime (289 distinct step rewards,
> bounded and varying). Its report: `HANDOFF_FROM_CLAUDE_CODE.md`.
>
> Three corrections it made are applied: the `_manifest.py` `dirty_files`
> off-by-one (every manifest recorded `HANGELOG.md`), the `check_manifest.py`
> inability to see out-of-tree runs (now `--dir`), and the arithmetic below.
>
> The suite has since grown to **48** tests with the D-40b work, all passing.
> A second verification pass then found three more issues, all now fixed:
> `check_manifest.py --dir` under-reported its tally (`0 failing` while exiting
> 1), G8 printed `[FAIL]` while exiting 0 (now `[INADEQUATE]` in advisory mode,
> `[FAIL]` only under `--strict`), and G8 was missing from `plan.md` §11.

**⚠️ The gap as it stood.** The 19 blocked tests (`test_b3_redundancy`, `test_b4_tau`,
`test_b8_structural`, `test_c_v1_baseline`) all fail on the same line —
`ModuleNotFoundError: No module named 'tensorflow'` — because the verification
environment has no TensorFlow. **None of them fails on anything I changed**, but
that is an inference from the error, not an observation of them passing. Before
trusting this session's work, run in your own `.venv` (Python 3.12 / TF 2.20):

```bash
cd E:\Project\IP_Arani\heterogenous
.venv\Scripts\python -m pytest -q                  # expect 36 passed
.venv\Scripts\python tools\verify_env.py           # G0
.venv\Scripts\python -m pytest tools\tests\test_followup_fixes.py -q
set SAFETAIL_SEED=0 && .venv\Scripts\python src\main.py --smoke
python tools\check_manifest.py --require <the smoke run dir>
```

The last two matter most: the smoke run is the only thing that exercises the new
manifest writer and the D-32 per-episode reset **at runtime** rather than
through source-level assertions.

---

## 8b. THE OPEN-ITEMS BOARD — everything not done, sorted by who decides

*The single place to look before committing, running, or writing anything up.
Updated 8 September 2026, after two independent verification passes.*

### 8b.1 🔴 Advisor decisions — do not proceed without these

Nothing below is a bug. Each is a judgement call that changes what the project
*claims*, and none should be made by whoever happens to be at the keyboard.

| # | Question for the advisor | Why it is his call, not ours | Blocks |
|---|---|---|---|
| **A1** | **The headline claim is false on the report's own data (S-17 / D-10).** SafeTail 2.0 loses to MinProp at every percentile (p99 115.6 vs 53.4 ms), and paper-faithful SafeTail 1.0 beats 2.0 at p95 (68.66 vs 75.66). **Retract, reframe, or re-establish on post-fix data at matched K?** | It determines whether the BTP result stands, is amended, or is withdrawn. That is a supervision decision with authorship implications. | everything downstream of the write-up |
| **A2** | **The testbed cannot demonstrate its own thesis (D-40 + D-41).** No tail to optimise (p99/p50 = 2.89× vs 5–50×), and MinProp is near-optimal by construction (ρ = 0.95). **Is the deliverable a corrected positive result, or an honest negative one?** | A negative result is publishable and defensible here — but only if he agrees that is the project. Otherwise it needs new data *and* new hardware assignment. | the entire experimental plan |
| **A3** | **Restore variance how?** (a) collect repeated measurements and fit a real distribution — correct, needs the campaign; (b) reuse SafeTail 1.0's measured `st_dev` table — cheap, but measured on 1.0's *homogeneous* hardware and must be disclosed as an approximation. | (b) is a modelling assumption that goes in the paper. Not ours to assume silently. | D-40, B1b, every latency number |
| **A4** | **β = 5 or β = 4?** This branch aliases server 2 → server 1; the external dossier excludes it. The real Ryzen 5 7600X trace is **not in the repository**. **β = 4 is arguably the honest configuration.** | Changes the action space (31 → 15 subsets) and every published number. The two workstreams must agree before merging results. | any merged comparison |
| **A5** | **Decorrelate ping from compute?** The ping-trace → machine assignment is arbitrary and lives outside the CSVs — a one-line change. Alternative: keep it and *report* that MinProp is near-optimal in this topology. | Changing it makes the problem learnable but is a testbed intervention that must be declared. Keeping it makes the negative result the finding. | D-41 |
| **A6** | **B1b — retrain the regressors?** Dropping the `total_processing_time` target leak (D-25) will make the honest held-out R² **fall**. | That fall is the finding, not a regression — but he should hear it from us first, not see it in a table. | D-25, M-15, D-26(b) |

### 8b.2 🟠 Krishna's call — no advisor needed, but a decision is needed

| # | Item | Options |
|---|---|---|
| **K1** | **Commit the working tree.** Every manifest is stamped `NOT PUBLISHABLE` for the correct reason (`git.dirty: true`), so **no run made before a commit can ever be provenanced**. `reference_v0` reproduction currently depends on `LEGACY_ENV` pinning `TRACE_SAMPLING="first"` — a guarantee only as durable as an uncommitted file. | Two commits: (1) the 11 fixes, (2) D-40b + G8. `baselines/` and `results/` out of both. |
| **K2** | **Re-run the SafeTail 2.0 side.** Required by **D-38** (episodic reward changed) and **D-44** (state vector +2 elements). Baselines / Oracle / 1.0 are unaffected — none uses that reward or that state builder — so the comparison structure survives. | Do it after K1 so the runs are provenanced. |
| **K3** | **`F8_*` is misnamed.** The files in `figures/` are the 3× data-budget experiment; B3's acceptance figure is the K-vs-`c_red` sweep. | Rename to `F9_budget_3x` and produce the real F8 (6 runs). |
| **K4** | **Merge the two defect registers.** Ours and the dossier's use colliding `D-nn` numbering. Worth an afternoon, with attribution. | — |
| **K5** | **Re-package `dist/`.** Anything zipped before the 3 Sep retraction still carries the withdrawn "2.0 beats 1.0" headline. | Re-package or delete. |
| **K6** | **Merge or delete branch `safetail-v1-paper`** (differs by one file). | — |

### 8b.3 🟡 Known-open technical items — no decision, just work

| ID | Item | State |
|---|---|---|
| **D-40** | no computation variance | plumbing ready (`TRACE_SAMPLING`, `COMPUTATION_SOURCE=empirical`, gate G8); **blocked on the new dataset** — see `NEW_DATASET_CHECKLIST.md` |
| **D-41** | ping ≈ compute (ρ = 0.95) | blocked on **A5** |
| **D-25 / M-15 / D-26(b)** | target leak, feature design, MLPs | blocked on **A6** |
| **D-37** | ε floor is a guard, not a clamp — the last step lands up to `epsilon_decay_step` below `epsilon_min` | deliberately unfixed; shifts ε ~1.5 % in every existing run. Bundle with the K2 re-run. |
| **D-45** | `latency_log.csv` mixes seconds and ms with no unit in the header | documented + de-magicked; **columns deliberately not renamed** (would make every run under `results/` unreadable). Unify at the next format break. |
| **M-07/08/09/11/12/13** | DRL-Linear, TLORA, sigmoid head, local fallback, job-length priority, joint chunk decision | deferred by §9.5 / §B11, each with a recorded disposition |
| **M-10** | rejection rate | `dropped_requests` is counted but never plotted — the cheapest remaining win, and the honest replacement for the deleted M/M/1 claim |
| — | **G8 not in CI** | advisory by design today (exit 0 + `[INADEQUATE]`); switch to `--strict` once the new data lands |

### 8b.4 ⚪ Doubts I could not resolve from the repository

Stated as open questions rather than guesses:

1. **What is request type `p` ("Predict")?** Defined **nowhere** in ST, HED or BTP. The dataset says it is real and it has its own deadline pair; no document says what workload it is.
2. **Do HED's MLP regressors exist?** HED §IV-D cites `github.com/amardeep786/Regressors`. If they exist they are the cheapest correct input to B1b. Not reachable from here — worth one look.
3. **Are the raw per-file timing logs recoverable?** The dossier suggests asking Jyoti Shokhanda. If the ~500 per-scenario files survive, D-40 is fixable *from existing measurements* with no new campaign.
4. **Was the 5 ms MinProp offset (D-03) deliberate?** No justification anywhere in the repository. Worth knowing before the write-up characterises it.
5. **Which workstream's environment is the reference going forward?** Ours (legacy-faithful) and the dossier's (repaired, β=4) give opposite redundancy results — both correct, different environments. Merging results requires picking one.

### 8b.5 🟢 Green — verified, no action

All eight gates pass (G8 advisory by design). **48 tests**, all passing on the
real machine. Smoke run clean, manifest written. D-32 and D-38 confirmed at
runtime, not just by source inspection. Two independent sessions have now
verified this tree.

---

## 9. What to do next, in order

1. **Get the missing 12 pages of the dossier.** The copy received is truncated at
   page 8 of 20, and page 8 ends at the *start* of a section headed "STILL OPEN".
   Four new S1 findings came out of the eight pages that survived; there is no
   reason to assume the remaining twelve are empty.
2. **Take D-40 and D-41 to your advisor as one question.** Together they say the
   testbed has no tail to optimise and little for a scheduler to learn. That is
   the decision that determines whether there is a research question left, and it
   is upstream of every other item on this list. The constructive framing: *the
   traces need repeated measurements, and the server set needs at least one
   machine whose distance disagrees with its speed.*
3. **Resolve S-17 in the same conversation.** The report's headline claim is false
   on its own data, and the paper-faithful SafeTail 1.0 now beats 2.0 as well.
   Lead with **S-01** (BTP's `P(T)` is a genuine, correct repair of a real HED
   defect) so the critique lands as analysis rather than attack — and note that
   the code did not actually implement it until today (**D-38**).
4. **Re-run the SafeTail 2.0 side** after the D-38 repair. Baselines, Oracle and
   1.0 are unaffected, so only the 2.0 rows need regenerating.
5. **Run the smoke test in §8**, to convert the ⚠️ there into an observation.
6. **Decide on B1b** (D-25 target leak + M-15 features). Until it runs, every
   computation-latency number rests on a regressor partly reading its own answer
   key. Bundle this with the D-40 decision — both are about the regressors.
7. **Rename the current `F8_*` files** to `F9_budget_3x` and produce the real
   F8 (`c_red` sweep), so B3's acceptance criterion is met by the figure it names.
8. **Plot the rejection rate.** `dropped_requests` is already counted and is
   currently invisible; it is the honest replacement for the deleted M/M/1
   claim (M-10) and costs almost nothing.
9. **Merge or delete `safetail-v1-paper`.**

**On the two workstreams.** Uttam's dossier and this branch converge on the same
defects from independent readings — that is the strongest evidence either of you
has that the findings are real, and it is worth saying so explicitly to your
advisor. The two are not redundant: the dossier caught D-38/D-39/D-40/D-41, which
this register missed; this branch has the gates, the manifests, the run matrix and
the specification lineage (S-01…S-20), which the dossier does not. **Merging the
two defect registers into one document, with attribution, is worth an afternoon.**

---

## 10. Sources

| Document | What it is |
|---|---|
| `plan.md` | master plan; §4 defect register, §4.5 specification lineage, §7 workstreams, §11 gates |
| `CHANGELOG.md` | per-change ledger; gate status |
| `KNOWME.md` | which SafeTail is which — variant map, access-rate analysis, what may be claimed |
| `audit/ERRATA.md` | S-01…S-20 with provenance codes (`HED-BUG` / `BTP-FIX` / `BTP-NEW` / `BTP-KEPT` / `BTP-REGRESS`) |
| `audit/BTP_REPORT_ERRORS.md` | consolidated report errors, severity-ranked for an advisor conversation |
| `results/BASELINE_COMPARISON_REPORT.md` | the comparison and the S-18 verdict |
| `results/RUN_PROVENANCE.md` | what was run, under what conditions, and what it may claim |
| `baselines/safetail_v1/README.md` | faithfulness register for the 1.0 port |
| git `c93a859` | the verbatim pre-fix `SafeTail-2.0-main` — the reference for every diff here |
| *SafeTail Defect Dossier* (Uttam, 3 Sep 2026) | the external audit — source of D-38…D-41 (§4b). **⚠️ the copy received is truncated at page 8 of 20** |

---

*Every claim in this report was verified against the code in this repository on
7 September 2026, with one exception, flagged in §8.*
