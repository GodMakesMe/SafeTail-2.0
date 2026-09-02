# KNOWME — which SafeTail is which

A map of every SafeTail variant in this project: what it is, where the code
lives, where its results live, and what it may be used to claim. Read this
first if you are returning to the project or handing it to someone else.

Last updated: 2 Sep 2026.

---

## TL;DR

| # | Name | Role | Code | Results |
|---|---|---|---|---|
| 1 | **SafeTail 1.0** | **THE BASELINE** | `baselines/safetail_v1/` (a *port*) | `results/safetail_v1_legacy_s{0,1,2}/` |
| 2 | **SafeTail 2.0 — original** | **heterogeneous, NOT updated** | `../SafeTail-2.0-main/` (upstream) | `results/reference_v0/` |
| 3 | **SafeTail 2.0 — updated** | **heterogeneous, UPDATED (repaired)** | `src/` (this repo) | `results/native_legacy_s{0,1,2}/` |
| 4 | Oracle | reference upper bound | `baselines/oracle/` | `results/oracle_legacy_s{0,1,2}/` |
| 5 | MinLoad / MinProp / Rand | fixed-K heuristics | `src/controller.py` | `results/reference_v0/baselines/` |

**The one-line answer:** SafeTail **1.0 is the baseline**; SafeTail **2.0 is the
heterogeneous** system; and there are **two** SafeTail 2.0s — the *original*
(unfixed, the one your mentor's plots come from) and the *updated* one in this
repo's `src/`.

---

## 1. SafeTail 1.0 — the BASELINE

The original published SafeTail.

* **Upstream source (read-only, never edited):** `E:\Project\IP_Arani\SafeTail\`
  → `github.com/Jyotishokhanda/SafeTail`. β=4, softmax+CCE, τ-referenced 5-case
  reward, flat homogeneous state.
* **What we actually run:** `baselines/safetail_v1/` — a **PORT, not a bridge**.
  It reimplements the 1.0 *algorithm* against the SafeTail **2.0** environment,
  so both sides are graded by the same servers, traces and admission rules.
  It never imports the old repo, never loads 1.0's pickles.
* **Frozen reference copies to read while porting:**
  `baselines/safetail_v1/_spec_source/` (11 files, never imported).
* **Every deviation from 1.0 is recorded** in
  `baselines/safetail_v1/README.md` (the faithfulness register) — notably
  `V1-DEV-03` (ε schedule rescaled to the run budget) and `V1-BUG-01`
  (1.0 returned `None` into its replay buffer).
* **Results:** `results/safetail_v1_legacy_s{0,1,2}/` (3 seeds) plus
  `results/v1_legacy_s0/` (an independent Keras-`fit()` run kept as the fidelity
  reference for the optimised training path).

> Deleting `baselines/` must leave a fully working repo. Enforced by gate **G4**.

---

## 2. SafeTail 2.0 — the HETEROGENEOUS system

There are **two** of these. Do not mix them up; they produce different numbers.

### 2a. ORIGINAL / NOT UPDATED  ← your mentor's results

* **Code:** `E:\Project\IP_Arani\SafeTail-2.0-main\SafeTail-2.0-main\`
  = `github.com/shrutya22487/SafeTail-2.0` branch `main` (tip `53eddaa`).
  Also preserved as commit `c93a859` in this repo ("verbatim copy" seed commit).
* **Results:** `results/reference_v0/` — **frozen, read-only**. The
  `safetail_training_logs/` inside it is **byte-identical** (md5, all 5 CSVs) to
  `SafeTail-2.0-main/results/safetail_training_logs/`, i.e. it *is* the published
  heterogeneous run your mentor has plots for.
* **Contains every defect in `plan.md` §4** — including **D-02**, which means the
  computation path was **not actually heterogeneous** (all 15 regressor wrappers
  loaded `models/server1/` + `dataset/server1.csv`).
* **Use it for:** the comparison against the baseline, because it is what was
  actually published.

### 2b. UPDATED / REPAIRED  ← this repo's `src/`

* **Code:** `src/` in this repository. Carries fixes for
  D-01, D-02/02b/02c, D-04, D-05, D-06, D-07, D-11, D-12, D-13, D-14, D-16,
  D-17, D-18, D-19, D-20, D-21, D-23, D-25(partial), D-27…D-36, plus M-01, M-02,
  M-03. Full ledger in `CHANGELOG.md`; specification errata in
  `audit/ERRATA.md`.
* **Results:** `results/native_legacy_s{0,1,2}/` — the updated code run under
  `SAFETAIL_LEGACY_ENV=1`.
* **Pushed as:** branch `heterogenous` on `github.com/GodMakesMe/SafeTail-2.0`
  (a fork of shrutya22487's repo), one commit on top of upstream `main`.

### The bridge between them: `SAFETAIL_LEGACY_ENV=1`

The updated code can **reproduce the original's physics on demand**. Setting
`SAFETAIL_LEGACY_ENV=1` restores exactly the four repairs that changed the
**latency model**:

| restored | effect |
|---|---|
| D-02 | every server predicts computation with `models/server1` + `server1.csv` |
| D-12 / D-34 | transmission = `random.choice([18.5,19.2,20,21.5,22])/1000` |
| D-18 | phase 2 and phase 7 each draw their own propagation/transmission |
| D-23 | queue wait = wall-clock elapsed |

It deliberately does **not** revert the reward-side repairs (D-04…D-07, D-16,
D-20, B3, B4) — those change only the 2.0 *learned policy*, never the latency a
request experiences, so they cannot affect a baseline comparison.

**Validated, not assumed:** legacy-mode transmission reproduces the original
exactly (mean 20.179 vs 20.197 ms, 5 distinct values both); computation 8.863 vs
8.638 ms (+2.6%, contention-profile noise).

**Every comparison run in this project used `SAFETAIL_LEGACY_ENV=1`**, so the
baseline is measured under the same physics that produced the published
heterogeneous numbers.

---

## 3. Access rate — the numbers you asked for

Access rate = |A| / β, the fraction of the 5 servers a request is dispatched to.
(BTP Fig. 5.1 calls this a "cache access rate"; **there is no cache** — see
`audit/ERRATA.md` S-15.) Mean K = access_rate × 5.

| policy | access rate | mean K | K=1 | K=2 | K=3 | K=4 | K=5 | 1st half → 2nd half |
|---|---|---|---|---|---|---|---|---|
| **SafeTail 2.0 — original** | **0.711** | **3.55** | .052 | .100 | .099 | **.739** | .009 | 3.25 → **3.86** |
| **SafeTail 1.0 — baseline** (s0) | 0.523 | 2.61 | .125 | .370 | .285 | .206 | .014 | 2.55 → 2.68 |
| SafeTail 1.0 (s1) | 0.508 | 2.54 | .098 | .435 | .323 | .116 | .028 | 2.57 → 2.51 |
| SafeTail 1.0 (s2) | 0.503 | 2.51 | .101 | .453 | .303 | .119 | .024 | 2.60 → 2.42 |
| SafeTail 2.0 — updated (s0) | 0.302 | 1.51 | **.731** | .101 | .109 | .048 | .011 | 1.85 → 1.16 |
| SafeTail 2.0 — updated (s1) | 0.574 | 2.87 | .052 | .099 | **.788** | .052 | .010 | 2.78 → 2.96 |
| SafeTail 2.0 — updated (s2) | 0.606 | 3.03 | .054 | .102 | .615 | .219 | .010 | 3.09 → 2.97 |
| Oracle | 1.000 | 5.00 | — | — | — | — | 1.000 | 5.00 → 5.00 |
| MinLoad-K / MinProp-K / Rand-K | K/5 | K | fixed by construction | | | | | |

### What this says

1. **The original 2.0 over-replicates.** 73.9% of its requests go to exactly
   **4** servers, and mean K **climbs monotonically** 3.25 → 3.86 across the run.
   This is defect **D-08**, and its cause is **D-07**: no term in the 2.0 reward
   decreases in |A|, so more servers is always weakly better. BTP §6.2 reads this
   climb as "deadline adaptation"; it is a climb up the reward gradient.

2. **SafeTail 1.0 is disciplined and flat** at K≈2.5, with no drift across the
   run. Its τ-referenced reward explicitly prices redundancy — finish *early* and
   the penalty scales with how **many** servers you used. This is the redundancy
   pricing 2.0 lacks (M-04).

3. **The comparison is therefore NOT compute-matched.** 2.0's tail-latency win
   over 1.0 (p95 75.66 vs 89.72) comes with **39% more compute** (K 3.55 vs 2.55).
   This is defect **D-09**. Use `SAFETAIL_MATCH_K=2.55` for a fair re-run.

4. **Fixing the reward fixed the over-replication.** The updated 2.0 drops to
   mean K 1.51 / 2.87 / 3.03 (median ~2.87) from the original's 3.55 — with
   `c_red = 0`, i.e. with **no explicit redundancy penalty at all**. Collapsing
   the step reward over |A| instead of over a constant 6 (D-07) was sufficient.
   The seed-0 collapse to K=1.51 also shows the updated policy is seed-unstable.

---

## 4. Which numbers may be used for what

| claim | supported? |
|---|---|
| SafeTail 2.0 beats SafeTail 1.0 on tail latency | **yes** — p95 75.66 vs 89.72 (−15.7%), every percentile, identical data |
| …because it schedules better | **no** — confounded by 39% more compute (D-09) |
| SafeTail 2.0 beats MinProp | **no** — MinProp-2 p99 53.37 vs 115.64. D-10 / S-17 |
| SafeTail 1.0 is under-trained | **no** — it gets 30× more gradient updates than 2.0 and *degrades* after decile 7 (figure F5) |
| These results say anything about heterogeneity | **no** — D-02 made computation identical across all five servers in the code that produced them |

Full argument and evidence: `results/BASELINE_COMPARISON_REPORT.md`.

---

## 5. File map

```
E:\Project\IP_Arani\
├── SafeTail\                       SafeTail 1.0 UPSTREAM (read-only, never edited)
├── SafeTail-2.0-main\              SafeTail 2.0 ORIGINAL (not updated) + mentor's results
│   └── ...\results\safetail_training_logs\   == results/reference_v0/safetail_training_logs
└── heterogenous\                   ← THIS REPO (the working project)
    ├── src/                        SafeTail 2.0 UPDATED
    ├── baselines/safetail_v1/      SafeTail 1.0 PORT (the baseline)
    ├── baselines/oracle/           Oracle (M-01)
    ├── results/reference_v0/       frozen ORIGINAL 2.0 results (read-only)
    ├── results/safetail_v1_legacy_s*/   baseline runs
    ├── results/native_legacy_s*/   updated-2.0 runs
    ├── results/oracle_legacy_s*/   oracle runs
    ├── figures/                    F1 F2 F5 F6 F7 (+ companion CSVs)
    ├── dist/safetail_v1_baseline.zip    the shareable deliverable
    ├── plan.md                     master plan, defect register (§4), errata (§4.5)
    ├── audit/ERRATA.md             HED vs BTP specification errata (S-01…S-17)
    ├── CHANGELOG.md                per-defect ledger of everything changed
    └── KNOWME.md                   this file
```

## 6. Remotes

| repo | what |
|---|---|
| `github.com/shrutya22487/SafeTail-2.0` | upstream SafeTail 2.0 (original). Read-only to us. |
| `github.com/GodMakesMe/SafeTail-2.0` | our fork. Branch **`heterogenous`** = the `src/` repairs; branch **`heterogenous-workspace`** = this full working repo. |
| `github.com/Jyotishokhanda/SafeTail` | SafeTail 1.0 upstream. Read-only. |
