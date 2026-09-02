# What is wrong in the BTP report — consolidated

Everything found against `documentation.pdf` (B.Tech report, Shamik Sinha,
Shrutya Chawla, Medha Kashyap, Shivankar Srijan Singh; 22 April 2026; advisor
Arani Bhattacharya), gathered from three sources:

| source | what it contributes |
|---|---|
| `SafeTail_2.0_Codebase_Analysis_v2.pdf` §7–§9 | code-vs-report traceability, gap analysis, bug list |
| `SafeTail_2.0_Addendum_A.pdf` §11–§13 | runtime-trace + training-log findings, one correction to the base report |
| `ERRATA.md` S-01…S-20 | specification defects, with provenance (which layer introduced each) |

Krishna is **not** an author of the BTP report; he inherited the codebase. The
framing for any write-up is *"inherited codebase, corrected and extended"*, not
self-criticism.

---

## A. Factually false claims (highest severity — the report asserts things the code contradicts)

| # | BTP says | Reality | Ref |
|---|---|---|---|
| **A1** | **MLP regressors** predict computation latency (§2.6, §4.6, **Abstract**) | The 15 shipped `.pkl` are **6 LinearRegression, 8 RandomForest/DecisionTree, 1 GradientBoosting**. No `MLPRegressor` anywhere. The wrapper even defaults `model_name = "Linear Regression"`. | v2 §8.3, §9#1 · S-13 · D-26 |
| **A2** | **M/M/1 queueing** at both controller and each server (§3.5, promoted to the **abstract**), `W = λ/(μ(μ−λ))` | **No queue exists.** No `λ`, no `μ`, no buffer. Servers are an `M/M/c/c` **Erlang-B loss system** (c=4) that *rejects* when full. | v2 §8.3 · Add. §12.5 · S-05 · D-13/D-24 |
| **A3** | `W_j = min` over selected servers (§3.5) | Absent — there is no per-server `W` to minimise over. | v2 §8.3 · S-05 |
| **A4** | Arrivals are **Poisson** (§3.5, inherited from HED) | Uniform: `random.randint(2,4)` chunks/burst, `random.uniform(0.2,0.8)s` between. "Randomness ⇒ Poisson" is a non-sequitur. | Add. §12.6 · S-06 · D-22 |
| **A5** | The run processes **500,000 requests** (Table 4.2) | **~15,225.** `no_of_burst=1000` binds before `total_no_request`, which only sizes a pre-generated pool. Overstated **~33×**. | Add. §12.9 · D-33 |
| **A6** | §6.2 headline: *"SafeTail has lower latency than MinLoad, MinProp, and Random at all percentiles, with larger gaps at the 95th and 99th"* | **False on the report's own shipped data.** SafeTail loses to MinProp at *every* percentile (p99 **115.6 vs 53.4 ms**) and to MinLoad-3 and Rand-3 at p95/p99. | S-17 · D-10 |

---

## B. The report contradicts itself

| # | Where | The contradiction |
|---|---|---|
| **B1** | §3.4 vs its own bullet list vs Fig. 3.2 | The **latency equation** appears three incompatible ways. The display equation `L_j = T_transmission + T_ctrl-queue + T_propagation` **omits `T_exec`**, while the bullets beneath define it and Fig. 3.2 includes it (twice). The code implements a **fourth** version. (S-04, D-11) |
| **B2** | §4.2 vs §3.2 | The **state**. §4.2: a `(β+1)`-dim load vector "accumulated across an episode … allowing the encoder to capture temporal patterns". §3.2: length varies because "servers have different hardware configurations". **§3.2 is right**; §4.2 is wrong on both counts and describes temporal accumulation the code never does. (S-10, D-35, Add. §11.1) |
| **B3** | §3.3 / §4.6 prose vs Table 3.1 | **Request-type names.** Prose says "Instance segmentation of images (s)" and "Removal of noise from audio (p)". Table 3.1 says Speech (s), 100/400 ms. **The dataset settles it: Table 3.1 is correct**, the prose is wrong. (S-14) ⚠️ |
| **B4** | §4.4 vs the code | **ε decay.** §4.4 states *multiplicative* `ε ← max(ε_min, ε(1−γ_ε))`; the code does *subtractive* `ε -= gamma_decay`. The constant's name matches neither. (S-16) |
| **B5** | §4.5.1 vs arithmetic | Claims the step reward "equals 1 when the server is completely free". `log(1+1) = log 2 ≈ **0.693**`. Three inconsistent statements exist (report, docstring, code). (S-03, D-27) |

---

## C. The report misdescribes its own system

| # | BTP says | Reality | Ref |
|---|---|---|---|
| **C1** | §4.6: regressor features are "input size, model complexity … CPU utilisation, GPU utilisation, number of active jobs" | Actual features: `num_speech/detect/predict, total_ops, position, is_first/is_last, peak_ram, peak_gpu, peak_gpu_memory, total_processing_time`. **No input size, no model complexity, no live utilisation** — just a static trace row keyed by the contention string. And it **leaks the target**. Ironically **HED §IV-D describes this correctly** and BTP does not. | S-12 · D-25 |
| **C2** | §5.2: the top-K servers are "passed through the same downstream decision pipeline as SafeTail" | **No such pipeline exists.** `_select_minload_servers(x)` returns x servers and **all x are scheduled redundantly**. The report describes a budget-neutral pre-filter; the code is a fixed-K redundant dispatcher. This is the report-side face of the budget confound. | S-11 · D-09 |
| **C3** | Fig. 5.1 calls it a **"cache access rate"** | **There is no cache** anywhere in the system. It is `\|A\|/β`, the replication fraction. | S-15 |
| **C4** | §3.6: a "step" = one chunk of k requests; episode = fixed number of steps | `current_step` increments **once per request**, not per chunk. | v2 §8.3 |
| **C5** | §3.7: `ω = (1/N)ΣP(T_j)` | Implemented with `N = total_no_request/no_of_episodes` rather than the actual request count in the episode — a constant mis-scaling. | v2 §8.3 |
| **C6** | §2.6: workloads are Mask R-CNN and Whisper | Unverifiable — only the letters `s/d/p` and trace CSVs exist. Also, **`p` ("Predict") is defined nowhere** in any document. | v2 §8.3 · S-14 |

---

## D. Claims that rest on defective evidence

Three §6.2 conclusions all derive from the access-rate figure, and **all three fail**:

| # | Claim | Why it fails |
|---|---|---|
| **D1** | The rising access rate shows "deadline adaptation" | It is a **monotone climb up the reward gradient**. Nothing in the reward decreases in \|A\| (D-07), so more servers is always weakly better. 73.9% of requests go to exactly 4 servers; mean K climbs 3.25 → 3.86. (Add. §12.1, §12.2 · D-08) |
| **D2** | The access-rate curve shows adaptation over time | It is a **cumulative (expanding) mean** — its final value is bit-identical to the overall request-wise mean. A cumulative average **cannot be non-monotonic**, so it structurally cannot show what is claimed. (Add. §12.3 · D-32) |
| **D3** | The baseline comparison shows SafeTail winning | **Not budget-controlled.** Baselines are hard-capped at K∈{1,2,3}; SafeTail runs free over all 31 subsets at mean K≈3.55, exceeding the K=3 ceiling in 83.3% of episodes. (Add. §12.4 · D-09) |

Plus, at plot time: `plot_baselines.ipynb` **adds a hardcoded +5 ms to the MinProp
family only**, in two cells, before percentiles are taken — an 11% inflation of
one baseline family (**D-03**).

---

## E. One thing BTP got right — say so

**§3.7's `P(T)`** is a genuine, correct repair of a real HED defect. HED §III-E-2a
defines the degree of satisfaction ω **twice and incompatibly** — once as a
ratio, then as a malformed piecewise whose middle branch is a *range* rather
than a value. BTP replaces it with a well-formed piecewise-linear `P(T)`, and
the code implements it correctly. **Do not "restore" the HED version.** (S-01)

Citing this makes the rest of the errata read as analysis rather than attack.

---

## F. Not BTP's fault — inherited from HED

| | |
|---|---|
| HED's step reward is **sign-inverted** for its own stated goal (maximising `Σ(l_i − l̄)` selects the *most*-loaded servers), and HED admits the balancing term is wrong in the next paragraph. BTP correctly replaced it — **but dropped `d` and `−W_step`, the only \|A\|-sensitive terms**, which is the true origin of the over-replication bug. | S-02 |
| HED's abstract is **truncated mid-word** — item 10 reads in full: *"The reward consists of the su"*. It is an unfinished draft. | S-09 |
| HED states a job-length-priority objective it never implements in its own reward. | S-08 |
| HED defines waiting time twice, incompatibly (analytical M/M/1 vs a `/proc` measurement). | S-07 |

---

## G. Severity ranking for a conversation with your advisor

1. **A6** — the headline result is false on the report's own data. Everything else is secondary to this.
2. **A1, A2** — factual errors in the **abstract**. These are the ones a reviewer notices first.
3. **D1–D3 + D-03** — the §6.2 analysis chain is unsound end to end, and one figure is rigged.
4. **B1–B5** — internal contradictions; cheap to fix in prose, but they undermine trust.
5. **C1–C6** — the report does not accurately describe its own system.
6. **E** — lead with this when framing, so the critique lands as analysis.

---

*Cross-references: `ERRATA.md` (S-xx, with provenance codes), `plan.md` §4
(D-xx defect register) and §4.5 (specification-defect register), `CHANGELOG.md`
(what has been repaired).*
