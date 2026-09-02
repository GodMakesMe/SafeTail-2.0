# ERRATA — corrections to HED and BTP against each other and the code

**Scope.** This document audits the *specifications* — the Heterogeneous Edge
Devices draft (**HED**, `audit/Heterogeneous_Edge_Devices____Tail_Latency.pdf`)
and the April 2026 B.Tech report (**BTP**, `documentation.pdf`) — **against each
other and against the shipped code/dataset**. It is the standalone form of
`plan.md` §4.5 (the specification-defect register) and its companion §7 / B12
correction table. `plan.md` §4 (incl. §4.5) is canonical where anything here is
ambiguous.

**Why it exists.** ST → HED → BTP is a *chain*, not a flat set of requirements.
Each layer fixed some of its parent's errors and added its own. Reading them
flat produces two failure modes this register is designed to prevent:

1. treating a **BTP correction of an HED bug** as a "missing feature" and
   restoring the broken original (see S-01, S-02);
2. treating an **HED or BTP prose statement as authoritative** when the code and
   dataset settle it the other way (see S-14).

**Provenance codes.**

| code | meaning | action |
|---|---|---|
| `HED-BUG` | defect originates in the HED draft | HED is not authoritative here; needs fresh design |
| `BTP-FIX` | HED defect that BTP correctly repaired | **do not restore the HED version** |
| `BTP-NEW` | defect BTP introduced | neither doc authoritative; decision is ours (`plan.md` §13.1) |
| `BTP-KEPT` | HED defect BTP inherited unchanged (often amplified) | both docs wrong |
| `BTP-REGRESS` | BTP replaced a working HED mechanism with a worse one | recover the lost property without the HED bug |

**Rule R9 (`plan.md` §0).** Before implementing anything sourced from HED or
BTP, consult this register. `BTP-FIX` ⇒ the HED version is a known bug.
`BTP-NEW` / `BTP-REGRESS` ⇒ design decision is ours, record it in `plan.md`
§13.1. `HED-BUG` with no BTP row ⇒ both silent/wrong, needs fresh design.

All rows verified 2 Sep 2026 against `audit/Heterogeneous_Edge_Devices____Tail_Latency.pdf`,
`documentation.pdf`, `src/`, and `dataset/`. Guarded end-to-end by gate **G7**
(`tools/verify_types.py`) for S-14 and by rule R9 for the rest.

---

## S-01 — Degree of satisfaction ω · `BTP-FIX`

* **HED** §III-E-2a defines ω **twice and incompatibly**: once as a ratio
  (satisfied / total), then as a malformed piecewise in which the middle branch
  is the *range* `0 ≤ ω ≤ 1` (a range, not a value), the condition `L + κ ≤ Ψ`
  describes the *satisfied* case yet is used for partial credit, and an
  episode-level ratio is mixed with a per-request indicator `B_all`.
* **BTP** §3.7 replaces it with a well-formed piecewise-linear `P(T)`:
  `1` for `T ≤ D1`; `1 − (T − D1)/(D2 − D1)` for `D1 < T ≤ D2`; `0` for `T > D2`.
* **Code:** `src/controller.py` `compute_step_reward` implements BTP's `P(T)`
  correctly (the `T ≤ D1 / D1 < T ≤ D2 / else` ladder).
* **Correction:** none — **record as a strength.** BTP's `P(T)` is a correct
  replacement for HED's malformed ω and is BTP's clearest genuine contribution.
  **Never restore HED's version.** Cite this in the next paper's related-work
  framing; it makes the rest of this register read as analysis, not attack.

## S-02 — Step reward / redundancy pricing · `BTP-REGRESS`

* **HED** §III-E-1: `R_step = Σ_{i=1..d}(l_i − l_total/m) − W_step`. Two HED
  problems: (a) the load term is **sign-inverted for its stated goal** —
  maximising `Σ(l_i − l̄)` selects the *most*-loaded servers, the opposite of
  load balancing; (b) HED itself admits in the next paragraph that the balancing
  term "would force the algorithm to have the same number of service requests
  allocated to each edge server which might not be the optimal solution".
* **BTP** §4.5.1 replaces it with `log(1 + (1−c_m)(1−c_u)(1−g_m)(1−g_u))`, which
  **correctly rewards idle servers** — the sign bug is fixed. **But** HED's form
  carried `d` (the redundancy count) and `−W_step` explicitly, so its reward was
  *sensitive to |A|*; BTP's is strictly monotone non-decreasing in |A|.
* **Code:** `src/controller.py:434` (`np.log(product + 1.0)`), collapsed with
  `np.mean(...)` over all 5 slots at `:676`.
* **Correction:** **BTP fixed the sign and broke the redundancy pricing.** This
  is the true origin of **D-07 / D-08 / M-04**. Workstream **B3** must **not**
  restore HED's term (R9); it must add a *new* explicit cost in |A| on top of
  BTP's corrected headroom product:
  `R_step = (1/|A|)·Σ_{i∈A} log(1+headroom_i) − c_red·(|A|−1)/(β−1)`.
  State this lineage in the paper — it is a defensible narrative.

## S-03 — Stated range of the step reward · `BTP-NEW`

* **BTP** §4.5.1: "This quantity equals 1 when the server is completely free."
  `log(1 + 1) = log 2 ≈ 0.693`, not 1.
* Three mutually inconsistent statements now exist: BTP §4.5.1 (max 1), the code
  docstring `src/controller.py:290` (`1 + log(exp(·) − 1)`), and the code itself
  `:434` (`log(product + 1)`).
* **Correction:** the reward range is **`[0, log 2]`**. Fix BTP §4.5.1, the
  docstring, and any slide that says 1. State once, cite everywhere
  (`plan.md` §13.1 decision 8; workstream B9 / D-27).

## S-04 — Latency equation · `BTP-NEW`

* **BTP** §3.4's display equation is
  `L_j = T_transmission + T_ctrl-queue + T_propagation` — `T_exec` is **absent**,
  though the bullet list beneath it defines `T_exec` and Figure 3.2 shows
  `L_j = T_transmission + T_ctrl-queue + T_exec + T_propagation + T_transmission`.
* **Code** implements a **fourth** model:
  `total_latency = computation + propagation + transmission`, excluding queueing
  (**D-11**; `src/controller.py` latency logging).
* **Correction:** one latency equation. Reconcile BTP §3.4's display equation
  with Figure 3.2 and the code. Workstream **B5** resolves **all four**
  definitions, not just the code's: recommendation — log both
  `service_latency` (comp+prop+trans) and `end_to_end_latency` (+ queueing),
  and plot the one the reward optimises.

## S-05 — M/M/1 waiting time · `BTP-KEPT`

* **HED** §III-D: `W = λ / (μ(μ − λ))` — the correct M/M/1 queue-wait formula,
  but applied to a system whose servers have **finite capacity and reject when
  full** (an `M/M/c/c` Erlang-B *loss* system — **D-24**).
* **BTP** §3.5 reproduces it verbatim, adds `W_j = min_{e_i∈A} W_i`, and
  **promotes it to the abstract**. Neither document defines λ or μ; neither
  appears anywhere in the code.
* **Code:** no `λ`, `μ`, or queue exists. `src/servers.py:199` returns
  `(False, "server full")` — requests are rejected, not buffered.
* **Correction:** **do not implement HED's M/M/1 as written** — it is the wrong
  model for the system that exists. **M-10**'s honest resolution: restate the
  system as `M/M/c/c` Erlang-B, report the rejection rate as a first-class
  metric, and delete the M/M/1 claims from README + abstract + BTP §3.5.
  Building real queues is a separate project.

## S-06 — "Randomness ⇒ Poisson" · `HED-BUG`

* **HED** §II-C-4: "The arrival of service requests is random and therefore it
  is safe to say that the arrival rate follows Poisson's distribution."
  Randomness does not imply Poisson.
* **BTP** §3.5 inherits the claim without examination.
* **Code:** the generator is **uniform** — `random.randint(2,4)` chunks per
  burst, `random.uniform(0.2,0.8)` s between bursts (`src/sender_bursts.py`,
  `src/constants.py`) — **D-22**.
* **Correction:** do not cite HED's justification. If Poisson arrivals are
  wanted, implement exponential inter-arrival times and say so; otherwise delete
  the Poisson/M/M/1 claims (bundle with S-05 / M-10 / D-22).

## S-07 — Waiting time defined twice, incompatibly · `HED-BUG`

* **HED** §III-D gives an analytical M/M/1 `W`; **HED** §IV-E gives a
  `/proc`-based measurement of a Linux process's ready-queue time,
  `W = E − (U_t + K_t + CU_t + CK_t + delayacct_blkio)`. Different quantities,
  never reconciled.
* **Code** implements a **third**: wall-clock elapsed from the receiver's
  arrival stamp to just after the agent returns (`src/controller.py:649`) —
  **D-23**. Reported deadline satisfaction is therefore a function of test-machine
  speed, and SafeTail is handicapped by its own TF forward pass.
* **Correction:** B5 must pick one definition deliberately and justify it;
  replace the wall-clock quantity with a simulated one.

## S-08 — Job-length-proportional priority not implemented in HED itself · `HED-BUG`

* **HED** §II item 9: "higher priority assigned to the larger jobs in proportion
  to their job lengths" — but **no job-length term appears in either HED reward.**
* **BTP** drops the objective silently.
* **Correction:** **M-12** is an *unimplemented HED aspiration*, not a
  regression. Deferring it is legitimate; claiming it as implemented is not.
  When the next paper cites HED, cite it for intent only. (Also blocked on B5 —
  no request size variation exists, **D-34**.)

## S-09 — HED is an unfinished draft · `HED-BUG`

* **HED**'s abstract item 10 reads *in full*: "The reward consists of the su" —
  the numbered contribution list stops mid-word. HED §II item 8 also writes the
  RL output dimension as `nk` (n = user count) where its own §II item 1 implies
  `mk` (m = edge devices).
* **Correction:** cite HED for *intent* only. Where HED and BTP disagree on a
  mechanism, prefer BTP unless this register says otherwise. Correct HED's RL
  output dimension `nk → mk` in any write-up. (See also M-09.)

## S-10 — State description · `BTP-NEW`

* **BTP** §4.2: the state is a `(β+1)`-dim load vector and "across an episode,
  these per-step vectors are accumulated and treated as a variable-length
  sequence, allowing the encoder to capture temporal patterns in server load."
* **BTP** §3.2 says the opposite — length varies because "servers have different
  hardware configurations". §3.2 matches the code; §4.2 is wrong on both counts
  (**D-35**): the real state is `agent.request_to_state_array` flattening *every*
  numeric attribute of `Request`, including the full ragged `server_np`; `nS` is
  dead; there is no temporal accumulation.
* **Correction:** rewrite BTP §4.2 to match §3.2 and the code — variable length
  comes from **hardware heterogeneity**, not temporal accumulation. Remove the
  `(β+1)` claim and the dead `nS`. This makes the encoder contribution
  *legitimate* (the input genuinely is variable-length); §4.2 as written makes
  the encoder look like it solves a non-problem.

## S-11 — How the K-baselines work · `BTP-NEW`

* **BTP** §5.2: the top-1/2/3 servers "selected subset is then passed through
  the same downstream decision pipeline as SafeTail for final request assignment."
* **Code:** **no such downstream pipeline exists.** `_select_minload_servers(x)`
  (`src/controller.py:234`) returns `x` servers and **all `x` are scheduled
  redundantly**. Fixed-K redundant dispatcher, not a budget-neutral candidate
  pre-filter feeding a shared chooser.
* **Correction:** this is the report-side face of **D-09** (the headline
  comparison is not budget-controlled). B6's write-up must correct the
  description *and* the experiment (add `minload_4/5`, `minprop_4/5`, `rand_4/5`
  and a `--match-k` evaluation).

## S-12 — Regressor features · `BTP-NEW`

* **BTP** §4.6 claims "task-specific characteristics (input size, model
  complexity) and concurrent load indicators (CPU utilisation, GPU utilisation,
  number of active jobs)".
* **Code** `_build_features` (`src/server*_regressor/*.py`): `num_speech`,
  `num_detect`, `num_predict`, `total_ops`, `position`, `is_first`, `is_last`,
  `peak_ram`, `peak_gpu`, `peak_gpu_memory`, `total_processing_time` — a static
  trace row keyed by the contention string. **No input size, no model
  complexity, no live utilisation**, and it **leaks the target** (`D-25`:
  `total_processing_time` is a component of the quantity being predicted). Also
  the per-server wrappers are **not identical** — server 3's `_build_features`
  uses a different schema (`num_tasks`, `has_speech`, `peak_cpu`,
  `avg_cpu_clock`, `num_files=500`, no scaler) yet still loads
  `models/server1/…` (see D-02).
* **HED** §IV-D by contrast correctly describes using **live CPU/GPU/RAM
  utilisation** as inputs.
* **Correction:** **HED is the better specification here.** B1's retrain
  (**M-15**) should follow HED §IV-D's feature design — live utilisation of the
  target server at request time + concurrency level + task input characteristics
  — not BTP §4.6's description of it. Disclose the removed target leak; the
  honest held-out R² will fall, and that fall is the finding.

## S-13 — "MLP regressors" · `BTP-KEPT` + `BTP-NEW`

* **HED** §IV-D genuinely describes training MLP regressors (cites
  `github.com/amardeep786/Regressors`). **BTP** §2.6 / §4.6 / abstract repeat it.
* **Code / models:** the 15 shipped pickles are **6 LinearRegression, 8
  RandomForest / DecisionTree, 1 GradientBoosting** (`D-26`; confirmed by gate
  **G0**, `tools/verify_env.py`). The wrapper docstrings even say "Linear
  Regression".
* **Correction:** state plainly which family each shipped regressor is; stop
  calling them MLPs (BTP §2.6, §4.6, abstract; HED §IV-D). Before rewriting,
  **try HED's cited repo** — HED may be truthful about *its* models even though
  BTP is not truthful about *this repo's*; if those MLPs exist they are the
  cheapest correct input to B1. Record the outcome either way.

## S-14 — Request-type identities · `BTP-NEW` ⚠️ (this reverses the 19 Aug 2026 note)

* **The dataset settles it.** `dataset/server{1..5}.csv` column `Scripts
  Executed` contains literally `Speech`, `Detect`, `Predict`; single-letter rows
  map `s → Speech`, `d → Detect`, `p → Predict` on **all five servers**. The
  regressor wrappers agree (`scripts.index("speech"/"detect"/"predict")`).
  Magnitude corroborates: `s` has the longest processing time on every server
  (e.g. server 1: Speech 0.032 s vs Detect 0.0026 s / Predict 0.0029 s;
  server 4: Speech 0.63 s vs Detect 0.055 s) and the longest deadline —
  matching HED §IV-D's own note that speech-to-text runs 60–300 ms while vision
  runs 5–20 ms.
* **Therefore BTP Table 3.1 ("Speech (s)", 100/400 ms) is CORRECT**, and it is
  **BTP §3.3 and §4.6 prose** ("Instance segmentation of images (s)", "Removal
  of noise from audio (p)") and **HED item 9** ("for the segmentation (S) D1
  100ms D2 400ms") that are **wrong**.
* **Correction:** fix the type mapping **in BTP §3.3 / §4.6 prose and HED item
  9 — not in Table 3.1 and not in the code**. `s = Speech`, `d = Detect`,
  `p = Predict`. This **reverses the decision recorded 19 Aug 2026** (which held
  §3.3 prose authoritative). Guarded by gate **G7** (`tools/verify_types.py`,
  PASS). **`p` = "Predict" is defined nowhere** in either paper — an
  undocumented workload type; someone must establish what the `Predict` trace
  actually is before it can be described (`plan.md` §13.1 decision 7).

## S-15 — "Cache access rate" · `BTP-NEW`

* **BTP** Figure 5.1 calls the access rate a "cache access rate". **There is no
  cache anywhere in the system.** Access rate is `|A|/β`, the replication
  fraction.
* **Correction:** terminology fix ("access rate" / replication fraction).
  Combined with **D-32** (it is a *cumulative* mean, structurally monotone) and
  **D-08** (it is a climb up the reward gradient, not "deadline adaptation"),
  **all three claims BTP §6.2 draws from this figure fail.** Re-derive every
  §6.2 claim that rests on it.

## S-16 — ε decay · `BTP-NEW`

* **BTP** §4.4 states **multiplicative** decay `ε ← max(ε_min, ε·(1−γ_ε))`.
* **Code** does **subtractive** `ε -= gamma_decay` (`src/agent.py:355`). The
  constant's name (`gamma_decay`) matches neither.
* **Correction:** pick one, implement it, rename the constant (workstream B9;
  `plan.md` §13.1 decision 9).

## S-17 — BTP §6.2 headline claim · `BTP-NEW`

* **BTP** §6.2: "SafeTail has lower latency than MinLoad, MinProp, and Random at
  all percentiles, with larger gaps at the 95th and 99th."
* **Recomputed from the shipped logs** (`results/reference_v0/`, common 14 723
  requests, `total_latency` ms) — see **D-10**: SafeTail **loses to MinProp at
  every percentile** (p99 116 ms vs 53 ms), and to MinLoad-3 and Rand-3 at
  p95/p99, while spending ~3.5× the compute.
* **Correction:** the single most important thing to resolve before anything is
  submitted. Retract or re-establish the claim **on post-fix data, at matched
  replication budget** (`plan.md` §13.2). D-02 (all regressors load
  `models/server1/`) is the likely mechanical cause and is fixed in B1; F3/F4/F7
  are designed so an honest negative result is still publishable.

---

## Cross-reference — S-row → workstream

| S | provenance | primary workstream | gate |
|---|---|---|---|
| S-01 | BTP-FIX | none (record as strength) | — |
| S-02 | BTP-REGRESS | B3 (new `c_red` cost) | G3 |
| S-03 | BTP-NEW | B9 / D-27 | — |
| S-04 | BTP-NEW | B5 | — |
| S-05 | BTP-KEPT | B9 / M-10 / D-24 | — |
| S-06 | HED-BUG | B5 or B9 / D-22 | — |
| S-07 | HED-BUG | B5 / D-23 | G3 |
| S-08 | HED-BUG | deferred / M-12 | — |
| S-09 | HED-BUG | B12 write-up / M-09 | — |
| S-10 | BTP-NEW | B9 / D-35 | — |
| S-11 | BTP-NEW | B6 / D-09 | — |
| S-12 | BTP-NEW | B1 / M-15 / D-25 | G1 |
| S-13 | BTP-KEPT+NEW | B9 (+B1) / D-26 | G0 |
| S-14 | BTP-NEW | B12 + gate | **G7** |
| S-15 | BTP-NEW | B12 write-up / D-32 / D-08 | — |
| S-16 | BTP-NEW | B9 / decision 13.1(9) | — |
| S-17 | BTP-NEW | runs + §13.2 | G5 / G6 |

---

## S-18 — SafeTail 1.0's published CODE does not implement its published PAPER: target construction · `CODE-BUG`

* **Paper** (camera-ready §IV, and Eq. 6): the reward is *translated into a
  target vector* `V_t` that is a **probability distribution** — "we ensure that
  the sum of all elements in the target vector equals 1". `R = 0` ⇒ one-hot on
  `A_k`; `R < 0` ⇒ initialise every element to `1/(2ⁿ−1)`, set
  `V_t(j) = max(0, 1/(2ⁿ−1) + R)` for `A_k` and every `A_j` with `E_j ⊆ E_k`,
  then distribute the remaining mass equally over the rest.
  **No bootstrapping, no `max` over the next state, no γ, no Bellman equation.**
  That is coherent with the paper's softmax output + categorical cross-entropy.
* **Code** (`_spec_source/v1_agent.py`, i.e. `github.com/Jyotishokhanda/SafeTail`):
  `targets[arange, actions] = rewards + γ·amax(next_q, axis=1)` followed by
  `categorical_crossentropy`. Bellman Q-targets fed to a distribution loss.
  The targets are negative reals that do not sum to 1; softmax cannot represent
  them. This is incoherent and is why the code-faithful port destabilises with
  further training.
* **Correction.** The published SafeTail 1.0 **is not** architecturally broken.
  Its *GitHub implementation* is. Any claim of the form "1.0's softmax+CCE head
  cannot do Q-learning, therefore 2.0's linear head is the essential fix" is a
  statement about the repository, **not** about the paper, and must not be
  attributed to the paper. `baselines/safetail_v1/paper_v1.py` implements the
  paper; `policy_v1.py` remains the code-faithful port. **Report both.**

## S-19 — …and the reward function differs too (Eq. 5) · `CODE-BUG`

| case | paper Eq. 5 | `v1_agent.py` |
|---|---|---|
| `L_R > τ`, `\|E_k\| < n` | `−δ·e^(n−\|E_k\|)` — depends **only** on remaining redundancy headroom | `−α·e^(n−\|E_k\|)·e^(L_R−τ)` — an extra lateness factor the paper does not have |
| `L_R < τ`, `\|E_k\| > 1` | `−δ·e^(L_R−τ)` — **decays** as the finish gets earlier | `−α·e^(\|E_k\|−1)·e^(τ−L_R)` — exponent **sign flipped** so it *grows*, plus a redundancy factor |
| `L_R < τ`, `\|E_k\| = 1` | `0` | **case absent**; returns `−δ` |

The paper's Property 4.4 (missing the target latency is penalised more heavily
than meeting it wastefully) holds for Eq. 5 but not for the code's version.

## S-20 — …and the network architecture differs · `CODE-BUG`

* **Paper §IV:** "The FNN comprises **5 hidden layers with ReLU activations** and
  a Softmax output layer. It is optimized using Adam with categorical
  cross-entropy as the loss function."
* **Code:** `Dense(2·nS, sigmoid) → BatchNorm → Dense(4·nS, sigmoid) →
  BatchNorm → Dense(nA, softmax)` — **2** hidden layers, **sigmoid** not ReLU,
  plus BatchNormalization the paper never mentions.

### Consequence for this project

`plan.md` §14.1 transcribed the 1.0 algorithm **from the code**, and §8.6
recorded "keep softmax+CCE — faithful even though it is wrong for Q-regression".
That note was right about the code and wrong about the paper. The baseline
therefore exists in two variants, and the paper one is the correct referent for
a *paper* comparison:

| policy name | faithful to | notes |
|---|---|---|
| `safetail_v1` | `_spec_source/v1_agent.py` (the GitHub code) | Bellman targets + CCE; destabilises after ~decile 7 |
| `safetail_v1_paper` | the camera-ready paper, Eq. 5 + Eq. 6 | distribution targets; 5×ReLU FNN |

**Credit:** S-18 was raised by an external reviewer of this work; S-19 and S-20
were found while verifying it.
