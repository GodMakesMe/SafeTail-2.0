# SafeTail 2.0 (Heterogeneous) — Master Implementation Plan

**Project root:** `E:\Project\IP_Arani\heterogenous`
**Owner:** Krishna Shukla (krishna23290@iiitd.ac.in) — IIIT-Delhi, advisor Arani Bhattacharya
**Plan authored:** 2 September 2026
**Audience:** Claude Code (code mode), operating alone or as an orchestrator over sub-agents.
**Status of this repo right now:** verbatim copy of `SafeTail-2.0-main`. **Zero code fixes have been applied yet.** Everything in §6–§10 is work to be done.

---

## 0. How to use this document

This is the single source of truth for the next phase of work. Read it end to end before touching code.

**Non-negotiable rules for any agent working in this repo:**

| # | Rule |
|---|------|
| R1 | **Never edit anything under `E:\Project\IP_Arani\SafeTail\`.** That is the pristine SafeTail 1.0 upstream (`github.com/Jyotishokhanda/SafeTail`). It is read-only reference. |
| R2 | **Never `import` across the `src/` ↔ `baselines/` boundary in the `baselines/ → src/` direction beyond the two published seam symbols** named in §8.3. `src/` must never import `baselines/`. |
| R3 | **Deleting `baselines/` must leave a fully working repo.** This is a hard acceptance test (`G4`, §11). |
| R4 | Every behavioural change gets a **defect ID** (`D-xx`) or **missing-feature ID** (`M-xx`) from §4/§5, and that ID appears in the code comment, the commit message, and the changelog row. |
| R5 | Never change a number in `src/constants.py` silently. Constants changes go through §10.5. |
| R6 | Do not delete or overwrite `results/reference_v0/`. It is the frozen pre-fix evidence base. |
| R7 | If a claim in this plan disagrees with the code you are reading, **the code wins** — record the discrepancy in §13.2 and stop for review rather than guessing. |
| R8 | No silent `except Exception: pass`. See §10.3. |
| R9 | **Check §4.5 before implementing anything sourced from HED or BTP.** The three documents are a *chain*, not a flat set: BTP corrected some HED bugs and introduced others. Restoring an HED mechanism that BTP deliberately fixed is a regression, not a feature. |

**Reading order for a fresh agent:** §1 → §2 → §4 → §8 → §12. Skip §6/§7 detail until assigned a workstream.

---

## 1. Project identity — claim verification

The user's claim was: *"SafeTail 2.0 is what our heterogeneous part is"* and *"SafeTail is the original"*. **Both confirmed.** Evidence:

### 1.1 `SafeTail/` = SafeTail 1.0 (original, published)

| Evidence | Value |
|---|---|
| Git remote | `https://github.com/Jyotishokhanda/SafeTail.git` |
| History | 2 commits: `02c89b8 first commit`, `58e04a4 Add aman folder to SafeTail repo` |
| Code root | `SafeTail/SafeTail/aman/ut_dqn/` |
| Servers | `beta = 4` (root `constants.py`), 5 or 10 in the per-task configs |
| Architecture | `Dense(2·nS, sigmoid) → BN → Dense(4·nS, sigmoid) → BN → Dense(nA, **softmax**)`, loss `categorical_crossentropy` |
| Reward | τ-referenced 5-case asymmetric penalty (paper Def. 4.3 / Eq. 5) — `agent.py:231-302` |
| State | Flat homogeneous dict: `LOAD[β], MESSAGE_SIZE, RESOLUTION, BANDWIDTH, PROPOGATION[β]` |
| Workloads | `yolo` (detection), `instance` (instance segmentation), `noise` (noise removal) — three separate runs |
| Servers modelled as | Homogeneous; differentiated only by an integer **user-count load** and a ping pool |

### 1.2 `SafeTail-2.0-main/` = the heterogeneous successor

| Evidence | Value |
|---|---|
| Servers | `beta = 5`, each with its own `dataset/server{i}.csv` hardware+trace profile |
| State | Variable-length, per-server hardware vector (`user.fill_server_np`) → learned encoder + `GlobalAveragePooling1D` |
| Architecture | Encoder (`Dense64→Drop→Dense32→Drop→GAP→Dense24`) + DQN head (`Dense48→BN→Drop→Dense64→BN→Drop→Dense31` **linear**), loss **MSE** |
| Reward | Two-level: resource-headroom step reward + episodic (Σγⁱ·R_step + ω − W̄) |
| Transport | Real TCP: `sender_bursts.py → receiver.py → controller.py` |
| Lineage | Central controller + per-device heterogeneous state + encoder + two-tier deadlines all trace to `Heterogeneous_Edge_Devices____Tail_Latency.pdf` (HED draft) |

**Conclusion:** the heterogeneous work *is* SafeTail 2.0. `heterogenous/` is therefore seeded from `SafeTail-2.0-main`, and SafeTail 1.0 becomes a **baseline to be ported**, not a codebase to be merged.

### 1.3 Where the audit lives

Copied into `audit/`:

| File | What it is |
|---|---|
| `SafeTail_2.0_Codebase_Analysis_v2.pdf` | **Primary audit** (19 Aug 2026), §1–§13. Function inventory, call graph, traceability matrix, gap analysis, 16 numbered defects, revised remediation priorities. |
| `SafeTail_2.0_Addendum_A.pdf` | §11–§13 only (subset of v2; kept for citation stability) |
| `SafeTail_2.0_Codebase_Analysis_v1_superseded.pdf` | First pass. Superseded. Do not cite. |
| `analysis_v2_extracted.txt` | Plain-text extraction of v2 — **grep this instead of re-parsing the PDF** |
| `SafeTail_Camera_Ready.pdf` | ST — the SafeTail 1.0 paper |
| `Heterogeneous_Edge_Devices____Tail_Latency.pdf` | HED — the heterogeneous draft |
| `../documentation.pdf` | BTP — the Apr 2026 B.Tech report that claims to document this code |

Krishna is **not** an author of the BTP report (Shamik Sinha, Shrutya Chawla, Medha Kashyap, Shivankar Srijan Singh, 22 April 2026, advisor Arani Bhattacharya); he inherited the codebase. Treat BTP claims as *assertions to be checked*, not as specification.

**The three documents form a chain, not a set.** ST → HED → BTP → code, where each layer corrected some of its parent's errors and added its own. Concretely: **HED is an unfinished draft** — its contribution list is truncated mid-word ("The reward consists of the su"), it defines waiting time twice incompatibly, and its step reward is sign-inverted relative to its own stated goal. **BTP repaired some of that** (its `P(T)` is a genuine, correct replacement for HED's malformed ω) **and broke other things** (it removed the only redundancy-sensitive term in the reward, and it misdescribes its own state, features, baselines and ε schedule). So:

* neither document is authoritative on its own;
* "the code does not implement X from HED" is **not automatically a defect** — X may be something BTP correctly removed;
* "BTP says X" is **not automatically a specification** — BTP contradicts itself in at least four places.

§4.5 is the register that resolves this, and **R9** makes consulting it mandatory. Where §4.5 has no row and the documents agree, the documents win; where they disagree and §4.5 is silent, stop and ask (R7).

---

## 2. Repository layout and the isolation contract

### 2.1 As it stands now (created this session, verbatim copy)

```
heterogenous/
├── plan.md                    ← this file
├── README.md                  (inherited from 2.0 — contains M/M/1 claims that are false, see D-13)
├── requirements.txt           (UTF-16LE, and missing scikit-learn — see D-01)
├── run_all_baselines.sh       (hardcodes /home/jyoti/miniconda3/bin/python — see D-14)
├── documentation.pdf          (BTP report)
├── src/                       3,222 lines, 8 modules — THE HETEROGENEOUS CORE
│   ├── constants.py           (69)
│   ├── main.py                (201)   entry point
│   ├── sender_bursts.py       (153)   synthetic traffic generator
│   ├── receiver.py            (340)   TCP receiver
│   ├── user.py                (320)   Request + per-server state hydration
│   ├── servers.py             (272)   Server model, admission, delay estimation
│   ├── agent.py               (712)   DQN + encoder
│   ├── controller.py         (1155)   orchestrator, rewards, training
│   └── server{1..5}_regressor/        15 predictor wrappers  ← ALL BROKEN, see D-02
├── dataset/                   5 server CSVs + propagation_delays.pkl
├── models/server{1..5}/       15 pickled sklearn regressors
├── testing/                   2 notebooks + test_agent.py
├── plotting/plot_baselines.ipynb  ← contains a rigged offset, see D-03
├── docs/                      poster, ST, HED
├── audit/                     the analysis PDFs (see §1.3)
├── results/
│   └── reference_v0/          FROZEN. The shipped pre-fix results. READ-ONLY (R6).
│       ├── safetail_training_logs/
│       ├── baselines/training_logs_{minload,minprop,rand}_{1,2,3}/
│       └── deadline_manipulation/{80,150}_percent*/
├── baselines/                 ← DELETABLE UNIT (§8)
│   └── safetail_v1/_spec_source/   11 read-only SafeTail 1.0 files, for porting reference
├── figures/                   (empty — final paper figures land here)
└── tools/                     (empty — audit/verification scripts land here)
```

### 2.2 The three-way isolation contract

There are **three** independent artefacts. They must never blend.

```
   ┌───────────────────────────┐        ┌──────────────────────────────────────┐
   │  E:\...\SafeTail\         │        │  E:\...\heterogenous\                │
   │  SafeTail 1.0 upstream    │        │                                      │
   │  READ-ONLY (R1)           │        │  ┌────────────────────────────────┐  │
   │  never imported, never    │──spec──┼─▶│ baselines/    DELETABLE (R3)   │  │
   │  edited, never on         │  only  │  │  safetail_v1/  ← ported policy │  │
   │  sys.path                 │        │  └──────────┬─────────────────────┘  │
   └───────────────────────────┘        │             │ registers via          │
                                        │             │ policy_registry only   │
                                        │  ┌──────────▼─────────────────────┐  │
                                        │  │ src/     HETEROGENEOUS CORE    │  │
                                        │  │ never imports baselines/       │  │
                                        │  └────────────────────────────────┘  │
                                        └──────────────────────────────────────┘
```

* **SafeTail 1.0 → baseline is a PORT, not a BRIDGE.** `baselines/safetail_v1/` reimplements the 1.0 *algorithm* (§14.1 is the spec) against the 2.0 environment API. It does **not** `sys.path.append` into `SafeTail/`, does not import `v1_agent.py`, and does not load 1.0's pickles.
* `baselines/safetail_v1/_spec_source/*.py` are **frozen read-only copies for a human/agent to read while porting**. They are never imported. A lint gate (`G4c`, §11) enforces this.
* `src/` gains exactly **one** new file (`src/policy_registry.py`) and **one** replaced block (the `BASELINE_MODE` if-chain). Both are permanent core infrastructure that stand alone with zero baselines present.

---

## 3. Environment and bootstrap

### 3.1 Known-good environment

| Item | Value | Note |
|---|---|---|
| Python | 3.11 or 3.12 | TF 2.20 supports both |
| tensorflow | 2.20.0 | pinned in requirements |
| keras | 3.13.1 | pinned |
| numpy | 1.26.4 | **do not upgrade to 2.x** — the pickled sklearn models and `pad_sequences` path are numpy-1 era |
| pandas | 2.3.3 | pinned |
| matplotlib | 3.8.1 | pinned |
| **scikit-learn** | **MISSING FROM requirements.txt** | required to unpickle `models/*/*.pkl` — see D-01 |
| **seaborn** | **MISSING** | imported by `plotting/plot_baselines.ipynb` |
| **scipy** | **MISSING** | needed by the TLORA baseline (`weibull_min`) if it is ever ported |

### 3.2 Bootstrap (task B0, do this first)

```bash
# 1. requirements.txt is UTF-16LE with CRLF. pip cannot read it as-is.
python - <<'PY'
import io
raw = open('requirements.txt','rb').read()
txt = raw.decode('utf-16').replace('\r\n','\n')
open('requirements.txt','w',encoding='utf-8',newline='\n').write(txt)
PY

# 2. add the three missing packages (versions: pick the newest compatible with numpy 1.26)
#    scikit-learn>=1.3,<1.6   seaborn>=0.13   scipy>=1.11,<1.14

# 3. install
python -m venv .venv && .venv/Scripts/activate       # Windows
pip install -r requirements.txt

# 4. PROVE the regressors load (this is gate G0)
python tools/verify_regressors.py
```

`tools/verify_regressors.py` must be written as part of B0 and must **fail loudly** if any of the 15 models cannot be unpickled. Reason: today the failure is silent (D-02b).

---

## 4. Verified defect register

Every row below was **re-verified against the code in this repository on 2 Sep 2026**, not copied from the audit on faith. `Δ` marks findings **not present in the audit PDF** — these are new.

Severity: **S1** = invalidates results · **S2** = materially distorts results · **S3** = correctness/quality · **S4** = hygiene.

### 4.1 S1 — invalidates results

| ID | Defect | Location | Verified evidence |
|---|---|---|---|
| **D-02** Δ | **All 15 regressor wrappers load `models/server1/` and `dataset/server1.csv`.** Servers 2–5 predict computation delay with server 1's model on server 1's trace. *The computation-latency path is not heterogeneous at all.* | `src/server{2,3,4,5}_regressor/*.py`, the `self.model_path` / `self.csv_path` lines | Grepped all 15 files: every one resolves `"models" / "server1"` and `"dataset" / "server1.csv"`. 15/15. |
| **D-02b** Δ | **Module-name collision masks D-02.** `_load_predictors_from_regressor_folder` does `sys.path.insert(0, server{i}_regressor)` then `importlib.import_module("detect_predictor")`. Module names are identical across the 5 folders, so `sys.modules` returns **server 1's already-imported module** for servers 2–5 regardless of `sys.path`. Even after fixing D-02, the import mechanism would still hand back server 1's class. | `src/servers.py:43-79` | Read the loader. No `sys.modules` invalidation, no package qualification, no `importlib.util.spec_from_file_location`. |
| **D-02c** Δ | **Loader failures are silent.** Each of the three predictor imports is wrapped in `except Exception: self.predictors['x'] = None`. With `sklearn` absent (D-01) all 15 loads fail, `predict_from_combination` is never called, and `_predict_using_letter` falls back to a **single-letter CSV row lookup that ignores contention entirely**. The run completes normally and produces plausible numbers. | `src/servers.py:48-77`, `src/servers.py:108-138` | Read both. Fallback uses `mask = Combination == letter`, not `combined_str`. |
| **D-01** Δ | `requirements.txt` (72 pinned packages) **contains no scikit-learn**, yet every `models/*/*.pkl` is a pickled sklearn estimator. A clean install per the repo's own instructions produces a system where D-02c fires on every server. | `requirements.txt` | `grep -c '^scikit-learn==' → 0`. Unpickling in a sklearn-free env raises `ModuleNotFoundError: No module named 'sklearn'` (reproduced). |
| **D-04** | **Replay memory stores the post-outcome state.** `controller.py:724-729` appends the **live `Request` object**; `agent.store()` (which flattens it) is not called until `finalize_episode` at `controller.py:798`. By then `total_processing_delay` holds *realised* latencies (overwritten at `:671`), `combination` holds the contention string (`servers.py:215`), and `queue_waiting_time` is set (`:649`). The state vector in replay therefore **contains the outcome of the action it is paired with**. | `src/controller.py:724-729`, `:798` | Read both sites. `"state": request, "next_state": request` — same live object. |
| **D-05** | **`next_state == state`.** Same dict literal stores the identical object in both slots, so the Bellman bootstrap is `r + γ·max Q(s)` on the same state. | `src/controller.py:724-729` | Verified verbatim. |
| **D-06** | **Step rewards are discarded.** `finalize_episode` stores every experience in the episode with the single `episodic_reward`, overwriting the per-step credit computed at `:676`. | `src/controller.py:796-803` | Verified: `reward=episodic_reward` inside `for exp in self.step_experiences`. |
| **D-07** | **The step reward is monotone non-decreasing in \|A\|.** `compute_step_reward` zero-fills unselected slots and fills selected ones with `log(product+1) ∈ [0, log2]`; the controller collapses with `np.mean(...)` over **all 5 slots**. So `R_step = (1/5)·Σ_{i∈A} log(1+headroom_i)`. Every extra server adds a non-negative term to a fixed denominator. **Nothing anywhere prices redundancy.** All three episodic terms push the same way. | `src/controller.py:285`, `:434`, `:676` | Read all three. Confirmed `rewards = np.zeros(num_servers)` + `np.mean`. |
| **D-08** | **The agent over-replicates as a direct consequence of D-07.** Per-episode selection rate climbs 0.51 → 0.78 monotonically; request-wise median 0.8 (4 of 5 servers); 73.9% of requests go to exactly 4 servers; mean K ≈ 3.55. BTP §6.2 reads this as "deadline adaptation" — it is a climb up the reward gradient. | `results/reference_v0/safetail_training_logs/request_wise_access_log.csv` | Recomputed from the shipped log. |
| **D-09** | **The headline baseline comparison is not budget-controlled.** Baselines are hard-capped at K∈{1,2,3} (`controller.py:622-639`); SafeTail runs free over all 31 subsets at mean K≈3.55, exceeding the K=3 ceiling in 83.3% of episodes. | `src/controller.py:622-639` | Read the if-chain. |

### 4.2 S1 — the empirical result itself

| ID | Finding | Evidence |
|---|---|---|
| **D-10** Δ | **On the shipped logs, SafeTail 2.0 does not beat MinProp — it loses to it at every percentile, and also loses to MinLoad-3 and Rand-3.** Recomputed from `results/reference_v0/`, all modes truncated to the common 14,723 requests, `total_latency` in ms: |

```
mode          p50      p90      p95      p99
safetail    39.88    59.88    75.66   115.64
minload_1   52.08   121.41   137.88   164.82
minload_2   42.09    81.98    99.63   127.32
minload_3   31.88    49.93    54.29    81.67     ← beats SafeTail
minprop_1   30.33    47.98    50.81    66.48     ← beats SafeTail
minprop_2   29.93    46.63    49.08    53.37     ← beats SafeTail (p99: 53 vs 116)
minprop_3   29.88    46.68    49.18    56.31     ← beats SafeTail
rand_1      64.78   125.83   140.58   165.38
rand_2      44.57    89.13   103.89   125.28
rand_3      36.08    57.63    72.36   102.99     ← beats SafeTail at p95/p99
```

> SafeTail's p99 is **117% worse** than MinProp-2's, while spending ~3.5× the compute. Including `queueing_delay` widens the gap further (SafeTail p95 181 ms vs MinProp-2 121 ms). **Any claim of a tail-latency win over MinProp is unsupported by the data in this repository.** D-02 is the most likely mechanical cause: with computation delay identical across servers, propagation is the only server-discriminating signal, and MinProp optimises propagation directly.

| ID | Defect | Location | Verified evidence |
|---|---|---|---|
| **D-03** Δ | **The plotting notebook adds a hardcoded +0.005 s (5 ms) to the MinProp baselines only**, in two places, before percentiles are taken. At a mean total latency of 44 ms this is an **11% inflation of one baseline family**. It does not reverse D-10 (MinProp still wins by ~20 ms at p95) but it is a rigged comparison and must be removed. | `plotting/plot_baselines.ipynb` cell 3 (`load_latencies(mode, v=0.005) if mode in ["minprop_1","minprop_2","minprop_3"]`) and cell 9 (same conditional) | Read both cells. |
| **D-11** Δ | **`total_latency` excludes `queueing_delay`.** Verified numerically: `total_latency == 1000·(computation+propagation+transmission)` for all 15,225 rows (max residual 5e-4 ms). But `queueing_delay` (mean **68 ms**) is *larger than the entire plotted total* (mean 44 ms) and **is** what feeds `P(T)` and the episodic waiting penalty. The reward optimises one quantity; the headline figure reports a different one. | `src/controller.py` latency logging; `results/reference_v0/.../latency_log.csv` | Recomputed over all rows. |
| **D-12** Δ | **~45% of the reported metric is policy-independent noise.** Mean contribution to `total_latency`: transmission **45.5%**, propagation 35.0%, computation **19.5%**. Transmission is `random.choice([18.5,19.2,20,21.5,22])/1000` — five distinct values, independent of server, `message_size` and `bandwidth` (which are hardcoded 1024 / 20 for every request). Nearly half the metric is a coin flip no policy can influence, which compresses every inter-policy gap. | `src/servers.py:93-94` (`_get_tramission_delay`), `src/main.py` request_factory | Verified: exactly 5 distinct transmission values in 15,225 rows. |

### 4.3 S2 — materially distorts results

| ID | Defect | Location | Evidence |
|---|---|---|---|
| **D-15** Δ | **`dataset/server1.csv` and `server2.csv` are byte-identical** (md5 `a2f7438…`). There are only **4 distinct server profiles for 5 servers**. | `dataset/` | `md5sum` match. |
| **D-16** Δ | **Servers 3 and 4 have no GPU columns at all** (21 cols vs 23; missing `Peak GPU Usage (%)`, `Peak GPU Memory (MB)`, `Total GPU Memory (GB)`, `Average GPU Clock (MHz)`, `GPU Model`, `Files Processed Per Script`). `populate_request_from_csv` defaults them to 0, so in the step reward `gm = 0/max(0,1e-8) = 0` and `gu = 0/100 = 0`. Product becomes `(1-cm)(1-cu)·1·1` — **CPU-only servers get a structurally inflated headroom reward purely from missing telemetry.** | `src/user.py:242-253`, `src/controller.py:411-434` | Column diff + read the division guards. |
| **D-17** Δ | **Off-by-one on the step-reward failure path.** Success path writes `rewards[i] = reward` (`:445`) where `i = server_idx` is 0-based (`:392`); the `except` writes `rewards[server_idx - 1] = 0.0` (`:449`). A failure on server 0 writes `rewards[-1]`, silently corrupting server 4's reward and leaving server 0's at its initialised 0. | `src/controller.py:392`, `:445`, `:449` | Read all three. |
| **D-18** | **The estimate the agent sees is not the outcome it is graded on.** Phase 2 calls `compute_request_time()` on all 5 servers; phase 7 `schedule_request()` calls it **again**, drawing fresh propagation and transmission samples. | `src/controller.py:564` vs `:659` | Read both call sites; `_get_propogation_delay`/`_get_tramission_delay` are fresh `random.choice`. |
| **D-19** | **`request.combination` is overwritten with the contention string** after scheduling (e.g. `"d"` → `"dps"`). Every downstream `combination[0]` type check reads the first letter of a mutated string, and it is what lands in the `request_type` column of `latency_log.csv` — **192 distinct values** where 3 were intended. The notebook's type filter (`str.startswith(prefix)`) happens to recover the right letter, undocumented and fragile. | `src/servers.py:215`, `src/controller.py:335` | Counted 192 distinct `request_type` strings in the shipped log. |
| **D-20** | **`request_x_done` counters are unconditional**, so the completion ratio in the episodic waiting-time denominator is ≡ 1.0 by construction. The deadline-conditional increments are commented out. | `src/controller.py:366-377`, `:1114-1119` | Read both. |
| **D-21** | **Unbounded recursion under saturation.** `process_step` sleeps 0.1 s and calls itself when all servers are full — no depth limit, no backoff cap. Sustained saturation overflows the stack. | `src/controller.py:546-550` | Verified verbatim. |
| **D-22** | **Arrivals are uniform, not Poisson.** `random.randint(2,4)` chunks per burst, `random.uniform(0.2,0.8)` s between bursts. The M/M/1 premise asserted in README and BTP §3.5 fails at the source. | `src/sender_bursts.py`, `src/constants.py:14-18` | Verified. |
| **D-23** | **`queue_waiting_time` is wall-clock Python execution time**, measured from the receiver's arrival stamp to just after the agent returns — CSV lookups, regressor inference, a TensorFlow forward pass, lock contention. It feeds both the reward penalty and `T` in `P(T)`. **Reported deadline satisfaction is a function of how fast the test machine is.** SafeTail pays a TF forward pass that the heuristics do not — a systematic handicap of the learned policy on its own metric. | `src/controller.py:649` | Verified; mean 68 ms vs computation mean 0.0086 s. |

### 4.4 S3/S4 — correctness and hygiene

| ID | Defect | Location |
|---|---|---|
| D-24 | **No queues anywhere.** Servers are a `c=4` **loss system** (Erlang-B): `schedule_request` returns `(False,"server full")` — requests are *rejected, not buffered*. The controller's "queue" is a mutex. `Receiver.pending` is a TCP-connection backlog (`MAX_QUEUE=20`), not a request queue. `Controller.get_queue_lengths()` is the only queue accessor and is never called. | `src/servers.py:13,199-205,233-249`; `src/controller.py:254,1106-1109`; `src/receiver.py:246,294-298` |
| D-25 | **Target-leaking regressor feature**: `_build_features` passes `total_processing_time` from the same CSV row into the model predicting a component of it. | `src/server*_regressor/*.py` (`_build_features`) |
| D-26 | **No MLP regressors exist** despite ST, HED and BTP all claiming them. Byte-inspection of the 15 pickles: **6 LinearRegression, 8 RandomForest/DecisionTree, 1 GradientBoosting**. The wrapper docstrings even say "Linear Regression". | `models/server*/*.pkl` |
| D-27 | Docstring/code mismatch in the step reward: docstring says `1+log(exp(·)-1)`, code computes `log(product+1)`. | `src/controller.py:290` vs `:434` |
| D-28 | `Controller.assign_request()` (def at `:261`) unpacks 3 values from `schedule_request`'s 7-tuple at `:267` — raises if ever called. Never called. | `src/controller.py:261,267` |
| D-29 | Orphans: `Controller.{assign_request, dispatch_to_agent, get_queue_lengths, save_checkpoint, export_training_data, generate_testing_plots}`, `Server.{time_until_next_free, print_active_requests}`, `DQNAgent.{reward, get_min_delay, save_metrics_summary}`. Two have fully commented-out bodies; one is an empty `pass` stub. | §5 of the audit |
| D-30 | Declared-but-unused constants: `nS` (real input is `shape=(None,1)`), `episode_size=4` (controller uses `chunks_per_episode=3`), `max_load=5` (`servers.py` hardcodes `MAX_CONCURRENT_REQUESTS=4`). | `src/constants.py:8,22,27` |
| D-31 | `np.load(allow_pickle=True)` on socket payloads — arbitrary code execution if the port ever leaves localhost. | `src/receiver.py:167` |
| D-13 | README and BTP §3.5 assert M/M/1 queueing at controller and servers. **No `λ`, `μ`, or queue exists.** Documentation is false. | `README.md`, `documentation.pdf` §3.5 |
| D-14 | `run_all_baselines.sh` hardcodes `PYTHON=/home/jyoti/miniconda3/bin/python`; comment says CSVs land in `src/results/` and then `src/logs/` — neither is where they land. | `run_all_baselines.sh` |
| D-32 | **`access_rate_log.csv` is a cumulative (expanding) mean, not a per-episode series.** Its final value is bit-identical to the overall request-wise mean. A cumulative average cannot be non-monotonic, so it structurally cannot show the "adaptation" BTP §6.2 claims. Baseline runs log all-zeros for access rate (the logging block sits inside `if BASELINE_MODE == "safetail"`). | `src/controller.py` access-rate logging |
| D-33 | **The run processes ~15,225 requests, not 500,000.** `no_of_burst = 1000` binds before `total_no_request = 500000` (which only sizes the pre-generated pool). BTP Table 4.2 overstates by ~33×. | `src/sender_bursts.py:105`, `src/constants.py:5,9` |
| D-34 | **Requests carry no workload variation.** `message_size=1024`, `bandwidth=20` are literals for every request; only the type letter is random. BTP §3.3's "input/output data sizes" and "FLOPs, input dimensions" do not exist. | `src/main.py` request_factory |
| D-35 | **BTP §4.2 misdescribes the state** as a `(β+1)`-dim load vector. The real state is `agent.request_to_state_array` flattening *every* numeric attribute of `Request`, including the full ragged `server_np`. `nS` is dead. **Note: this makes the encoder contribution legitimate** — the input genuinely is variable-length. The report understates its own system. | `src/agent.py:37`, `:231` |

### 4.5 Specification-defect register — the three-layer lineage

§4.1–§4.4 audit **code against specification**. This section audits **the specifications against each other**, because they are not a flat set of requirements: ST → HED → BTP is a chain in which each layer corrected some of its parent's errors and introduced new ones. Reading them as a flat set produces two dangerous mistakes:

* **treating a BTP correction of an HED bug as a "missing feature"** and restoring the broken original, and
* **treating an HED statement as authoritative** when BTP already replaced it for good reason — or the reverse.

Provenance codes: **`HED-BUG`** = defect in the HED draft · **`BTP-FIX`** = HED defect that BTP correctly repaired (**do not restore the HED version**) · **`BTP-NEW`** = defect BTP introduced · **`BTP-KEPT`** = HED defect BTP inherited unchanged and amplified · **`BTP-REGRESS`** = BTP replaced a working HED mechanism with a worse one.

All rows verified against the primary documents (`audit/Heterogeneous_Edge_Devices____Tail_Latency.pdf`, `documentation.pdf`) on 2 Sep 2026.

| ID | Provenance | Finding | Consequence for our work |
|---|---|---|---|
| **S-01** | `BTP-FIX` | **Degree of satisfaction ω.** HED §III-E-2a defines it twice and incompatibly: once as a ratio (satisfied/total), then as a malformed piecewise where the middle branch is the *range* `0≤ω≤1` (a range, not a value), the condition `L+κ ≤ Ψ` describes the *satisfied* case yet is used for partial credit, and an episode-level ratio is mixed with a per-request indicator `B_all`. BTP §3.7 replaces it with a well-formed piecewise-linear `P(T)`: `1` for `T≤D1`; `1−(T−D1)/(D2−D1)` for `D1<T≤D2`; `0` for `T>D2`. | **Keep BTP's version. Never restore HED's.** The code implements BTP's correctly. This is BTP's clearest genuine contribution. |
| **S-02** | `BTP-REGRESS` | **Step reward.** HED §III-E-1: `R_step = Σ_{i=1..d}(l_i − l_total/m) − W_step`. Two problems in HED: (a) **the load term is sign-inverted for its stated goal** — maximising `Σ(l_i − l̄)` selects the *most*-loaded servers, the opposite of load balancing; (b) HED itself admits in the next paragraph that the balancing term "would force the algorithm to have the same number of service requests allocated to each edge server which might not be the optimal solution". BTP §4.5.1 replaces it with `log(1+(1−c_m)(1−c_u)(1−g_m)(1−g_u))`, which **correctly rewards idle servers** — the sign bug is fixed. **But** HED's form contained `d` (the redundancy count) and `−W_step` explicitly, so its reward was *sensitive* to `\|A\|`; BTP's is strictly monotone non-decreasing in `\|A\|`. | **BTP fixed the sign and broke the redundancy pricing.** This is the true origin of **D-07/D-08/M-04**. B3 must not restore HED's term — it must add a *new* explicit cost in `\|A\|` on top of BTP's corrected headroom product. State this lineage in the paper; it is a defensible narrative. |
| **S-03** | `BTP-NEW` | **BTP §4.5.1 misstates the range of its own reward:** "This quantity equals 1 when the server is completely free." `log(1+1) = log 2 ≈ 0.693`, not 1. There are now **three** mutually inconsistent statements of this formula: BTP §4.5.1 (claims max 1), the code docstring `controller.py:290` (`1+log(exp(·)−1)`), and the code itself `:434` (`log(product+1)`). | Fix all three to one statement in B9 (**D-27**). Whichever is chosen, say so once and cite it everywhere. |
| **S-04** | `BTP-NEW` | **BTP §3.4's latency equation omits execution time.** The display equation is `L_j = T_transmission + T_ctrl-queue + T_propagation` — `T_exec` is absent, even though the bullet list beneath it defines `T_exec` and Figure 3.2 shows `L_j = T_transmission + T_ctrl-queue + T_exec + T_propagation + T_transmission`. BTP's own equation, bullet list and figure are three different latency models. | The code implements a **fourth**: `total_latency = comp + prop + trans`, excluding queueing (**D-11**). B5's "decide and document what latency means" must resolve all four, not just the code. |
| **S-05** | `BTP-KEPT` | **M/M/1.** HED §III-D states `W = λ/(μ(μ−λ))` — the correct M/M/1 queue-wait formula, but applied to a system whose servers have finite capacity and **reject** when full (an `M/M/c/c` Erlang-B loss system, **D-24**). BTP §3.5 reproduces it verbatim, adds `W_j = min_{e_i∈A} W_i`, and **promotes it to the abstract**. Neither document defines λ or μ, and neither appears anywhere in the code. | **Do not implement HED's M/M/1 as written** — it is the wrong model for the system that exists. M-10's honest resolution is to restate the system as Erlang-B and report the rejection rate, or to build real queues as a separate project. |
| **S-06** | `HED-BUG` | **HED's Poisson assumption is a non-sequitur.** §II-C-4: "The arrival of service requests is random and therefore it is safe to say that the arrival rate follows Poisson's distribution." Randomness does not imply Poisson. BTP §3.5 inherits the claim without examination. The generator is uniform (**D-22**). | Whichever way D-22 is resolved, do not cite HED's justification. If Poisson arrivals are wanted, implement exponential inter-arrival times and say so. |
| **S-07** | `HED-BUG` | **HED defines waiting time twice, incompatibly.** §III-D gives an analytical M/M/1 `W`; §IV-E gives a `/proc`-based measurement of a Linux process's ready-queue time, `W = E − (U_t + K_t + CU_t + CK_t + delayacct_blkio)`. These measure different things and are never reconciled. | The code implements a **third** (wall-clock elapsed, **D-23**). B5's replacement must pick one definition deliberately and justify it. |
| **S-08** | `HED-BUG` | **HED's stated priority objective has no implementation in its own reward.** §II item 9: "higher priority assigned to the larger jobs in proportion to their job lengths" — but no job-length term appears in either HED reward. BTP drops the objective silently. | **M-12** is therefore an *unimplemented HED aspiration*, not a regression. Deferring it is legitimate; claiming it as a feature is not. |
| **S-09** | `HED-BUG` | **HED's abstract is truncated mid-sentence.** Item 10 reads in full: "The reward consists of the su". The draft's numbered contribution list simply stops. HED §II item 8 also writes the RL output dimension as `nk` where `n` is the user count; over devices it should be `mk` (`m` = edge devices, per HED item 1). | HED is an **unfinished draft**, not a specification. Cite it for *intent* only. Where HED and BTP disagree on a mechanism, prefer BTP unless §4.5 says otherwise. |
| **S-10** | `BTP-NEW` | **BTP §4.2 invents a temporal-sequence story and contradicts its own §3.2.** §4.2: the state is a `(β+1)`-dim load vector and "across an episode, these per-step vectors are accumulated and treated as a variable-length sequence, allowing the encoder to capture temporal patterns in server load." §3.2 says the opposite — length varies because "servers have different hardware configurations". §3.2 is right and matches the code; §4.2 is wrong on both counts (see **D-35**), and its temporal-accumulation claim describes something the code never does. | When writing up the encoder, use §3.2's justification. §4.2's version makes the encoder look like it solves a problem that does not exist. |
| **S-11** | `BTP-NEW` | **BTP §5.2 misdescribes how the baselines work.** It says the top-1/2/3 servers "selected subset is then passed through the same downstream decision pipeline as SafeTail for final request assignment." **No such downstream pipeline exists.** `_select_minload_servers(x)` returns `x` servers and all `x` are scheduled redundantly. The report describes a budget-neutral candidate pre-filter; the code implements a fixed-K redundant dispatcher. | This is the report-side face of **D-09**. A reader of §5.2 would not realise the comparison is budget-confounded. B6's write-up must correct the description as well as the experiment. |
| **S-12** | `BTP-NEW` | **BTP §4.6 misdescribes its own regressor features.** Claims "task-specific characteristics (input size, model complexity) and concurrent load indicators (CPU utilisation, GPU utilisation, number of active jobs)". Actual `_build_features`: `num_speech, num_detect, num_predict, total_ops, position, is_first, is_last, peak_ram, peak_gpu, peak_gpu_memory, total_processing_time`. No input size, no model complexity, **no live utilisation** — only a static trace row keyed by the contention string — and it **leaks the target** (**D-25**). HED §IV-D by contrast correctly describes using live CPU/GPU/RAM utilisation as inputs. | **Here HED is more faithful than BTP.** B1's retrain should follow HED §IV-D's feature design, not BTP §4.6's description of it. |
| **S-13** | `BTP-KEPT` + `BTP-NEW` | **MLP claim.** HED §IV-D genuinely describes training MLP regressors (public repo cited: `github.com/amardeep786/Regressors`). BTP §2.6/§4.6/abstract repeat it. The 15 shipped pickles are **6 LinearRegression, 8 RandomForest/DecisionTree, 1 GradientBoosting** (**D-26**). HED may well be truthful about *its* models; BTP is not truthful about *this repo's*. | Do not assume the shipped models are the ones HED describes. If HED's repo is reachable, compare — that is the cheapest route to a correct regressor for B1. |
| **S-14** | `BTP-NEW` | **Request-type identities are wrong in BTP prose — and the earlier project note about this was backwards.** The dataset settles it: `dataset/server{1..5}.csv` column `Scripts Executed` contains literally `Speech`, `Detect`, `Predict`, and single-letter rows map `s→Speech`, `d→Detect`, `p→Predict` on **all five servers**. The regressor wrappers agree (`scripts.index("speech"/"detect"/"predict")`). Therefore **BTP Table 3.1 ("Speech (s)", 100/400 ms) is CORRECT**, and it is **BTP §3.3 and §4.6 prose** ("Instance segmentation of images (s)", "Removal of noise from audio (p)") and **HED item 9** ("for the segmentation (S) D1 100ms D2 400ms") that are wrong. Corroborated by magnitude: `s` has the longest processing time on every server (0.032 s vs 0.0026/0.0029 on server 1) and the longest deadline, matching HED §IV-D's own observation that speech-to-text runs 60–300 ms while vision runs 5–20 ms. | ⚠️ **This reverses the decision recorded on 19 Aug 2026** (which held §3.3 prose authoritative and Table 3.1 the error). **Table 3.1 is authoritative.** Fix §3.3/§4.6 prose and HED item 9, not the table or the code. Note also that **`p` = "Predict" is defined nowhere** in either paper — an undocumented workload type. |
| **S-15** | `BTP-NEW` | **BTP Figure 5.1 calls the access rate a "cache access rate".** There is no cache anywhere in the system; access rate is `\|A\|/β`, the replication fraction. | Terminology fix in the write-up. Combined with **D-32** (it is a cumulative mean) and **D-08** (it is a monotone climb, not adaptation), all three claims BTP §6.2 draws from this figure fail. |
| **S-16** | `BTP-NEW` | **BTP §4.4 states multiplicative ε decay** `ε ← max(ε_min, ε·(1−γ_ε))`; the code does **subtractive** `ε -= gamma_decay`. The constant's name (`gamma_decay`) matches neither. | Pick one, implement it, rename the constant (B9). |
| **S-17** | `BTP-NEW` | **BTP §6.2's headline claim is false on the shipped data**: "SafeTail has lower latency than MinLoad, MinProp, and Random at all percentiles, with larger gaps at the 95th and 99th." See **D-10**: SafeTail loses to MinProp at every percentile, and to MinLoad-3 and Rand-3 at p95/p99. | The single most important thing to resolve before anything is submitted. See §13.2. |

**Rule R9 (provenance).** Before implementing anything sourced from HED or BTP, check §4.5. If the item carries `BTP-FIX`, the HED version is a known bug — do not restore it. If it carries `BTP-REGRESS` or `BTP-NEW`, neither document is authoritative and the design decision is *ours* to make and record in §13.1. If it carries `HED-BUG` with no BTP row, both documents are silent or wrong and the item needs a fresh design.

**Consequence for D-02.** §4.5 makes **D-02** worse than §4.1 states. The per-server traces show servers 3 and 4 are genuinely ~20× slower than servers 1, 2 and 5 (Speech: 0.433 s and 0.630 s versus 0.032 s, 0.032 s, 0.021 s), consistent with their having no GPU columns at all (**D-16**). Because all 15 wrappers load `models/server1/`, the fast GPU server's regressor is being used to predict the latency of two CPU-only servers that are an order of magnitude slower. The prediction error is not noise — it is a systematic ~20× underestimate on exactly the servers the scheduler most needs to avoid. This is very likely why the learned policy has no usable computation signal and why MinProp wins (**D-10**).

---

## 5. Missing implementations

| ID | Missing | Source | Why it matters |
|---|---|---|---|
| **M-01** | **Oracle / optimal baseline** — schedule to all β servers, record the min. No `oracle` string anywhere in the repo. | ST §V-B(i) | Without it the optimality-gap table (ST Table V: "within 8–30% of optimal while using 1–3 of 5 servers") cannot be reproduced. **Cheapest high-value addition.** |
| **M-02** | **SafeTail 1.0 as a baseline** — the τ-referenced policy. | This project's core task | §8 |
| **M-03** | **Target latency τ and the 5-case asymmetric reward.** Absent. Replaced by resource-headroom + `P(T)`. **Tail latency — the stated objective of all three documents — never appears in the reward.** | ST Def. 4.2/4.3, Eq. 4/5 | The single most consequential design gap. |
| **M-04** | **A redundancy cost term.** No term anywhere is decreasing in \|A\|. | — | Root cause of D-07/D-08. Any other fix will be undone without it. |
| **M-05** | **K-matched / budget-controlled evaluation.** Baselines at K=4, K=5; or subsample SafeTail to a matched mean K. | — | Root cause of D-09. |
| **M-06** | **Seeded repeats and error bars.** The `_1/_2/_3` suffixes in `results/baselines/` are **K variants, not seeds**. There is not a single repeated run in the repo. | — | No result is currently distinguishable from noise. |
| **M-07** | **DRL-Linear baseline.** | ST §V-B(v) | Medium priority. |
| **M-08** | **TLORA baseline** (Weibull fit, cost `ξ·mean + τ·δ_p`). 1.0 has a reference implementation at `baselines/safetail_v1/_spec_source/v1_baseline_tlora.py`. | ST §V-B(vi) | Medium; deferred (§9.5). |
| **M-09** | **Sigmoid multi-label output head + argmax fallback.** HED's signature mechanism; dropped in favour of 31-way discrete. Note HED states the output dimension as `nk` (user count × requests) where its own §II item 1 implies `mk` (device count × requests) — **S-09**. | HED abstract item 8 | Deferred — record as a deliberate design decision, not a bug. The dimension in HED is itself wrong, so any implementation is a redesign, not a restoration. |
| **M-10** | **M/M/1 waiting time, or removal of the claim.** ⚠️ **Read S-05 before touching this.** HED's formula is the correct M/M/1 queue-wait but is applied to a finite-capacity **loss** system that rejects rather than buffers. Implementing HED as written would be implementing a wrong model faithfully. | HED §III-D, BTP §3.5 | Preferred resolution: restate the system as `M/M/c/c` Erlang-B, report the rejection rate as a metric, and delete the M/M/1 claims from README + abstract + BTP §3.5. Building real queues is a separate project. |
| **M-11** | **Local fallback on the user device** when the cluster is saturated. Today: recursion (D-21). | ST §II | Medium. |
| **M-12** | **Job-length-proportional priority.** No job-size term in either reward. ⚠️ **S-08:** HED *states* this objective but never implements it in its own reward either. It is an unimplemented aspiration in the source document, not something the code lost. | HED abstract item 9 | Large jobs get no protection. Legitimate to defer; not legitimate to claim as implemented. Blocked on B5 anyway (no size variation exists — D-34). |
| **M-13** | **Batch/joint decision over the k requests in a chunk.** Chunks of 5 arrive together but are decided sequentially and independently. HED §III-A and BTP §3.6 agree with each other and both disagree with the code. | HED §III-A, BTP §3.6 | Medium. Code-vs-both-specs, so no provenance conflict — either implement or restate. |
| **M-14** | **`generate_testing_plots()`** is an empty `pass`; `export_training_data()` and `save_metrics_summary()` have fully commented-out bodies. | — | Testing-phase results are not plotted at all. |
| **M-15** | **Feature set for the latency regressors, per HED §IV-D** — live CPU / GPU / RAM utilisation of the target server at request time, plus concurrency level and task-specific input characteristics. The code uses none of these (static trace row + leaked target). **S-12:** HED describes this correctly and BTP §4.6 misdescribes it, so **HED is the better specification here.** | HED §IV-D | Fold into B1's retrain. Without live utilisation the "contention modelling" claim is only a lookup keyed by which task types are co-resident, not by how loaded the machine is. |

---

## 6. Fixtures applied so far — an honest ledger

**Nothing in `src/` has been modified.** `heterogenous/src/` is byte-identical to `SafeTail-2.0-main/src/`. What this session actually did:

| # | Action | Result |
|---|---|---|
| F-01 | Created `heterogenous/` as the new main project root | done |
| F-02 | Copied `src/ dataset/ models/ docs/ testing/ plotting/` + `README.md requirements.txt run_all_baselines.sh .gitignore documentation.pdf` verbatim | done |
| F-03 | Relocated the shipped `results/` to `results/reference_v0/` and froze it as the pre-fix evidence base (R6) | done |
| F-04 | Created `audit/` with the two live analysis PDFs, the superseded v1, a grep-able text extraction, and the ST + HED source papers | done |
| F-05 | Created `baselines/safetail_v1/_spec_source/` with 11 read-only SafeTail 1.0 files for porting reference | done |
| F-06 | Created empty `figures/` and `tools/` | done |
| F-07 | Re-verified the audit's 16 findings against the copied code; found **11 additional defects** (marked Δ in §4) | done |

**Everything else is a backlog.** The fixture backlog is §7 (core repairs) and §8 (baseline). Do not describe any §7 item as "applied" until its acceptance test in §11 passes.

---

## 7. Workstream B — core repairs

Ordered so that each step's acceptance test is meaningful given the previous ones. **Do not reorder without recording the reason in §13.2.**

### B0 — Environment truth (blocks everything)
Fixes **D-01**. Per §3.2. Deliverable: `tools/verify_regressors.py` that loads all 15 models and asserts non-None, plus a fixed UTF-8 `requirements.txt` with scikit-learn, seaborn, scipy added.
**Accept:** `python tools/verify_regressors.py` exits 0 and prints 15 model class names.

### B1 — Make heterogeneity real (D-02, D-02b, D-02c)
This is the highest-value repair in the project; §4.2 D-10 is probably downstream of it.

1. **Delete the 15 wrapper files** and replace them with a single parameterised module `src/regressors.py`:
   ```python
   class TracePredictor:
       def __init__(self, server_index: int, task: str):  # task ∈ {"detect","speech","predict"}
           ...loads models/server{server_index}/{task}_regressor_model.pkl
              and dataset/server{server_index}.csv
   ```
   This removes the copy-paste class of bug (D-02) *and* the `sys.modules` collision (D-02b) in one move, because there is now one module and the server index is an argument.
2. Rewrite `Server._load_predictors_from_regressor_folder` → `Server._load_predictors(self.server_index)`. **Remove every bare `except Exception: → None`.** A load failure must raise unless `constants.ALLOW_DEGRADED_PREDICTORS` is explicitly `True`, and when it is, it must emit `[SAFETAIL][DEGRADED][D-02c]` once per server.
3. In `_predict_using_letter`, the CSV fallback must key on `combined_str`, not on the bare `letter`. If the contention row is absent, raise — do not silently return the contention-free number.
4. **Do not delete `src/server{1..5}_regressor/`** in the same commit as the rewrite. Move them to `src/_legacy_regressor_wrappers/` with a `DEPRECATED.md`, delete in a later commit. Reason: the shipped `reference_v0` results were produced against them and we may need to reproduce that path.

**Accept:** a test asserts that for a fixed `combined_str`, `Server(i).compute_request_time(...)` returns **five materially different computation delays** for i∈{1..5} — specifically that servers 1 and 3 differ (their CSVs differ, D-15 says 1 and 2 legitimately will not).

### B2 — Fix the RL training signal (D-04, D-05, D-06)
1. **Snapshot the state at action time.** In `process_step`, immediately after the action is chosen, compute `s_t = agent.request_to_state_array(request)` and store the **array**, not the object. Same for `s_{t+1}` — build it after `schedule_request` returns, from the post-action request.
2. Store per-step rewards. Keep the episodic reward as a *separate* terminal bonus (add it to the last transition of the episode, or as an explicit `R_ep/N` broadcast) — but **do not overwrite** `r_t`.
3. `next_state` must be the genuine successor state.

**Accept:** `tools/audit_replay.py` samples 200 stored transitions and asserts (a) `state` and `next_state` arrays are not element-wise identical for >95% of them, (b) no stored state contains a value that equals a realised `total_processing_delay` from the same transition, (c) `len(set(rewards_in_episode)) > 1`.

### B3 — Price redundancy (D-07, M-04, S-02)
⚠️ **Read S-02 first.** The missing redundancy cost is not an oversight — it is the collateral damage of a correct fix. HED's step reward `Σ_{i∈d}(l_i − l_total/m) − W_step` was *sign-inverted for load balancing* (maximising it selects the **most**-loaded servers), and BTP correctly replaced the load term with a headroom product that rewards idle servers. In doing so it dropped `d` and `−W_step`, the only two `|A|`-sensitive quantities. **Do not restore HED's term** (R9) — add a new explicit cost on top of BTP's corrected product.

Add an explicit cost term. Recommended form, parameterised so it can be swept:
```
R_step = (1/|A|)·Σ_{i∈A} log(1+headroom_i)  −  c_red · (|A| − 1)/(β − 1)
```
Two changes: the mean is over `|A|` (not a constant 5), and `c_red ≥ 0` penalises redundancy. `c_red` goes in `constants.py` with a documented default and a sweep list.

**Accept:** with `c_red` large, the learned mean K must fall below the K it reaches at `c_red = 0`. Report the K-vs-`c_red` curve — this is a **publishable ablation**, not just a fix.

### B4 — Introduce τ and a tail-referenced objective (M-03)
Port SafeTail 1.0's mechanism (§14.1 gives the exact formulas). Add as an **additive, switchable** term so B3's reward remains reproducible:
`constants.REWARD_MODE ∈ {"headroom", "tau", "headroom+tau"}`.
τ per request type, derived from the reference latency distribution (document the derivation — do not hardcode 0.05 without justification, given D-11/D-12 show the metric's composition is pathological).

### B5 — Make the metric mean something (D-11, D-12, D-23, S-04, S-07)
1. **Decide and document what `total_latency` is.** ⚠️ **Four incompatible definitions are in play** (S-04): BTP §3.4's display equation (omits `T_exec`), BTP's own bullet list (includes it), BTP Figure 3.2 (includes it twice over), and the code (`comp+prop+trans`, omits queueing — D-11). Waiting time separately has three definitions (S-07): HED's analytical M/M/1, HED's `/proc` measurement, and the code's wall clock. Resolve **all** of them, not just the code's. Recommendation: log **both** `service_latency` (comp+prop+trans) and `end_to_end_latency` (+ queueing), and plot the one the reward optimises. Never plot a quantity the reward ignores.
2. **Replace wall-clock `queue_waiting_time` with a simulated quantity** (D-23). Until this is done, results are machine-dependent and SafeTail is handicapped by its own TF forward pass.
3. **Make transmission delay a function of `message_size` and `bandwidth`** (D-12, D-34), and give requests real size variation. Until then ~45% of the metric is noise no policy can move.

**Accept:** re-run the shipped configuration; assert `transmission_delay` takes >5 distinct values and correlates with `message_size` (|ρ| > 0.5).

### B6 — Budget-controlled evaluation (D-09, M-05)
Add `minload_4/5`, `minprop_4/5`, `rand_4/5` to the mode table, **and** an explicit `--match-k` evaluation that subsamples SafeTail decisions to a target mean K.
**Accept:** the comparison table reports every policy at a stated K, and SafeTail's mean K is printed alongside it.

### B7 — Seeds and error bars (M-06)
`constants.SEED`, threaded to `random`, `np.random`, and `tf.random`. Run each configuration ≥3 seeds. All plots get error bars or a shaded IQR band. **Rename the `_1/_2/_3` suffix convention** — it currently means K, and a reader will assume seeds.

### B8 — Structural correctness (D-16 … D-21, D-17)
* D-17: one-line index fix, plus a test that forces a reward failure and asserts the right slot is zeroed.
* D-16: define an explicit "no GPU" policy. Recommended: a `has_gpu` flag per server; for CPU-only servers **drop** the GPU factors from the product and renormalise, rather than letting `(1-0)(1-0)=1` inflate them.
* D-19: stop mutating `request.combination`. Add a separate `request.contention_str`. This also fixes the 192-value `request_type` column.
* D-18: reuse the phase-2 delay draw in phase 7 (single sample per request per server, taken once).
* D-20: restore the deadline-conditional `request_x_done` increments.
* D-21: depth-limited retry with exponential backoff and a hard drop + counter.

### B9 — Hygiene (D-13, D-14, D-27 … D-34)
Delete or resurrect the orphans (D-29), delete dead constants (D-30), fix the docstring (D-27), rewrite `README.md` to remove the M/M/1 claims (D-13, M-10), fix `run_all_baselines.sh` (D-14), replace `np.load(allow_pickle=True)` with a safe codec (D-31), and correct the request-count and state-dimension claims (D-33, D-35).

### B12 — Specification errata (the S-register)

Separate from code repair, because it is written work and can run in parallel with everything. Produce `audit/ERRATA.md`: a corrections list against HED and BTP, each entry citing the section, quoting the claim, and stating the correction with its evidence. This is what makes the next paper defensible, and it is also the artefact to hand the advisor.

Minimum contents, one entry per S-row:

| Entry | Correction |
|---|---|
| S-01 | **No correction — record as a strength.** BTP's `P(T)` is a correct replacement for HED's malformed ω. Say so explicitly in the errata and in the next paper's related-work framing; it is the clearest thing BTP got right, and citing it makes the rest of the errata read as analysis rather than attack. **Do not "restore" HED's version** (R9). |
| S-03 | Reward range is `[0, log 2]`, not `[0,1]`. One statement, cited everywhere. |
| S-04 | One latency equation. BTP §3.4's display equation is missing `T_exec`; reconcile with Figure 3.2 and the code. |
| S-05, S-06 | Either delete the M/M/1 + Poisson claims (README, BTP §3.5, abstract, HED §III-D) or implement them. Do not leave the abstract asserting a model the code does not contain. |
| S-10 | Rewrite BTP §4.2 to match §3.2 and the code: the state is the flattened request including per-server hardware vectors; variable length comes from **hardware heterogeneity**, not temporal accumulation. Remove the `(β+1)` claim and the dead `nS`. |
| S-11 | Rewrite BTP §5.2's baseline description: the K variants are fixed-K **redundant dispatchers**, not candidate pre-filters feeding a shared chooser. |
| S-12 | Rewrite BTP §4.6 to describe the features that exist — and after B1, the features that should exist per HED §IV-D. Disclose the removed target leak. |
| **S-14** | **Fix the type mapping in BTP §3.3 and §4.6 prose and in HED item 9 — not in Table 3.1 and not in the code.** The dataset is decisive: `s = Speech`, `d = Detect`, `p = Predict` on all five servers. Also define what `p` ("Predict") actually is — no document says. |
| S-08, S-09 | Mark HED's job-length-priority objective (item 9) and its truncated item 10 as **unimplemented in HED itself**. When the next paper cites HED, cite it for intent only, and do not present either as a capability the codebase lost. Also correct HED's RL output dimension `nk` → `mk`. |
| S-13 | State plainly which model family each shipped regressor is (6 Linear, 8 RandomForest/DecisionTree, 1 GradientBoosting) and stop calling them MLPs (BTP §2.6, §4.6, abstract; HED §IV-D). **Before rewriting, try HED's cited repo `github.com/amardeep786/Regressors`** — HED may be truthful about *its* models even though BTP is not truthful about this repo's, and if those MLPs exist they are the cheapest correct input to B1. Record the outcome either way. |
| S-15 | "Cache access rate" → "access rate" / replication fraction. There is no cache. Then re-derive every §6.2 claim that rests on that figure (D-08, D-32). |
| S-16 | State one ε schedule; rename `gamma_decay` to match it. |
| S-17 | Retract or re-establish BTP §6.2's headline claim, on post-fix data, at matched replication budget. See §13.2. |

**Accept:** `audit/ERRATA.md` exists, every S-row has an entry, and every entry cites a document section and a code location or dataset fact.

### B10 — Remove the rigged plot offset (D-03)
Delete the `v=0.005` MinProp offset from both notebook cells. **Do not do this quietly** — record it in the changelog with the before/after percentile table from §4.2, because it changes published numbers.

### B11 — Disposition of every remaining ID

Every defect and every missing feature must have a disposition. "Not mentioned" is not a disposition. This table closes the loop on the IDs not already assigned above.

| ID | Disposition | Where |
|---|---|---|
| D-08 (over-replication) | **Fixed indirectly** by B3 (`c_red`). It is the *symptom*; D-07 is the cause. Its acceptance test is B3's K-vs-`c_red` curve. | B3 |
| D-25 (target-leaking regressor feature) | **Fix in B1.** When `regressors.py` is written, drop `total_processing_time` from `_build_features` and **retrain** all 15 models. Report held-out R² before/after — the "prediction" is currently partly a ground-truth lookup, so the honest R² will fall. That fall is the finding, not a regression. **Design the replacement feature set from HED §IV-D (M-15), not from BTP §4.6** — S-12 shows BTP misdescribes what the code actually does, while HED describes a sound design (live CPU/GPU/RAM utilisation + concurrency level + input characteristics). | B1 |
| D-26 (no MLP regressors) | **Two-part.** (a) Documentation: correct README/BTP §2.6, §4.6 — the shipped models are 6 LinearRegression / 8 RandomForest / 1 GradientBoosting. (b) Optional: train actual MLPs alongside B1's retrain and pick by held-out R². Part (a) is mandatory in B9; part (b) is optional. | B9 (+B1) |
| D-22 (arrivals uniform, not Poisson) | **Decide, don't drift.** Either switch `sender_bursts` to exponential inter-arrival times, or delete every Poisson/M/M/1 claim from README, abstract and BTP §3.5. Deleting is nearly free; switching changes every latency number. Bundle with M-10. | B9 or B5 |
| D-24 (no queues anywhere) | **Restate, don't fake.** The servers are a `c=4` Erlang-B **loss** system, not M/M/1. Recommended: say so plainly and report the drop/rejection rate as a first-class metric (it is currently invisible). Implementing real queues is a separate project. Bundle with M-10, D-22. | B9 |
| D-28 (`assign_request` broken 3-of-7 unpack) | **Delete the method.** It is unreachable dead code; keeping a broken orphan invites someone to "fix" it into the live path. | B9 (with D-29) |
| M-11 (local fallback on user device) | **Deferred.** B8's depth-limited retry + explicit drop counter is the interim behaviour. Record the drop rate; if it is non-trivial, promote M-11. | deferred |
| M-12 (job-length-proportional priority) | **Deferred.** Meaningless until B5 gives requests real size variation. Revisit after B5. | deferred |
| M-13 (joint decision over a chunk of k) | **Deferred.** Sequential-independent decisions are the current semantics; state it explicitly in the paper rather than letting HED §III-A imply otherwise. | deferred |
| M-14 (`generate_testing_plots` is `pass`; two commented-out bodies) | **Fix in B9.** Testing-phase results are currently not plotted at all, so the train/test split produces no reportable evidence. Implement, or delete the stub and the split's plotting claim. | B9 |
| M-07 (DRL-Linear), M-08 (TLORA), M-09 (sigmoid head), M-10 (M/M/1) | **Out of scope for this phase** (§9.5). M-09 should be recorded in the paper as a *deliberate design choice* (31-way discrete over sigmoid multi-label), not left looking like an omission. | §9.5 |

---

## 8. Workstream C — SafeTail 1.0 as an isolated baseline

This is the primary deliverable. **It is a port, not a bridge** (§2.2).

### 8.1 What "SafeTail 1.0 baseline" means here

Not "run the old repo". It means: **the SafeTail 1.0 *policy* — its state abstraction, its network, its τ-referenced 5-case reward — driving the SafeTail 2.0 *environment*,** so that latencies are produced by the same servers, the same traces and the same admission rules as every other policy in the comparison. Anything less is not a controlled comparison.

The 1.0 algorithm specification is §14.1. The frozen source is `baselines/safetail_v1/_spec_source/` (read, never import).

### 8.2 Target layout

```
baselines/                              ← delete this whole tree ⇒ repo still works (R3, gate G4)
├── README.md                           what this is, how to run, how to delete
├── __init__.py                         empty
├── register.py                         the ONLY file that touches src.policy_registry
├── run_baseline.py                     CLI entry point
├── common/
│   ├── env_adapter.py                  read-only view of controller/server state
│   └── metrics.py                      percentile/CDF helpers shared by baselines
├── safetail_v1/
│   ├── __init__.py
│   ├── config_v1.py                    β, α, τ, lr, γ, ε schedule — 1.0's own hyperparameters
│   ├── state_v1.py                     2.0 Request  →  1.0 flat state vector
│   ├── model_v1.py                     1.0's exact network (sigmoid ×2 + BN, softmax head, CCE)
│   ├── reward_v1.py                    the τ-referenced 5-case reward
│   ├── policy_v1.py                    SafeTailV1Policy — implements the Policy protocol
│   ├── train_v1.py                     ε-greedy loop + replay, 1.0 semantics
│   └── _spec_source/                   frozen 1.0 files, READ-ONLY, never imported
└── oracle/                             M-01 lives here too (same seam, trivially)
    └── policy_oracle.py
```

### 8.3 The seam — exactly two symbols in `src/`

Create **`src/policy_registry.py`** (~50 lines, no third-party imports, no knowledge of any baseline):

```python
"""[SAFETAIL][SEAM] Policy plug-in point. Core infrastructure.
This module must remain functional and importable with baselines/ absent."""

from typing import Protocol, Sequence, Callable, Dict

class Policy(Protocol):
    name: str
    def select(self, ctx) -> Sequence[int]: ...      # returns 0-based server indices
    def observe(self, ctx, action, reward) -> None: ...   # no-op for stateless policies
    def finish_episode(self, episode_index: int) -> None: ...

_REGISTRY: Dict[str, Callable[[], "Policy"]] = {}

def register(name: str, factory): ...
def get(name: str): ...        # raises KeyError with the list of known names
def available() -> list: ...
```

And in `src/controller.py`, replace the `BASELINE_MODE` if-chain (`:606-641`) with:

```python
# [SAFETAIL][SEAM][M-02] External policies register into policy_registry.
# Native modes below are unconditional and survive deletion of baselines/.
if self.policy is not None:                 # set once in __init__ (see below)
    action_subset = self.policy.select(ctx)
    action_index  = subset_to_index(action_subset)
elif self.BASELINE_MODE == "safetail":
    ...existing DQN path, unchanged...
elif ...native heuristics, unchanged...
```

and in `Controller.__init__`:

```python
# [SAFETAIL][SEAM] Optional external policy. Absent baselines/ ⇒ self.policy is None.
self.policy = None
if constants.POLICY not in (None, "", "native"):
    from policy_registry import get           # stdlib-only module, always present
    self.policy = get(constants.POLICY)()     # KeyError here is a loud, correct failure
```

`constants.POLICY = os.environ.get("POLICY", "native")`.

**Who registers?** `baselines/register.py`, imported by `baselines/run_baseline.py` *before* it calls `main()`. `src/` never imports `baselines/`. That single direction of dependency is what makes R3 hold.

### 8.4 `ctx` — the read-only environment view

`ctx` is a frozen dataclass built in `process_step` and passed to `policy.select`. It exposes exactly what a scheduling policy is entitled to see **before** acting:

| Field | Type | Source |
|---|---|---|
| `request_id` | int | `request.request_id` |
| `request_type` | str | the **original** type letter ∈ {`s`,`d`,`p`} = {Speech, Detect, Predict} (S-14) — see D-19; this is why B8 must land first |
| `deadline` | (D1, D2) ms | `request.deadline` |
| `message_size`, `bandwidth` | float | `request` (constants today; real after B5) |
| `free_slots[β]` | int, −1 = full | `find_free_servers()` |
| `est_delay[β]` | ms | phase-2 `compute_request_time` — the **one** draw (D-18) |
| `est_components[β]` | (comp, prop, trans) | same call |
| `server_static[β]` | dict | RAM/cores/GPU capacities from `server_dicts` |
| `server_dynamic[β]` | dict | current utilisation from `server_dicts` |
| `arrival_time` | float ms | `request.arrival_time` |

**Nothing post-action is in `ctx`.** That is deliberate: it makes the D-04 class of leak structurally impossible for any policy that uses the seam.

### 8.5 Porting SafeTail 1.0 — the four adaptation decisions

The 1.0 state is homogeneous; the 2.0 environment is not. Four choices must be made explicitly, recorded in `config_v1.py`, and stated in the paper. **These are the honest-reporting crux of the whole baseline.**

| # | 1.0 quantity | 1.0 meaning | 2.0 mapping (recommended) |
|---|---|---|---|
| A-1 | `LOAD[i]` | integer count of concurrent users on device i, 1..max_load | `len(server_list[i].active_requests)` clipped to `[1, MAX_CONCURRENT_REQUESTS]`. **Never feed the raw heterogeneous vector** — 1.0 by construction cannot see it, and giving it more information than the original had would flatter the baseline. |
| A-2 | `PROPOGATION[i]` | `random.choice(ping_data[LOAD[i]-1])` | `ctx.est_components[i].prop` — the same single draw the environment used (D-18). |
| A-3 | `MESSAGE_SIZE`, `BANDWIDTH` | per-request file size, link rate | `ctx.message_size`, `ctx.bandwidth`. Constant today (D-34); becomes meaningful after B5. |
| A-4 | `RESOLUTION` | YOLO input resolution | **Omit.** 2.0 has no resolution attribute; the 1.0 `instance` and `noise` configs also omit it. Set `nS = 2β + 2` (the `instance`/`noise` shape), not `2β + 3`. |

**τ (`median_computation_delay`):** 1.0 used a per-task constant (yolo 1.2 s, instance 2.5 s, noise 0.3 s); 2.0's `constants.py` carries `0.05`. Do **not** inherit either blindly. Note the request types differ by more than an order of magnitude in service time (Speech 0.032 s vs Detect 0.0026 s on server 1, and 0.63 s vs 0.055 s on server 4 — S-14, §4.5), so a single global τ would be meaningless. Derive τ **per request type** from `results/reference_v0` service-latency medians, document the derivation in `config_v1.py`, and record it in §13.1. Getting τ wrong makes the 1.0 baseline look arbitrarily good or bad — this is the single most gameable knob in the comparison.

**Latency the policy is graded on:** the 1.0 reward uses `min` over the selected subset. 2.0 also takes a min. These agree — keep it, and use the *same* latency definition B5 settles on.

### 8.6 Faithfulness register — what is kept, what is changed, and why

Every deviation from 1.0 must appear in this table in `baselines/safetail_v1/README.md`. A reviewer will ask.

| 1.0 element | Port | Rationale |
|---|---|---|
| 5-case τ reward (Eq. 5) | **keep exactly** — §14.1 | It is the baseline's identity |
| `softmax` head + `categorical_crossentropy` | **keep** | Faithful even though it is wrong for Q-regression; changing it would make this "SafeTail 1.5", not SafeTail 1.0. Note the oddity in the paper. |
| 2 hidden layers, sigmoid, BN | **keep** | ST §IV |
| `2^β − 1` subset action space | keep (β = 5 ⇒ 31) | matches 2.0 |
| ε-greedy, `ε -= gamma_decay` per replay | keep | ST Eq. 3 |
| replay `deque(maxlen=2500)`, batch 128 | keep | 1.0 values |
| `experience_replay` called **inside** `reward()` | **change** — call it from `finish_episode`/step boundary | 1.0's placement is a structural accident; keeping it would couple the baseline to 2.0's reward call sites |
| `reward()` returns `None` when `abs(λ) ≥ 1000` | **fix** — return 0.0 and count the event | Latent 1.0 bug: `None` reaches the replay buffer. Record as `V1-BUG-01`. |
| `nS = 2β + 3` (yolo) | **change to `2β + 2`** | A-4 |
| Task-specific MLP latency regressors | **drop** | 2.0's environment provides latency; the baseline must not carry its own simulator or the comparison is meaningless |
| Homogeneous `LOAD` abstraction | **keep** | A-1 — this *is* the thing being tested |

### 8.7 Oracle (M-01) — same seam, ~30 lines

`baselines/oracle/policy_oracle.py`: `select()` returns all β indices; the harness records `min(est_delay)`. Gives the optimality gap. Do this **first** — it is the cheapest way to prove the seam works end to end before the 1.0 port lands.

### 8.8 Deletability — the acceptance test (R3)

```bash
git stash -u                      # or: cp -r baselines /tmp/
rm -rf baselines/
python -c "import sys; sys.path.insert(0,'src'); import policy_registry, controller, main"
POLICY=native BASELINE_MODE=safetail  python src/main.py --smoke
POLICY=native BASELINE_MODE=minload_2 python src/main.py --smoke
POLICY=safetail_v1 python src/main.py --smoke   # MUST fail loudly with KeyError listing known names
```
All four behaviours must hold. This is gate **G4**.

---

## 9. Workstream D — runs and graphs

**Scope for this phase, per the user: SafeTail 1.0 baseline only.** Oracle is included because it is nearly free and unlocks the optimality gap. TLORA / DRL-Linear are explicitly deferred.

### 9.1 Run matrix

| Run | POLICY | K | Seeds | Purpose |
|---|---|---|---|---|
| R-0 | `native` / `safetail` | free | 3 | Post-fix SafeTail 2.0. The new reference. |
| R-1 | `safetail_v1` | free | 3 | The deliverable baseline. |
| R-2 | `oracle` | 5 | 3 | Optimality gap (M-01). |
| R-3 | `native` / `minload_{1..5}` | 1–5 | 3 each | Budget-controlled (B6) |
| R-4 | `native` / `minprop_{1..5}` | 1–5 | 3 each | Budget-controlled |
| R-5 | `native` / `rand_{1..5}` | 1–5 | 3 each | Budget-controlled |
| R-6 | `native` / `safetail`, `--match-k` | matched | 3 | The confound-free headline (D-09) |

Every run writes to `results/<run_id>/` where `run_id = {policy}_{k}_{seed}_{gitsha7}`. **Never overwrite `results/reference_v0/`.**

Each run directory must contain a `manifest.json`: git SHA, dirty flag, seed, full `constants` dump, package versions, host, wall-clock start/end, and the SHA-256 of every input CSV/pkl. A result without a manifest is not a result.

### 9.2 Figures (land in `figures/`)

| Fig | Content | Guards |
|---|---|---|
| **F1** | Tail-latency bars: p50/p90/p95/p99, SafeTail-2.0 vs SafeTail-1.0 vs Oracle. Error bars over 3 seeds. | No per-mode offsets (D-03). State which latency definition (D-11). |
| **F2** | Latency **CDF/CCDF**, log-y, tail zoom ≥p90. More honest than bars at the tail — bars hide distribution shape. |
| **F3** | Latency **vs replication budget K** — every policy plotted at its actual mean K, SafeTail as a point with its measured K. This is the figure that answers D-09 directly. |
| **F4** | Optimality gap: (policy p95 − oracle p95)/oracle p95, per policy. ST Table V analogue. |
| **F5** | Learning curves: episodic reward, ε, mean K, p95 latency vs episode — SafeTail-2.0 vs SafeTail-1.0 on shared axes. **Per-episode, not cumulative** (D-32). |
| **F6** | Per-request-type panel (s / p / d), p95, both policies. Filter on the true type field, not the contention string (D-19). |
| **F7** | Latency decomposition stacked bars (comp / prop / trans / queue) per policy. **Publish this** — it makes D-12 visible and pre-empts the reviewer question "what is actually being optimised?". |
| **F8** | Ablation: mean K vs `c_red` (B3). Turns a bug fix into a contribution. |

**Plotting rules:** one script `tools/make_figures.py`, no notebooks in the publication path (notebooks hid D-03 for months). Every figure function takes an explicit list of run directories and writes a `.csv` of the plotted numbers next to the `.png`/`.pdf`. If a number is in a figure, it must be in a CSV.

### 9.3 The comparison table

`figures/table_main.csv` — one row per (policy, K, seed), columns: p50, p90, p95, p99, mean, mean K, deadline-satisfaction ω, requests completed, requests dropped. The aggregate table reports median across seeds with IQR.

### 9.4 Smoke mode
Add `--smoke` to `src/main.py`: 20 episodes, ~300 requests, no plots, deterministic seed. Every gate in §11 runs against smoke. Full runs only after all gates pass.

### 9.5 Explicitly deferred
TLORA (M-08), DRL-Linear (M-07), sigmoid head (M-09), M/M/1 (M-10), Tail-Learning (see §13.3). Do not start these without a new instruction.

---

## 10. Code quality, instrumentation and debug tags

### 10.1 The tag taxonomy

Every log line and every non-obvious code block carries a bracketed tag. This is the debuggability contract the user asked for: **given a symptom, `grep` finds the code.**

```
[SAFETAIL][<COMPONENT>][<KIND>][<ID>] message
```

| Field | Allowed values |
|---|---|
| `COMPONENT` | `MAIN` `SENDER` `RECEIVER` `CONTROLLER` `SERVER` `AGENT` `REGRESSOR` `REWARD` `REPLAY` `SEAM` `POLICY` `PLOT` `AUDIT` |
| `KIND` | `TRACE` (per-request, off by default) · `STEP` · `EPISODE` · `STATE` · `ACTION` · `REWARD` · `SCHED` · `DEGRADED` · `INVARIANT` · `FIX` · `SEAM` · `PERF` |
| `ID` | the defect/feature ID: `D-04`, `M-02`, `V1-BUG-01`, … Omit only for routine tracing. |

Examples:
```python
# [SAFETAIL][CONTROLLER][FIX][D-04] snapshot state at action time; never store the live Request.
log.debug("[SAFETAIL][REPLAY][STATE][D-04] s_t dim=%d checksum=%s", s.size, _ck(s))
raise RuntimeError("[SAFETAIL][REGRESSOR][DEGRADED][D-02c] server=%d task=%s load failed" % (i, t))
```

**Rule:** when you fix `D-xx`, the tag `[D-xx]` must appear at the fix site **and** in the regression test's name (`test_D04_state_snapshot_not_live_object`). A future agent grepping `D-04` finds the defect description here, the fix, and the test. That is the whole point.

### 10.2 Invariants — assert them, don't hope

Add `src/invariants.py` with cheap runtime checks, on by default in smoke mode and behind `constants.CHECK_INVARIANTS` in full runs:

| Invariant | Guards |
|---|---|
| `0 < len(action_subset) <= β` and all indices in range | action space |
| `est_delay` used in the reward is the same object as the one shown to the policy | D-18 |
| stored `state` array contains no value equal to a realised delay of the same step | D-04 |
| `state is not next_state` and not element-wise equal | D-05 |
| step rewards within an episode are not all identical | D-06 |
| `np.isfinite(reward)` for every server | D-16, D-17 |
| `rewards[i]` written on both success and failure path with the same `i` | D-17 |
| `total_latency` recomputes from its logged components to <1e-6 | D-11 |
| `request.type_letter` ∈ {s,p,d} at log time | D-19 |
| every predictor is non-None unless `ALLOW_DEGRADED_PREDICTORS` | D-02c |

Each assertion message carries its tag and defect ID.

### 10.3 Error handling policy

The current code has ~40 `try/except Exception` blocks that print and continue. That is why D-02c can hide a total collapse of the regressor layer. New policy:

* **Never** `except Exception: pass`.
* **Never** swallow an exception that changes a number. Degradation must be either (a) impossible, or (b) explicitly enabled by a constant *and* logged with `[DEGRADED]` *and* counted in the run manifest.
* Existing swallow-sites: keep the `except` only where it protects the traffic generator's hot loop (`main.request_factory`), and even there, **count** the fallbacks and put the count in the manifest.
* A run whose manifest reports any `[DEGRADED]` count > 0 is **not publishable**. Make `tools/check_manifest.py` enforce it.

### 10.4 Style and structure
Python ≥3.10 typing on every new/edited function; `ruff` + `black` (line length 100); no new module over ~400 lines — `controller.py` at 1,155 is already the main obstacle to review. When touching it substantially, split reward computation into `src/rewards.py`. Docstrings state the formula **actually implemented** (D-27 exists because one did not).

### 10.5 Configuration discipline
No magic numbers in logic. Anything that changes a result lives in `constants.py` with (a) a comment naming its source (ST/HED/BTP section, or "chosen, see plan §x"), (b) its unit, (c) whether it is swept. `print_constants()` already dumps the module at startup — it must go into `manifest.json`, not just stdout.

### 10.6 Changelog
`CHANGELOG.md`, newest first, one row per merged change:
`date · defect/feature ID · files · one-line what · one-line why · which gate proves it · does it change published numbers (Y/N)`.
Any `Y` row must link to the before/after table.

---

## 11. Audit gates

A gate is a script in `tools/` that exits non-zero on failure. **No workstream is "done" until its gate is green.** Run all gates before every full run.

| Gate | Script | Checks | Blocks |
|---|---|---|---|
| **G0** | `verify_env.py` | 15 regressors unpickle; package versions match `requirements.txt`; numpy is 1.x | everything |
| **G1** | `verify_heterogeneity.py` | the 5 servers give materially different computation delays for the same contention string; server 1 vs 3 differ; reports the 1≡2 duplicate (D-15) as a **warning with an explicit acknowledgement**, not a failure | B1 |
| **G2** | `audit_replay.py` | the D-04/D-05/D-06 assertions of §7-B2 over ≥200 sampled transitions | B2 |
| **G3** | `audit_reward.py` | reward is **not** monotone in \|A\| (sweep \|A\|=1..5 on a fixed state, assert non-monotonicity once `c_red>0`); finite for all servers incl. CPU-only (D-16); failure path writes the right slot (D-17) | B3, B8 |
| **G4** | `verify_isolation.sh` | (a) `rm -rf baselines/` ⇒ core imports and both native modes still run; (b) `grep -rn "baselines" src/` returns nothing; (c) `grep -rn "_spec_source" baselines/ --include=*.py` finds no `import`; (d) no `sys.path` manipulation pointing outside `heterogenous/` | C |
| **G5** | `check_manifest.py` | every `results/*/manifest.json` present, complete, zero `[DEGRADED]` counts, git clean | D |
| **G6** | `verify_figures.py` | every figure has its companion CSV; no per-mode additive offsets anywhere in `tools/make_figures.py` (grep for `+ 0.005` and friends) (D-03) | D |
| **G7** | `verify_types.py` | asserts the request-type mapping against the **dataset**, not against prose: for every `dataset/server{i}.csv`, the single-letter rows satisfy `s→Speech`, `d→Detect`, `p→Predict`; asserts the regressor for letter `x` looks up the matching script name; asserts `ORIGINAL_DEADLINES` pairs `s`→(100,400) and `d`,`p`→(30,200). Guards **S-14** — the one defect where a plausible-sounding document sentence would send someone to "fix" correct code | B1, B8, B12 |

**Regression suite:** `pytest tools/tests/` — one test per fixed defect, named `test_<ID>_<slug>`. Target: every `D-xx` marked fixed in `CHANGELOG.md` has a test.

---

## 12. Multi-agent orchestration

### 12.1 Shape of the work

The workstreams have a real dependency graph. Parallelise only where it is genuinely parallel.

```
B0 ──▶ B1 ──▶ B2 ──▶ B3 ──▶ B4
        │      │      │
        │      └──────┴──▶ B6 ──▶ B7 ──▶ [RUNS] ──▶ [FIGURES]
        │                          ▲
        ├──▶ B5 ───────────────────┤
        ├──▶ B8 ───────────────────┤
        ├──▶ B9  (anytime)         │
        └──▶ B12 (anytime, docs)   │
                                   │
C-seam ──▶ C-oracle ──▶ C-v1 ──────┘
(needs B8 for ctx.request_type)
```

**Safe to run in parallel:** `B5`, `B8`, `B9`, `B12`, and the `C-seam` work — they touch disjoint files. **Never parallel:** anything else touching `controller.py`, which is the contention point for B2/B3/B6 and the seam.

**B12 is the odd one out** — it writes prose, not code, and it can start immediately, before B0. But its S-17 entry (retract or re-establish the headline claim) cannot be closed until the runs finish. Start it early, close it last.

### 12.2 Agent roles

| Role | Owns | Files it may write | Must not touch |
|---|---|---|---|
| **Orchestrator** | plan, sequencing, merge, changelog | `plan.md`, `CHANGELOG.md` | source |
| **Env agent** | B0, G0 | `requirements.txt`, `tools/verify_env.py` | `src/` logic |
| **Core-RL agent** | B2, B3, B4, G2, G3 | `controller.py`, `agent.py`, `rewards.py` | `baselines/`, `servers.py` |
| **Sim agent** | B1, B5, B8, G1 | `servers.py`, `user.py`, `regressors.py`, `sender_bursts.py` | `agent.py`, `baselines/` |
| **Seam+baseline agent** | C (all), G4 | `src/policy_registry.py`, the seam block in `controller.py`, all of `baselines/` | everything else in `src/` |
| **Eval agent** | B6, B7, D, G5, G6 | `tools/`, `figures/` | `src/` logic |
| **Errata agent** | B12, G7 | `audit/ERRATA.md`, `tools/verify_types.py` | all source; it *reads* code to cite it, never edits |
| **Auditor** | independent verification | `tools/tests/`, `audit/` notes | nothing it reviews |

### 12.3 Hand-off protocol

Each agent returns a **hand-off block**, and nothing else is trusted:

```markdown
## HANDOFF — <role> — <workstream ID>
STATUS: complete | blocked | partial
IDS ADDRESSED: D-04, D-05
FILES CHANGED: src/controller.py (lines 700-760), tools/audit_replay.py (new)
GATES: G2 PASS (log: tools/out/g2_20260902.txt)
INVARIANTS ADDED: state-is-not-next-state, no-outcome-in-state
NUMBERS THAT MOVED: mean K 3.55 → 2.81 on smoke; p95 75.7 → 61.2 ms  [before/after CSV: ...]
ASSUMPTIONS MADE: <each one, or "none">
DISCOVERED, NOT FIXED: <new defect IDs proposed for §4>
NEXT AGENT NEEDS: <exact preconditions>
```

`NUMBERS THAT MOVED` is mandatory. A change that moves no number and passes no gate did nothing.

### 12.4 Rules for the orchestrator

1. **Give each sub-agent this file's §1, §2, §4 (including §4.5) and its own workstream section — not a paraphrase.** Sub-agents start cold; a summary loses the file:line evidence that makes the defects actionable. **§4.5 is not optional context** — without it an agent reading HED will "restore" mechanisms BTP correctly removed, and an agent reading BTP §3.3 will "fix" a correct type mapping into a wrong one (S-14).
2. **One writer per file.** If two workstreams need `controller.py`, serialise them.
3. **Auditor never fixes.** It only verifies and files new `D-xx`. An agent that grades its own work grades generously.
4. **Re-verify before trusting.** This plan's §4 was itself produced by re-verifying a 2-week-old audit and found 11 new defects. Apply the same scepticism to §4.
5. **Stop on contradiction (R7).** If the code disagrees with this plan, stop and report; do not reconcile silently.
6. **Number-moving changes need the before/after table** in the hand-off, or they do not merge.
7. **Cap the blast radius.** No single change should touch more than one workstream's files. If it must, split it.

### 12.5 Suggested first three tasks

1. **Env agent → B0.** Nothing else is trustworthy until the regressors demonstrably load. Expect this to change results.
2. **Sim agent → B1.** D-02 is likely the mechanical explanation for D-10. Fixing it may move every number in the paper. Do it before anything is written up.
3. **Seam+baseline agent → C-seam + C-oracle.** ~80 lines, proves R3/G4 early, and M-01 unlocks the optimality-gap figure independently of the 1.0 port.

Do **not** start the SafeTail 1.0 port before B8 lands — `ctx.request_type` depends on D-19.

---

## 13. Risks, decisions, open questions

### 13.1 Decisions that must be recorded before the port
| # | Decision | Default if unanswered |
|---|---|---|
| 1 | τ per request type — derivation source | median service latency per type from `reference_v0`, documented |
| 2 | Latency definition for all figures | `service_latency` (comp+prop+trans); `end_to_end` reported alongside |
| 3 | `c_red` default | swept; headline uses the value where SafeTail's mean K ≈ 3 |
| 4 | Does the 1.0 baseline train online in-run, or train-then-freeze? | **train then freeze**, matching 2.0's `_save_and_enter_testing` split, so both are evaluated in inference mode |
| 5 | Seeds | 3 minimum, 5 preferred |
| 6 | **Request-type mapping** (S-14) | `s = Speech`, `d = Detect`, `p = Predict` — **settled by the dataset**, not open. BTP Table 3.1 and the code are correct; BTP §3.3/§4.6 prose and HED item 9 are wrong. ⚠️ This **reverses the 19 Aug 2026 decision** that treated §3.3 prose as authoritative. Guarded by gate G7. |
| 7 | What is `p` ("Predict")? | **Undocumented in every source.** Someone must establish what workload the `Predict` trace actually is before it can be described in a paper. Ask the BTP authors or the trace collector. |
| 8 | Reward range statement (S-03) | `[0, log 2]`. Fix the report, the docstring and any slide that says 1. |
| 9 | ε schedule (S-16) | Pick subtractive (code) or multiplicative (BTP §4.4); rename `gamma_decay` accordingly. |

### 13.2 Risk register
| Risk | Severity | Mitigation |
|---|---|---|
| **B1 changes every published number** | high | Expected and correct. Freeze `reference_v0`; publish a before/after table; never quietly restate. |
| **SafeTail 2.0 may still lose to MinProp after all fixes** (§4.2 D-10, S-17) | high | This is a legitimate scientific outcome. Plan for it: F3/F4/F7 are designed so an honest negative result is still a publishable contribution (a budget-controlled evaluation showing redundancy-based learned scheduling does not beat a propagation heuristic *in this environment*, with D-12 explaining why). Decide with the advisor before results are written up, not after. |
| **An agent "fixes" correct code to match a wrong document** | high | Real and specific: BTP §3.3 and §4.6 prose both misname the request types, and they read far more authoritatively than a CSV column does. Gate **G7** exists solely to catch this. R9 generalises it. |
| **The specification errata undermine the BTP report itself** | medium | They do. Four of BTP's own sections contradict each other or the code (S-03, S-04, S-10, S-11, S-12), and its headline claim fails on its own data (S-17). Krishna did not write it, so the framing is "inherited codebase, corrected and extended", not self-criticism. Agree the framing with the advisor early — it affects whether the next paper cites BTP or supersedes it. |
| `controller.py` merge conflicts | medium | One writer at a time (§12.4.2) |
| The seam leaks and R3 breaks | medium | G4 in CI from day one, not at the end |
| 1.0 port flattered by extra information | medium | A-1 in §8.5 is the guard; the auditor checks `state_v1.py` sees only `free_slots`, not `server_dynamic` |
| Silent degradation returns | medium | §10.3 + G5 manifest check |

### 13.4 Discrepancies found during implementation (R7 log)

Recorded here rather than silently reconciled. None contradict the plan's design;
each is an *additive* finding.

| ID | Found | Detail | Disposition |
|---|---|---|---|
| **D-36** Δ | 2 Sep, B0 | `src/` is full of emoji `print()`s (`✅`, `⏱`, `📊`, `🎨`). On a Windows console (cp1252) these raise `UnicodeEncodeError` and abort the run. | `src/main.py` now forces `stdout/stderr` to UTF-8 at startup (interim). Full fix = route prints through `_safetail_log` in **B9**. |
| **W-01** Δ | 2 Sep, B1 recon | The 15 regressor wrappers are **not** identical across servers. `server3_regressor/detect_predictor.py` uses a *different* `_build_features` schema (`num_tasks`, `has_speech`, `peak_cpu`, `avg_cpu_clock`, `num_files=500`, no scaler) yet still loads `models/server1/…`. D-02 (all load server 1) holds; the per-server feature code diverges on top of it. | No change to plan. **B1** deletes all 15 wrappers for one parameterised `src/regressors.py`, which erases this. Noted so B1 does not assume a single source schema. |
| **W-02** Δ | 2 Sep, B8 | `Request.server_dicts` has length **6**, not 5 (`[{} for _ in range(6)]`). `compute_step_reward` iterates `range(len(server_dicts))` = 6 and `np.mean`s a length-6 array; slot 5 is always `{}` → contributes 0. Amplifies D-07's "mean over all slots". | Left as-is for B8 (out of scope); **B3** replaces the mean-over-slots collapse with `mean over |A|`, which removes the phantom slot from the denominator. Flagged for the B3 agent. |
| **W-03** Δ | 2 Sep, B8 | `compute_step_reward` `required_keys` lists `cpu_core_usage`, but the body reads `d["cpu_usage"]`. `fill_server_dict` populates both, so it works today; a future refactor that drops one key would pass the guard and then `KeyError`. | Harmless now. Fold the key-list/body mismatch into **B9** hygiene. |
| **D-18 deferral** | 2 Sep, B8 | B8 was scoped to land D-16/D-17/D-19/D-20/D-21. **D-18 (single delay draw)** touches `servers.py`'s `compute_request_time`/`schedule_request` signatures and overlaps B5's metric rework (which also rewrites the draw path). | D-18 moved to **B5**. Recorded here so it is not lost. The seam's `PolicyContext` already carries the *phase-2* estimate only, so seam policies are D-18-safe today. |

### 13.3 Out of scope, but on file
**Tail-Learning** (arXiv 2312.16883; ACM TAAS 2025, 10.1145/3737289) was assessed as a possible additional baseline. Verdict on file: usable **in singleton-plan mode** (`B_i = {(1),…,(5)}`), which is a first-class case of the paper's own action space — no job-splitting or virtualisation needed. It tests class-level queueing-theoretic single-assignment routing against per-request learned scheduling, at matched K=1. No public code exists. **Not part of this phase.**

---

## 14. Appendix

### 14.1 SafeTail 1.0 algorithm — the porting specification

Transcribed from `baselines/safetail_v1/_spec_source/v1_agent.py` (verified 2 Sep 2026). This is what `baselines/safetail_v1/` must reproduce.

**Constants** (`v1_constants.py`, root): `β=4`, `α=0.005`, `τ=median_computation_delay=0.048`, `lr=1e-7`, `γ=0.95`, `gamma_decay=1.8e-6`, `batch=128`, `nS=β+1`, `nA=2^β−1`.
Per-task (`instance`): `β=5`, `α=0.001`, `τ=2.5`, `lr=1e-6`, `nS=2β+2`, `nA=31`, `epochs=2`. **Use the `instance` shape** (β=5 matches 2.0).

**State** — `get_state_input(state)` concatenates, in this order:
`LOAD[0..β-1]`, then `MESSAGE_SIZE`, then `RESOLUTION` (omit — A-4), then `BANDWIDTH`, then `PROPOGATION[0..β-1]`. ⇒ `nS = 2β + 2` without resolution.

**Action** — `get_subsets(set(range(β)))[action]`, bitmask enumeration, index 0 dropped ⇒ 31 non-empty subsets for β=5.

**Network** — `build_model()`:
```
Dense(2·nS, input_dim=nS, activation='sigmoid')
BatchNormalization()
Dense(4·nS, activation='sigmoid')
BatchNormalization()
Dense(nA, activation='softmax')
compile(loss='categorical_crossentropy', optimizer=Adam(lr))
```

**Action selection** — ε-greedy: `rand() <= ε` ⇒ `randint(0, nA)`, else `argmax(model.predict(state))`. Log `access_rate = |subset|/β`.

**Observed latency** — for each node in the chosen subset:
`node_latency = computation + transmission + propagation`, and `obs_latency = min` over the subset.
1.0's own components (replaced by the 2.0 environment in the port):
`transmission = 8·message_size/uplink_kbps + 8·message_size/downlink_kbps`, with `uplink = downlink = BANDWIDTH/LOAD[node]`;
`propagation = random.choice(ping_data[LOAD[node]−1])`;
`computation = regressor(...) + |N(0, σ_load)|`.
For `instance`/`noise`, 1.0 applies `node_latency = log(1 + node_latency)` before the min. **Decide and record** whether the port keeps this compression — it changes τ's meaning. Recommendation: **drop it**, use raw latency, and set τ accordingly; record as `V1-DEV-01`.

**Reward** — the 5-case τ-referenced penalty:
```
λ = obs_latency − τ
γ_r = |A| − 1                       # number of redundant servers
δ  = α·exp(−λ)   if λ < 0
     α·exp(+λ)   if λ ≥ 0

λ == 0                        →  R = 0
λ  > 0 and (β − γ_r) == 1     →  R = 0
λ  > 0 and (β − γ_r)  > 1     →  R = −exp(β − γ_r − 1) · δ
λ  < 0                        →  R = −exp(γ_r) · δ
```
Reward is always ≤ 0. Interpretation: **late** ⇒ penalty scaled by how *few* servers were used (you should have replicated more); **early** ⇒ penalty scaled by how *many* were used (you wasted compute). This is exactly the redundancy pricing that SafeTail 2.0 lacks (M-04, D-07) — porting it is therefore both a baseline *and* the reference implementation for B4.

Guarded by `if abs(obs_latency − τ) < 1000:` — outside that band the function **returns `None`**, which then enters the replay buffer. Port as `0.0` + a counter (`V1-BUG-01`, §8.6).

**Training** — `experience_replay(batch)`:
```
minibatch = random.sample(memory, batch)
targets = model.predict(states)
targets[arange(batch), actions] = rewards + γ·max(model.predict(next_states), axis=1)
model.fit(states, targets, epochs=epochs, validation_split=0.2)
if ε > ε_min: ε -= gamma_decay
```
Replay memory `deque(maxlen=2500)`. Note: 1.0 uses **no target network** and takes the max over the online net. Keep it — it is faithful.

**Episode structure** — `no_of_episodes` × `len_of_episode` steps; state advances by walking rows of `data.csv` (a load trace). In the port, episodes come from the 2.0 controller instead; keep 1.0's ε schedule and replay cadence.

**1.0's evaluation harness** (`v1_basline_compare.py`) — the metric conventions to mirror:
| Series | Definition |
|---|---|
| `MIN` (Oracle) | `subsets[-1]` = all β servers, take the min ⇒ M-01 |
| `DQN` | argmax policy, min over selected subset; also logs `|subset|/β` |
| `RAND_k` | uniform random subset of size k, min over it |
| `MIN_LOAD` | sample 3 nodes, pick lowest `LOAD`, single node |
| `MIN_PROP` | `argmin(PROPOGATION)`, single node |
| Reported | `np.percentile(series, [50, 90, 95, 99])` + grouped bar charts by access rate |

### 14.2 Fast file map

| Need | Go to |
|---|---|
| what a function does | `audit/analysis_v2_extracted.txt` §4 (inventory) and §6 (explanations) |
| who calls what | same file, §5 (call graph) |
| paper ↔ code traceability | same file, §7 |
| what is missing vs each paper | same file, §8 |
| **whether a document claim is trustworthy** | **§4.5 of this file (S-register) + R9** — check *before* implementing anything sourced from HED or BTP |
| the 1.0 algorithm | §14.1 above; source in `baselines/safetail_v1/_spec_source/` |
| pre-fix numbers | `results/reference_v0/` |
| the defect list | §4 of this file — **canonical**, supersedes the PDF where they differ |
| what `s`/`d`/`p` mean | the dataset (`Scripts Executed` column), never the prose — see S-14, gate G7 |

### 14.3 Where the audit and this plan differ

The audit (19 Aug) is correct on everything it covers. This plan **adds** 11 code defects it did not find (marked Δ in §4.1–§4.4), of which four are decisive: **D-02** (all regressors point at server 1 — heterogeneity does not exist in the computation path), **D-03** (a +5 ms handicap applied to one baseline family at plot time), **D-10** (SafeTail loses to MinProp on the shipped logs at every percentile), and **D-11/D-12** (the plotted metric excludes the queueing delay the reward optimises, and ~45% of what remains is policy-independent noise).

It also adds an entire layer the audit did not attempt: **§4.5, the specification-defect register**, which audits HED and BTP *against each other* rather than against the code. The audit's §8.2/§8.3 list gaps "from HED" and "from BTP" as though both were valid requirements. They are not — HED is a truncated draft with a sign-inverted reward and two incompatible waiting-time definitions, and BTP fixed some of that while contradicting itself in four places. Without §4.5, an agent working from the audit alone would restore HED's broken reward term (S-02) and rename correct code to match wrong prose (S-14).

Where they differ, **§4 of this file — including §4.5 — is canonical.**
