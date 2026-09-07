# Handoff → Claude Code (terminal session)

> ## ✅ CLOSED — 8 September 2026
>
> This handoff was **completed**. The terminal session reported back in
> `HANDOFF_FROM_CLAUDE_CODE.md`: **45 passed / 0 failed**, all seven gates PASS,
> clean smoke run. **There was no TensorFlow problem** — §5 below was never
> needed. Read this file now only for the context in §2–§3.
>
> **Its §4 mystery is resolved: the "different actor" editing the tree during
> verification was the Cowork session**, applying the D-40b dataset-prep work in
> the same window. Not a second agent, not a rogue process — a coordination
> failure on the Cowork side, which should have paused while a verification run
> was in flight. Nothing was lost; the overlap only made that session's 45/45
> describe a tree that had already moved.
>
> **All four of its findings are applied:**
>
> | Finding | Resolution |
> |---|---|
> | §3.1 `_manifest.py` recorded `HANGELOG.md` (`l[3:]` after a `.strip()`) | fixed — `l[2:].strip()`; verified live |
> | §3.2 step 5 unachievable (G5 scans `results/*`, smoke writes elsewhere) | `check_manifest.py --dir` added; step 5 corrected below |
> | §3.3 "35 pass / 10 blocked" arithmetic | corrected in the report — 19 blocked, and the gap is now closed |
> | §4 `server5.csv` is 369/363, not 363/363 | corrected at three sites; see report §4b.12 |
> | §4 `max_burst` hardcoded as `4` in the `no_of_episodes` derivation | fixed — declared above the derivation, now genuinely derived |
>
> **The suite is now 48 tests**, not 45 — the D-40b work added three. Re-measure
> rather than citing either number.
>
> One thing that session flagged and everyone should keep: because
> `TRACE_SAMPLING` now defaults to `"sample"` and server 5 has six multi-row
> contention strings, **runs made after 8 Sep are not comparable to runs made
> before it unless `SAFETAIL_TRACE_SAMPLING=first` is set.** `SAFETAIL_LEGACY_ENV=1`
> pins `first`, so `results/reference_v0` reproduction is unaffected.
>
> **Its §6 step 6 stands and is now the top priority: commit.** Every manifest
> written from this tree is stamped `NOT PUBLISHABLE` for the correct reason
> (`git.dirty: true`), so no run made before a commit can ever be provenanced.


**Read this file first.** It carries the full context of the Cowork session of
7–8 September 2026 so you do not have to re-derive it.

**Repo:** `E:\Project\IP_Arani\heterogenous` (branch `master`)
**Your job:** run the verification that the Cowork session could not run, and fix
anything that actually fails. **Do not re-audit the codebase.**

---

## 1. Why you exist

The Cowork session applied 11 fixes and wrote
`FIXES_REPORT_SafeTail2.0_to_heterogenous.md`. It verified everything it could,
with **one gap**: its Linux sandbox has no TensorFlow, so **10 of 45 tests could
not run**. Every one of them fails on the same line —
`ModuleNotFoundError: No module named 'tensorflow'` — in a module the sandbox
imports transitively. None appeared to fail on the changed code, **but that is an
inference, not an observation.** You convert it into an observation.

**Important:** there is probably **nothing wrong with TensorFlow on this
machine.** `requirements.txt` pins `tensorflow==2.20.0`, and `CHANGELOG.md`
records gate **G0 PASSING** in `.venv` (Python 3.12, tf 2.20, sklearn 1.4.2).
Expect the tests to pass. Only start repairing if something actually breaks.

---

## 2. Context you need (do not re-derive)

This is a **B.Tech project audit**. Krishna inherited a codebase (`SafeTail 2.0`,
heterogeneous edge scheduling) written by a previous batch, and is correcting it.
The framing is *"inherited codebase, corrected and extended"* — not self-criticism.

**Read these, in this order, before touching anything:**

| File | What it gives you |
|---|---|
| `FIXES_REPORT_SafeTail2.0_to_heterogenous.md` | **the master narrative** — what was fixed, what is open, why |
| `KNOWME.md` | which SafeTail is which (there are five variants; do not mix them up) |
| `CHANGELOG.md` | per-change ledger; the newest four rows are this session's |
| `plan.md` §4, §4.4b, §11 | defect register (`D-01`…`D-45`) and the eight audit gates |
| `audit/ERRATA.md` | specification errata `S-01`…`S-20` |

**Scope rule, stated by Krishna and non-negotiable:**

> **Only SafeTail 2.0 (the heterogeneous branch) is in scope.**
> SafeTail 1.0 is the *baseline*. Its defects (`S-18`/`S-19`/`S-20` — its published
> code does not implement its published paper) are **documentation only**.
> **Do not "fix" SafeTail 1.0.** `baselines/` was not modified this session and
> must stay that way unless Krishna says otherwise.

**Do not touch, without being asked:**

* `baselines/**` — the isolated baseline unit (gate G4 asserts `rm -rf baselines/`
  leaves a working repo)
* `results/reference_v0/**` — frozen, read-only. It is byte-identical (md5, all 5
  CSVs) to the published run the advisor has plots for.
* `baselines/safetail_v1/_spec_source/**` — read-only reference copies, never imported

---

## 3. What changed in the session you are verifying

Modified: `src/agent.py`, `src/constants.py`, `src/controller.py`, `src/main.py`,
`src/user.py`, `tools/make_figures.py`, `plotting/plot_baselines.ipynb`,
`CHANGELOG.md`, `plan.md`.
New: `src/_manifest.py`, `tools/check_manifest.py`,
`tools/tests/test_followup_fixes.py`, `FIXES_REPORT_*.md`, `audit/dossier_uttam/`.

| ID | Change | Risk to watch |
|---|---|---|
| **D-38** | `P(T)` — restored the missing leading `1 -`. Satisfaction was *increasing* with lateness. | changes the **episodic** reward → the 2.0 policy will differ |
| **D-42** | socket entry point waited on an unreachable event with no timeout → bounded `SAFETAIL_DRAIN_TIMEOUT` (default 120 s) | only the socket path; `--run` was always fine |
| **D-43** | episode length (3 chunks) and episode count (÷4) disagreed → one `constants.CHUNKS_PER_EPISODE` drives both | default is still **3**, so episode length is **unchanged** |
| **D-44** | `Request.__init__` accepted `deadline` and discarded it → now stored | **adds 2 elements to the state vector** (`request_to_state_array` flattens every attribute) |
| **D-32** | access-rate log was a cumulative mean; fixed-K baselines logged nothing | new `Controller.episode_access_rates`, reset in `finalize_episode` |
| **M-14** | `generate_testing_plots()` was `pass` *and* never called → implemented + wired | writes `plots/testing/` + `testing_metrics.csv` |
| **G5** | new: `manifest.json` per run + `tools/check_manifest.py` | new file written at end of every run |
| **D-03** | deleted the hardcoded `+0.005` MinProp offset from both notebook cells | gate G6 was warning about it |
| **S-16** | one correctly-named `epsilon_decay_step`; `gamma_decay` is now an alias | schedule shape **unchanged** (still subtractive) |
| **D-11** | `constants.LATENCY_METRIC` was read nowhere → now sets `make_figures` default | |
| **D-45** | `latency_log.csv` mixes seconds (components) and ms (totals) → explicit `_TO_MS` map, columns **deliberately not renamed** | renaming would make every run under `results/` unreadable |

---

## 4. Run this — in order

```powershell
cd E:\Project\IP_Arani\heterogenous

# 0. environment gate (expected: PASS)
.\.venv\Scripts\python.exe tools\verify_env.py

# 1. THE ONE THAT MATTERS — the 10 tests Cowork could not run
.\.venv\Scripts\python.exe -m pytest -q
#    expected: 45 passed

# 2. the new tests on their own, if step 1 fails and you need to isolate
.\.venv\Scripts\python.exe -m pytest tools\tests\test_followup_fixes.py -q
#    expected: 21 passed

# 3. all eight gates
.\.venv\Scripts\python.exe tools\verify_heterogeneity.py    # G1
.\.venv\Scripts\python.exe tools\audit_replay.py            # G2
.\.venv\Scripts\python.exe tools\audit_reward.py            # G3
bash tools\verify_isolation.sh                              # G4
.\.venv\Scripts\python.exe tools\check_manifest.py          # G5  (new)
.\.venv\Scripts\python.exe tools\verify_figures.py          # G6
.\.venv\Scripts\python.exe tools\verify_types.py            # G7
.\.venv\Scripts\python.exe tools\verify_dataset.py          # G8 (advisory: exit 0
                                                            #     + [INADEQUATE] on
                                                            #     today's data.
                                                            #     --strict to BLOCK)

# 4. THE OTHER THING COWORK COULD NOT DO — a real smoke run.
#    This is the only thing that exercises the new manifest writer, the D-32
#    per-episode reset and the D-38 P(T) repair AT RUNTIME rather than through
#    source-level assertions.
$env:SAFETAIL_SEED="0"
.\.venv\Scripts\python.exe src\main.py --smoke

# 5. then confirm the smoke run is provenanced.
#    NOTE (corrected 8 Sep): --smoke writes to tools/out/smoke_logs, which is
#    OUTSIDE results/, and G5 scans results/* by design (plan.md §11). The bare
#    invocation therefore cannot see it. Use --dir, added for exactly this:
.\.venv\Scripts\python.exe tools\check_manifest.py --dir tools\out\smoke_logs
#    Expect it to FAIL on `git.dirty: true` until the fixes are committed --
#    that is the gate working, not a defect.
```

### What "success" looks like

* `pytest` → **45 passed**
* all eight gates → **PASS** (G8 is advisory; G1 emits one *acknowledged* warning about server 1 ≡
  server 2 — that is D-15, expected, not a failure)
* the smoke run exits on its own and writes `manifest.json`
* `check_manifest.py` reports the smoke dir as ok

---

## 5. If TensorFlow actually is broken

Diagnose before reinstalling. In order of likelihood:

**(a) OpenMP DLL collision — the known Windows hazard.**
sklearn/scipy and TensorFlow each ship `libiomp5md.dll`. Importing one then the
other in the same process gives `DLL initialization routine failed`.
`tools/tests/conftest.py` already sets `KMP_DUPLICATE_LIB_OK=TRUE` and imports TF
first. If you see this **outside** pytest, set it in the shell:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

**(b) numpy ABI.** TF 2.20 against the pinned `numpy==1.26.4`. If you see
`numpy.dtype size changed` or `_ARRAY_API not found`, something upgraded numpy to
2.x. Fix by restoring the pin — **not** by upgrading TF:

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy==1.26.4" --force-reinstall
```

**(c) Clean rebuild — last resort.** `requirements.lock.txt` is the authority:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

**Rules while repairing:**

* **Do not upgrade `scikit-learn` off `1.4.2`.** The 15 `models/*/*.pkl` were
  pickled with it; a newer sklearn raises
  `AttributeError: Can't get attribute '__pyx_unpickle_CyHalfSquaredError'`.
  (Confirmed — the Cowork sandbox hit exactly this on sklearn 1.7.)
* **Do not upgrade numpy to 2.x.** Gate G0 asserts numpy 1.x.
* **Do not silence a test to make it pass.** If a test genuinely fails, that is
  the finding — report it.

---

## 6. If a test fails on the changed code

Report, don't improvise. For each failure give: test name, full traceback, the
defect ID from its name (`test_D38_*` → D-38), and which of the changes in §3 it
implicates. Then propose a fix and wait — several of these changes were made
deliberately and reverting one would undo a real repair.

**Two changes are *supposed* to change numbers.** If a test asserts old values,
the test is wrong, not the fix:

* **D-38** changes the episodic reward
* **D-44** adds 2 elements to the state vector

Both affect **only the native SafeTail 2.0 policy**. SafeTail 1.0 builds its own
flat state (`baselines/safetail_v1/state_v1.py`, `nS = 2β+2`) and uses its own
τ-referenced reward; the Oracle uses neither. Verified this session.

---

## 7. After verification passes

Report back and stop. Do **not** start these without Krishna's go-ahead — each
changes published numbers or needs hours:

* **re-run the SafeTail 2.0 side** (required by D-38 + D-44; baselines/Oracle/1.0
  are unaffected, so only the 2.0 rows need regenerating)
* **B1b** — drop the `total_processing_time` target leak (D-25), adopt HED §IV-D
  features (M-15), retrain all 15 regressors
* **D-40** — restore variance in computation time. *This is a research decision,
  not a bug fix.* See §4b.3 of the report.
* the real **F8** (`c_red` sweep) — the current `F8_*` files are a different
  experiment that took the name

---

## 8. Open questions Krishna is deciding — do not pre-empt

1. **D-40 / D-41** — the testbed has no tail to optimise and little for a
   scheduler to learn. Going to the advisor.
2. **Server 2** — this branch aliases it to server 1 (β=5); the external dossier
   excludes it (β=4). Both defensible; **must not be mixed in one comparison.**
3. **S-17** — BTP §6.2's headline claim is false on its own data.

---

*Written by the Cowork session, 8 September 2026. If this file disagrees with the
code, the code is right and this file is stale — say so.*
