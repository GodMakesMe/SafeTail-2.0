# Reverse handoff ← Claude Code (terminal session, 8 September 2026)

**Counterpart to `HANDOFF_CLAUDE_CODE.md`.** That file asked this session to run
the verification the Cowork session could not run. This file reports the result.

**Repo:** `E:\Project\IP_Arani\heterogenous` (branch `master`)
**Verdict:** **the gap is closed. Everything passes. No fix was required.**

---

## 0. Read this part first — the snapshot boundary

**These results describe the tree as it stood at 01:03:00 IST on 8 Sep 2026:**
commit `fd6cc64` ("S-18 VERDICT: the headline reverses…") plus the Cowork
session's 11 uncommitted fixes, and **nothing else**.

**At 01:03:30 — after my smoke run finished — a different actor began editing
this repo, and was still editing it when I stopped writing this file (01:07:24).**
I did not make those edits. They are described in §4. Everything in §1–§3 was
measured *before* they landed.

If you are reading this later, **`git status` will not match what I saw.** Take
§1–§3 as a statement about `fd6cc64` + the 11 fixes, not about whatever is on
disk now.

---

## 1. The verdict

The Cowork session's one gap was that its Linux sandbox had no TensorFlow, so a
set of tests could not run and its conclusion — "none of them fails on anything I
changed" — was an inference. **It is now an observation.**

There was no TensorFlow problem on this machine. §5 of the inbound handoff (the
OpenMP DLL collision, the numpy ABI break, the clean rebuild) was never needed.
Nothing was diagnosed, nothing was reinstalled, nothing was repaired.

| Check | Command | Result |
|---|---|---|
| **G0** environment | `tools/verify_env.py` | **PASS** — Python 3.12, numpy 1.26.4, all 77 pins match, 15/15 pickles load |
| **Full suite** | `python -m pytest -q` | **45 passed, 0 failed** |
| Previously-blocked files, isolated | `pytest test_b3 test_b4 test_b8 test_c_v1` | **19 passed, 0 failed** |
| **G1** heterogeneity | `tools/verify_heterogeneity.py` | **PASS** — 12.45 / 12.45 / 388.38 / 266.42 / 13.18 ms |
| **G2** replay | `tools/audit_replay.py` | **PASS** — 450 transitions; D-04, D-05, D-06 all ok |
| **G3** reward | `tools/audit_reward.py` | **PASS** — D-07/M-04, D-16, D-17 |
| **G4** isolation | `bash tools/verify_isolation.sh` | **PASS** — all four seam checks |
| **G5** provenance | `tools/check_manifest.py` | **PASS** — 0 failing, 20 unprovenanced (pre-gate) |
| **G6** figures | `tools/verify_figures.py` | **PASS** — warning-free |
| **G7** types | `tools/verify_types.py` | **PASS** |
| **Smoke run** | `SAFETAIL_SEED=0 python src/main.py --smoke` | **exit 0** — 20 episodes, 300 requests, 11.8 s, `dropped=0`, `manifest.json` written |

G1's only warning is the acknowledged D-15 alias (server 1 ≡ server 2). That is
expected and is not a failure, exactly as the inbound handoff said.

---

## 2. The two runtime facts only a real run could establish

The inbound handoff was right that the smoke run is the only thing that exercises
D-32 and D-38 at runtime rather than through source-level assertions. Both hold.

**D-32 — the per-episode access rate is genuinely per-episode.** From
`tools/out/smoke_logs/`:

```
0.480  0.413  0.480  0.573  0.480  0.493  0.533  0.440  0.507  0.533
0.520  0.560  0.600  0.453  0.453  0.520  0.507  0.520  0.533  0.547
```

* non-monotone — a cumulative mean cannot do this
* equals the per-episode mean of `request_wise_access_log.csv` exactly
  (`np.allclose` → `True`)
* is **not** the running cumulative mean (`np.allclose` → `False`)

**D-38 — P(T) is not inverted at runtime.** 289 distinct step rewards spanning
[0.3259, 0.5741]; 20 distinct episodic rewards spanning [3.574, 3.950]. Bounded,
varying, no sign inversion. (G2 independently reports 431 distinct stored
transition rewards over its own 450-transition run.)

---

## 3. Three things I found. None is a failure; two want a decision.

### 3.1 A real one-line bug in `src/_manifest.py` (new this session)

Every manifest records its first dirty filename with the first character
missing — the smoke manifest says `"HANGELOG.md"`, not `"CHANGELOG.md"`.

**Cause.** `_git()` (line ~54) returns `out.stdout.strip()`. That strip removes
the leading space of the *first* `git status --porcelain` line only. Line 69 then
does `l[3:]`, which is correct for every line except that first one, where it
eats one character too many.

**Impact.** Cosmetic. `git.dirty` (the boolean the gate actually tests) and all
of `check_manifest.py`'s logic are unaffected. But it corrupts a provenance
record, which is the one thing a manifest exists to get right.

**Fix — one line, not applied:**

```python
# src/_manifest.py:69
"dirty_files": [l[2:].strip() for l in status.splitlines()][:50] if status else [],
```

`l[2:].strip()` is correct both for the stripped first line (`"M CHANGELOG.md"`)
and for every unstripped later line (`" M plan.md"`, `"?? src/_manifest.py"`).

### 3.2 `HANDOFF_CLAUDE_CODE.md` §4 step 5 is not achievable as written

It says the smoke run's directory "should now report as ok" from
`check_manifest.py`. **It never can.** G5 scans `results/*` only — by design, per
plan.md §11 ("every `results/*/manifest.json`") — and `--smoke` writes to
`tools/out/smoke_logs`. The gate is right; the instruction is stale.

I validated the smoke manifest through the gate's own `check_one()` instead:

* all 7 required keys present and non-empty; `schema` 1; `seed` 0
* 20 input hashes, 10 tracked packages, 59 constants
* **`degraded` ledger empty** — the plan.md §10.3 publishability condition
* one flag: `git.dirty: true`

That flag is **correct**, not a defect: the 11 fixes are uncommitted, which is
precisely why the runner printed `NOT PUBLISHABLE`. Commit them and it clears.

**Suggested edit** to the inbound handoff (or to the gate, your call): either
change step 5 to run `check_one` against `tools/out/smoke_logs`, or teach
`check_manifest.py` an explicit `--dir <path>` for out-of-tree runs.

### 3.3 A bookkeeping discrepancy in `FIXES_REPORT_*.md` §8

The report says "**35 pass, 10 blocked**". The four files it names as blocked —
`test_b3_redundancy`, `test_b4_tau`, `test_b8_structural`, `test_c_v1_baseline` —
contain **19** tests, not 10, and 35 + 19 ≠ 45. The full suite is 45 and all 45
pass, so nothing is missing; the arithmetic in that row is just wrong. Worth
correcting before anyone cites it.

---

## 4. ⚠️ The repo was being edited by someone else while I verified it

**This is the item that needs your attention, and it is not mine to resolve.**

Starting at 01:03:30 — after my smoke run, during my session — these files
changed under me:

| Time | File | Status |
|---|---|---|
| 01:03:30, again 01:07:13 | `src/regressors.py` | modified |
| 01:03:39, 01:05:20, 01:07:24 | `src/constants.py` | modified |
| 01:03:48 | `src/servers.py` | modified |
| 01:04:38 | `tools/verify_dataset.py` | **new, untracked** |
| 01:08:11 | `tools/tests/test_followup_fixes.py` | modified — **the suite itself** |

**I wrote none of it.** The changes are tagged `[FIX][D-40]` and
`[SAFETAIL][AUDIT][G8]`, and they are substantial: +199 lines across the three
`src/` files, plus a 229-line new gate script. The tree was still changing when I
finished writing this — `constants.py` was last touched 26 seconds before I saved
this file.

**Why this matters procedurally.** `HANDOFF_CLAUDE_CODE.md` §7 lists D-40 as work
that must not start without Krishna's go-ahead ("*a research decision, not a bug
fix*"), and §8 lists it as an open question going to the advisor. Something is
implementing it right now. **Establish who before anything else proceeds** — if
this is a second agent working from the same handoff, two sessions are editing
one working tree with no lock and no branch.

**What the work itself looks like** (read, not audited — it is mid-flight and not
mine to review):

* `constants.TRACE_SAMPLING` — `sample` | `first` | `mean`, pinned to `first`
  under `LEGACY_ENV` so `results/reference_v0` still reproduces
* `constants.COMPUTATION_SOURCE` — `model` | `empirical`
* `TracePredictor._row` and `Server`'s trace lookup sample among repeated
  measurements instead of hardcoding `row.iloc[0]`
* `tools/verify_dataset.py` — a proposed **gate G8**, "is the trace dataset fit to
  train on?", to be run when new data lands and before training on it

It carries **two findings that correct the existing record**, and they look right
to me on inspection:

1. **`FIXES_REPORT_*.md` §8 says the dataset is "363 rows / 363 unique combos /
   `Iteration==[1]`" for servers 1, 3 and 5. That is wrong for server 5.**
   `server5.csv` has **369 rows for 363 combinations** — `s`, `d`, `p`, `ss`,
   `sd`, `sp` each carry two genuine measurements (e.g. `sd` = 0.033799 s vs
   0.023559 s, a ~30% spread, with differing telemetry — not duplicated rows).
2. **Repeated measurements alone will not fix D-40.** The shipped regressors are
   point predictors fitted on squared error, so they return `E[y|x]`. On the one
   repeat pair that exists today (server5, `sd`): measured 33.799 ms and
   23.559 ms, model **20.906 ms for both**. Collecting a distribution and feeding
   it to a mean-predictor yields a better-estimated mean, not a distribution.
   Hence `COMPUTATION_SOURCE=empirical` as the inference-side half.

**One consequence nobody should miss:** because `TRACE_SAMPLING` now defaults to
`"sample"`, and server 5 *does* have six multi-row contention strings, this is
**not** a no-op on the current dataset — server 5's computation times become
stochastic. The author documented this honestly in the constants comment, and
pinned `LEGACY_ENV` to `"first"` to protect `reference_v0`. But it does mean
**runs made after ~01:03 today are not comparable to runs made before it** unless
`SAFETAIL_TRACE_SAMPLING=first` is set.

**One small fragility I noticed while reading** (low priority, flagging only
because I was already in the file): `constants.py` computes
`_deliverable_chunks = min(no_of_chunk, no_of_burst * 4)` with the comment
`# 4 == max_burst, set below`. The `4` is hardcoded because `max_burst` is
defined further down, so changing `max_burst` silently desynchronises the
derived `no_of_episodes`.

**What I did about the moving tree.** After the first wave of edits I re-ran the
full suite: still **45 passed, 0 failed**, including
`test_D40_computation_prediction_is_still_deterministic`. But `regressors.py` and
`constants.py` changed again afterwards, so **that re-run is already stale too.**
I did not attempt to verify in-flight work.

**And as of 01:08:11 the suite is no longer the same suite.**
`tools/tests/test_followup_fixes.py` went from 21 tests to **24** — three new
D-40 tests appeared alongside the existing one:

```
test_D40_repeated_measurements_are_not_silently_discarded
test_D40_legacy_env_pins_deterministic_row_choice
test_D40_empirical_source_exists_and_is_not_the_default
test_D40_computation_prediction_is_still_deterministic   (pre-existing)
```

So the full suite is now **48 tests, not 45**. **Treat every "45 passed" in this
document as a statement about a test set that no longer exists.** The number to
verify from here is 48, and it must be measured once the D-40 work stops moving —
not before. I deliberately did not run it: a suite being edited mid-run produces
a number that means nothing.

---

## 5. Scope I respected

* **Did not re-audit the codebase.** Every file I opened, I opened to interpret a
  result I had just produced.
* **Did not touch `baselines/`**, `results/`, or `results/reference_v0/`. Nothing
  outside `tools/out/` was written by my run; `git status` gained no entry from
  me.
* **Did not modify a single source file.** The `_manifest.py` fix in §3.1 is
  proposed, not applied.
* **Did not silence, skip or weaken any test.**
* **Did not start anything from §7** of the inbound handoff — no SafeTail 2.0
  re-run, no B1b, no D-40, no F8.

New files on disk from my run, all under the gitignored `tools/out/`:
`tools/out/smoke_logs/` (my smoke run) and `tools/out/g2_logs/` (written by G2).

---

## 6. What the next session should do, in order

1. **Find out who is writing the D-40 code** (§4). Nothing else is safe to
   conclude until the working tree has one owner.
2. **Re-run `pytest -q` and the gates once that work settles.** My 45/45
   describes a tree *and a test set* that no longer exist — expect **48** tests,
   and note that `tools/verify_dataset.py` proposes an **eighth** gate (G8) that
   plan.md §11 does not yet list.
3. **Decide on the `_manifest.py` one-liner** (§3.1). One line, zero risk.
4. **Correct `FIXES_REPORT_*.md` §8** on two points: the "35 pass / 10 blocked"
   arithmetic (§3.3), and the server5 `363/363` claim, which the in-flight work
   shows is `369/363` (§4).
5. **Fix or drop step 5 of `HANDOFF_CLAUDE_CODE.md` §4** (§3.2).
6. **Consider committing.** Every manifest written from this tree is stamped
   `NOT PUBLISHABLE` for the correct reason — `git.dirty: true`. No run made
   before a commit can ever be provenanced.

Still untouched and still awaiting Krishna, per §7–§8 of the inbound handoff: the
SafeTail 2.0 re-run required by D-38 + D-44, B1b, the real F8 `c_red` sweep, the
D-40/D-41 advisor conversation, the server-2 β question, and S-17.

---

*Written by the Claude Code terminal session, 8 September 2026, 01:07 IST.
Results in §1–§3 describe `fd6cc64` + the Cowork session's 11 uncommitted fixes,
measured before 01:03:30. §4 describes a tree that was still moving as I wrote.
If this file disagrees with the code, the code is right and this file is stale —
say so.*
