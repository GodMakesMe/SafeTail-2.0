# Changelog

Newest first. One row per merged change. Format (plan.md §10.6):

`date · ID · files · what · why · gate · changes published numbers? (Y/N)`

A `Y` row must link to a before/after table (kept under `results/` or `figures/`).

Convention: **ID** is the defect (`D-xx`), missing-feature (`M-xx`), spec-defect
(`S-xx`), workstream (`B0`…`B12`, `C-*`), or v1-port (`V1-BUG-xx`, `V1-DEV-xx`)
identifier from `plan.md`. The same ID must also appear as a `[SAFETAIL][…][ID]`
tag at the code site and in the regression test name.

---

## Unreleased

| date | ID | files | what | why | gate | Δ numbers |
|---|---|---|---|---|---|---|
| 2026-09-02 | B3 / D-07 M-04 S-02 (+W-02) | `src/controller.py` (`_collapse_step_reward`), `src/constants.py` (`C_RED`, `C_RED_SWEEP`), `tools/audit_reward.py` (new), `tools/tests/test_b3_redundancy.py` (new) | step reward collapsed as `mean_{i∈A} log(1+headroom_i) − c_red·(|A|−1)/(β−1)`: mean over selected servers (not constant 6, drops the W-02 phantom slot) + explicit redundancy cost. `c_red=0` reproduces old behaviour. Gate **G3** PASS (c_red=0 flat in \|A\|; c_red=0.1 strictly decreasing). | nothing priced redundancy — all three reward terms pushed toward more servers (D-07 → over-replication D-08). S-02: BTP's headroom fix dropped HED's only \|A\|-sensitive quantities. | G3 | **pending** — the K-vs-`c_red` ablation (F8) needs full runs; smoke is too short for the agent to exploit. |
| 2026-09-02 | B2 / D-04 D-05 D-06 | `src/controller.py`, `src/agent.py`, `tools/audit_replay.py` (new) | snapshot `s_t` as an array at action time (pre-mutation); `s_{t+1}` snapshotted post-action; `DQNAgent.store_arrays`; keep per-step reward + broadcast `R_ep/N` bonus instead of overwriting. Gate **G2** PASS (450 transitions, 0 outcome-leaks, 100% `s_t≠s_{t+1}`, per-step rewards vary in 30/30 episodes). | replay state used to contain the outcome of its own action (D-04), `next_state==state` (D-05), step rewards discarded (D-06) — Bellman target was meaningless | G2 | **pending** — training signal changes; measure at next full run |
| 2026-09-02 | B1 / D-02 D-02b D-02c (+D-15, W-01) | `src/regressors.py` (new), `src/servers.py`, `src/constants.py`, `src/_legacy_regressor_wrappers/` (moved), `tools/verify_heterogeneity.py` (new), `tools/tests/test_b1_regressors.py` (new) | one `TracePredictor(server_index, task)` loading the right per-server model+CSV (ports all 3 shipped feature schemas); server 2 aliases server 1 (D-15); load/lookup failure raises unless `ALLOW_DEGRADED_PREDICTORS`. Gate **G1** PASS. | all 15 wrappers hardcoded `models/server1/` + `dataset/server1.csv` — computation path was not heterogeneous; failures were silent | G1 | **YES (expected)** — computation delay now server1/2≈12ms, server3≈388ms, server4≈266ms, server5≈13ms (was ≈12ms for all 5). Re-baseline at next run; likely moves D-10. |
| 2026-09-02 | B8 / D-16 D-17 D-19 D-20 D-21 | `src/controller.py`, `src/servers.py`, `src/user.py`, `tools/tests/test_b8_structural.py` (new), `pytest.ini` (new) | D-17 zero the failing server's own slot (was `[server_idx-1]`); D-16 drop GPU factors for CPU-only servers + geometric-mean renormalise (range stays `[0,log2]`, S-03); D-19 `Request.contention_str` — stop clobbering `.combination`, add `contention_str` column to `latency_log.csv` (`request_type` now ∈ {s,d,p}); D-20 `request_*_done` deadline-conditional at the one site `T` is known, unconditional increments removed; D-21 saturation retry is bounded (6×, exp backoff, `dropped_requests` counter) not unbounded recursion. 6/6 regression tests pass. | plan.md B8 | **pending** — reward math (D-16) and completion ratio (D-20) change; measure at next run. No run yet. |
| 2026-09-02 | B12 / S-01…S-17 | `audit/ERRATA.md` (new) | standalone specification-errata register: 17 entries, provenance code + document section + code/dataset evidence + correction per row; cross-ref table to workstreams | plan.md B12 accept: every S-row has an entry citing a section and a code location or dataset fact; advisor-facing artefact | G7 (S-14) | N (docs) |
| 2026-09-02 | B12 / S-14 | `tools/verify_types.py` (new) | gate **G7**: assert `s→Speech, d→Detect, p→Predict` from the dataset on all 5 servers, Speech slowest, `ORIGINAL_DEADLINES` pairing, and (when present) `regressors.TASK_FOR_LETTER` / legacy-wrapper `scripts.index(...)`. PASS | S-14: BTP §3.3/§4.6 prose + HED item 9 misname the types and read more authoritatively than a CSV column; G7 makes the dataset the arbiter (reverses the 19 Aug note) | — | N |
| 2026-09-02 | B7 / smoke / D-36 | `src/constants.py`, `src/_seeding.py` (new), `src/main.py`, `src/controller.py` | `--smoke` (socket-free ~20-episode run), `SEED` threaded to random/numpy/tf, `POLICY` env; UTF-8 stdout reconfigure (D-36: emoji prints crash on Windows cp1252); plots suppressed under SMOKE | every pipeline gate (G2/G3/G4a) needs a fast deterministic run; none existed | G4 | N |
| 2026-09-02 | C / M-01 / M-02 | `src/policy_registry.py` (new), `src/controller.py`, `baselines/**` | policy seam: `PolicyContext` + registry + `subset↔index`; `Controller.policy` branch in `process_step`/`finalize_episode`; `baselines/` deletable unit with oracle (M-01) running end-to-end; `tools/verify_isolation.sh` gate **G4** | plan.md §8: SafeTail 1.0 baseline is a port through one 50-line seam; `rm -rf baselines/` must leave a working repo (R3) | G4 | N |
| 2026-09-02 | B0 / D-01 | `requirements.txt` | re-encode UTF-16LE→UTF-8/LF; add `scikit-learn==1.5.2`, `scipy==1.13.1`, `seaborn==0.13.2`, `joblib`, `threadpoolctl` | a clean install per the repo's own instructions produced a system with no sklearn, so every `models/*/*.pkl` unpickle raised `ModuleNotFoundError` and D-02c fired silently on every server | G0 | N (enables real numbers; no run yet) |
| 2026-09-02 | B0 | `tools/verify_env.py` (new) | gate **G0**: assert Python 3.11/3.12, numpy 1.x, every pinned version present, all 15 regressor pickles load + expose `.predict()` | today the regressor-layer collapse is silent (D-02c); G0 makes it loud and blocks every downstream workstream | — | N |
| 2026-09-02 | — | `.git`, `CHANGELOG.md` (new) | initialise git in `heterogenous/`; seed commit of the verbatim `SafeTail-2.0-main` copy; start this changelog | checkpointing discipline (plan.md §0, §10.6) | — | N |

---

## Gate status

| Gate | Script | State |
|---|---|---|
| G0 | `tools/verify_env.py` | **PASS** (venv `.venv`, Python 3.12, tf 2.20, sklearn 1.4.2) |
| G1 | `tools/verify_heterogeneity.py` | **PASS** (computation delay now truly per-server) |
| G2 | `tools/audit_replay.py` | **PASS** (D-04/D-05/D-06 on instrumented smoke) |
| G3 | `tools/audit_reward.py` | **PASS** (redundancy priced; reward well-shaped) |
| G4 | `tools/verify_isolation.sh` | **PASS** (seam + oracle; baselines/ deletable) |
| G5 | `tools/check_manifest.py` | not started (D) |
| G6 | `tools/verify_figures.py` | not started (D) |
| G7 | `tools/verify_types.py` | **PASS** (S-14 dataset mapping) |
