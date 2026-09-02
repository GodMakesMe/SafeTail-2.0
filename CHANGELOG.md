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
| G1 | `tools/verify_heterogeneity.py` | not started (B1) |
| G2 | `tools/audit_replay.py` | not started (B2) |
| G3 | `tools/audit_reward.py` | not started (B3) |
| G4 | `tools/verify_isolation.sh` | **PASS** (seam + oracle; baselines/ deletable) |
| G5 | `tools/check_manifest.py` | not started (D) |
| G6 | `tools/verify_figures.py` | not started (D) |
| G7 | `tools/verify_types.py` | **PASS** (S-14 dataset mapping) |
