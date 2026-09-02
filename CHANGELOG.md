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
| 2026-09-02 | B0 / D-01 | `requirements.txt` | re-encode UTF-16LE→UTF-8/LF; add `scikit-learn==1.5.2`, `scipy==1.13.1`, `seaborn==0.13.2`, `joblib`, `threadpoolctl` | a clean install per the repo's own instructions produced a system with no sklearn, so every `models/*/*.pkl` unpickle raised `ModuleNotFoundError` and D-02c fired silently on every server | G0 | N (enables real numbers; no run yet) |
| 2026-09-02 | B0 | `tools/verify_env.py` (new) | gate **G0**: assert Python 3.11/3.12, numpy 1.x, every pinned version present, all 15 regressor pickles load + expose `.predict()` | today the regressor-layer collapse is silent (D-02c); G0 makes it loud and blocks every downstream workstream | — | N |
| 2026-09-02 | — | `.git`, `CHANGELOG.md` (new) | initialise git in `heterogenous/`; seed commit of the verbatim `SafeTail-2.0-main` copy; start this changelog | checkpointing discipline (plan.md §0, §10.6) | — | N |

---

## Gate status

| Gate | Script | State |
|---|---|---|
| G0 | `tools/verify_env.py` | written — pending first green run (needs venv) |
| G1 | `tools/verify_heterogeneity.py` | not started (B1) |
| G2 | `tools/audit_replay.py` | not started (B2) |
| G3 | `tools/audit_reward.py` | not started (B3) |
| G4 | `tools/verify_isolation.sh` | not started (C) |
| G5 | `tools/check_manifest.py` | not started (D) |
| G6 | `tools/verify_figures.py` | not started (D) |
| G7 | `tools/verify_types.py` | not started (B12) |
