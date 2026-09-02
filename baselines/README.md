# `baselines/` — a deletable unit

Everything in this directory is **optional**. Deleting the whole tree must leave a fully
working SafeTail 2.0 repository. That property is a hard requirement (plan.md R3) and is
enforced by gate **G4** (`tools/verify_isolation.sh`).

## The rules

1. `src/` **never** imports anything from `baselines/`. One direction only.
2. `baselines/` touches `src/` through exactly one module: `src/policy_registry.py`
   (see plan.md §8.3). Nothing else.
3. `safetail_v1/_spec_source/` holds frozen, **read-only** copies of the SafeTail 1.0
   source, taken from `E:\Project\IP_Arani\SafeTail\` (upstream:
   `github.com/Jyotishokhanda/SafeTail`). They exist **to be read while porting**.
   They are never imported, never executed, never edited. Gate G4c greps for violations.
4. The SafeTail 1.0 baseline is a **port**, not a bridge: it reimplements the 1.0
   *algorithm* (specified in plan.md §14.1) against the SafeTail 2.0 environment, so both
   policies are graded by the same servers, traces and admission rules.

## Status

| Piece | Plan section | State |
|---|---|---|
| `src/policy_registry.py` seam | §8.3 | **done** — `Policy` protocol + `PolicyContext` + registry + `subset↔index` |
| `common/env_adapter.py` + `metrics.py` | §8.4 | **done** |
| `register.py` / `run_baseline.py` | §8.3 | **done** — `python baselines/run_baseline.py --policy <name> [--smoke]` |
| `oracle/policy_oracle.py` (M-01) | §8.7 | **done** — runs end-to-end through the seam; gate G4 green |
| `safetail_v1/` (M-02) | §8.5, §8.6, §14.1 | **done** — port of the 1.0 algorithm; τ per type from `reference_v0`; faithfulness register in `safetail_v1/README.md` |

Remaining for workstream D: call `policy.report()` into each run's `manifest.json`
(gate G5), and the train-then-freeze orchestration for the R-1 run matrix (§9.1).

## Deleting this directory

```bash
rm -rf baselines/
python -c "import sys; sys.path.insert(0,'src'); import policy_registry, controller, main"
POLICY=native BASELINE_MODE=safetail  python src/main.py --smoke
POLICY=native BASELINE_MODE=minload_2 python src/main.py --smoke
POLICY=safetail_v1 python src/main.py --smoke   # must fail loudly, listing known policy names
```

All four behaviours must hold.
