# Run provenance — the SafeTail 1.0 baseline comparison

What was run, under what conditions, and what the comparison is and is not
entitled to claim. Read this before quoting any number from `figures/`.

## The question

> Run SafeTail **1.0** over the same data the heterogeneous SafeTail **2.0**
> servers were given, and compare against the heterogeneous results that already
> exist.

## The two sides

| Side | Source | Code that produced it |
|---|---|---|
| **SafeTail 2.0 (heterogeneous)** | `results/reference_v0/safetail_training_logs/latency_log.csv` — 15,225 requests, shipped with `SafeTail-2.0-main` | the **original, unfixed** 2.0 code |
| **K-baselines** (MinLoad/MinProp/Rand ×{1,2,3}) | `results/reference_v0/baselines/…` | same original code |
| **SafeTail 1.0** | `results/v1_legacy_s0/latency_log.csv` — new, 15,225 requests | `baselines/safetail_v1/` (the 1.0 algorithm) driving the 2.0 environment **in legacy mode** |
| **Oracle** (M-01) | `results/oracle_legacy_s0/latency_log.csv` — new | all-β dispatch, min over the subset; same legacy mode |

## Why "legacy mode", and what it means

The heterogeneous numbers came from the **unfixed** code. To compare against
them, the baseline must experience the **same latency physics**. `SAFETAIL_LEGACY_ENV=1`
restores exactly the four repairs that changed the latency model:

| Restored | Effect |
|---|---|
| **D-02** | every server predicts computation with `models/server1` + `dataset/server1.csv` (the original bug: computation was never actually heterogeneous) |
| **D-12 / D-34** | transmission = `random.choice([18.5,19.2,20,21.5,22])/1000`, independent of payload and bandwidth |
| **D-18** | phase 2 and phase 7 each draw their own propagation/transmission |
| **D-23** | queue wait = wall-clock elapsed |

It deliberately does **not** restore the reward-side repairs (D-04…D-07, D-16,
D-20, B3, B4). Those change only the *SafeTail 2.0 learned policy* — they never
change the latency a request experiences — so they cannot affect a baseline
comparison. The SafeTail 1.0 policy uses its own τ-referenced reward regardless.

### Legacy mode was validated, not assumed

A legacy-mode run of the native 2.0 policy was compared against `reference_v0`:

| component | reference_v0 (original code) | legacy mode (this code) |
|---|---|---|
| transmission mean | 20.197 ms, **5 distinct values** | 20.179 ms, **5 distinct values** |
| computation mean | 8.638 ms | 8.863 ms |

Transmission is reproduced exactly (same 5-value discrete distribution).
Computation matches to ~2.6%, which is contention-profile noise, not a model
difference. Propagation differs because the *policy* selects different servers —
that is the thing being measured, not a confound.

## Configuration

```
requests            15,225 (3,045 chunks × 5)     — matches reference_v0 exactly
episodes            1,015 (chunks_per_episode=3)
seed                SAFETAIL_SEED=0
mode                SAFETAIL_LEGACY_ENV=1
runner              src/main.py --run  (socket-free, in-process)
SafeTail 1.0 τ      per request type, from reference_v0 service medians:
                    s 51.8 ms · d 34.5 ms · p 35.5 ms   (decision 13.1(1))
SafeTail 1.0        trains online across the whole run (matching how the
                    reference_v0 2.0 run was produced — no train/freeze split)
```

Reproduce:

```bash
SAFETAIL_LEGACY_ENV=1 SAFETAIL_SEED=0 python baselines/run_baseline.py \
    --policy safetail_v1 --run --chunks 3045 --episodes 1015 \
    --out results/v1_legacy_s0 --label v1_s0
python tools/make_figures.py
```

## The metric

`total_latency` = computation + propagation + transmission, in ms — **the same
column, with the same meaning, as in `reference_v0`**. This is *service*
latency; it excludes queueing (D-11). `end_to_end_latency` (= service + queue
wait) is also logged in the new runs and can be plotted with
`--metric end_to_end_latency`, but `reference_v0` has no such column, so the
headline comparison uses `total_latency`. Decision 13.1(2).

All policies are truncated to a **common request count** before percentiles are
taken, so no policy can win on sample size.

## Known limitations — state these if the numbers are used

1. **Arrival timing differs.** `reference_v0` was produced through the TCP
   sender with `random.uniform(0.2,0.8)s` gaps between bursts; the new runs feed
   chunks back-to-back in-process. Servers therefore drain slightly differently,
   which changes the contention strings and hence computation delay. Measured
   impact: computation mean 8.638 → 8.863 ms (**+2.6%**). Small, but it is not
   zero, and it is the main reason this is a *comparison* rather than a
   controlled experiment.
2. **One seed.** These are single runs (`seed=0`). No error bars. plan.md B7
   wants ≥3 seeds before anything is published.
3. **The heterogeneous side is unfixed code.** Every defect in `plan.md` §4 is
   still present in the `reference_v0` numbers — including **D-02**, which means
   the "heterogeneous" system's computation path was not actually heterogeneous.
   A fair reading is: *this compares SafeTail 1.0 against SafeTail 2.0 as it was
   actually run*, not against SafeTail 2.0 as designed.
4. **`reference_v0`'s `request_type` column holds the mutated contention string**
   (D-19 was not fixed then), so per-type panels key on its first character.
5. The K-baselines are **fixed-K redundant dispatchers**, not budget-neutral
   candidate pre-filters (S-11), and SafeTail 2.0 ran free over all 31 subsets
   at mean K ≈ 3.55 while they were capped at K ∈ {1,2,3} — the comparison is
   **not replication-budget controlled** (D-09).
