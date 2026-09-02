# SafeTail 1.0 baseline vs. the heterogeneous SafeTail 2.0 — comparison report

**Question.** Run the original SafeTail (1.0) over the *same data* the heterogeneous
SafeTail 2.0 servers were given, and compare against the existing heterogeneous results.

**Data compared.**
`SafeTail-2.0-main/SafeTail-2.0-main/results/safetail_training_logs/` — verified
**byte-identical** (md5, all five CSVs) to `results/reference_v0/safetail_training_logs/`,
so the heterogeneous side is exactly the published run.
SafeTail 1.0 side: 3 seeded runs, `results/safetail_v1_legacy_s{0,1,2}/`.

**Method.** `baselines/safetail_v1/` is a **port**: the SafeTail 1.0 *algorithm*
(flat homogeneous state, sigmoid×2+BN network with a softmax head and
categorical-crossentropy loss, the τ-referenced 5-case reward, ε-greedy replay)
driving the SafeTail **2.0** environment, under `SAFETAIL_LEGACY_ENV=1` so the
latency physics match the code that produced the heterogeneous results.
15,225 requests / 1,015 episodes per run, identical to the reference. All
policies truncated to a common **n = 14,723** before percentiles.

---

## 1. Headline

Service latency `total_latency` = computation + propagation + transmission (ms);
the same column with the same meaning on both sides. SafeTail 1.0 is the median
of 3 seeds.

| policy | p50 | p90 | p95 | p99 | mean | mean K |
|---|---|---|---|---|---|---|
| **SafeTail 2.0** (heterogeneous) | **39.88** | **59.88** | **75.66** | **115.64** | **44.44** | **3.55** |
| **SafeTail 1.0** (baseline) | 41.67 | 69.48 | 89.72 | 122.59 | 45.37 | **2.55** |
| *2.0 better by* | 4.3% | 13.8% | **15.7%** | 5.7% | 2.0% | *(uses 39% more compute)* |

**SafeTail 2.0 beats SafeTail 1.0 at every percentile**, with the largest margin
at p90–p95 — the tail the paper's claim is about.

### ⚠️ But the comparison is not compute-matched

SafeTail 2.0 dispatches each request to **3.55 servers on average**; SafeTail 1.0
to **2.55**. 2.0 buys its tail-latency win with **≈39% more compute**. This is
defect **D-09** (the evaluation is not replication-budget controlled) applied to
this comparison, and it is not a small effect — redundancy is the main lever on
tail latency in this system.

The asymmetry is not accidental. SafeTail 1.0's τ-referenced reward explicitly
prices redundancy: when a request finishes *early* the penalty scales with how
**many** servers were used. SafeTail 2.0's resource-headroom reward has **no term
that decreases in |A|** (defect **D-07**), so it drifts upward to K≈3.55 (**D-08**).
*The 2.0 policy is not choosing to spend more; nothing stops it.*

An approximate budget-matched read, using the shipped fixed-K heuristics:

| policy | mean K | p95 |
|---|---|---|
| SafeTail 1.0 | 2.55 | 89.72 |
| MinProp-2 | 2 | **49.08** |
| SafeTail 2.0 | 3.55 | 75.66 |
| MinProp-3 | 3 | **49.18** |
| MinLoad-3 | 3 | 54.29 |

---

## 2. The result that matters more

**Both learned schedulers lose decisively to a trivial propagation heuristic.**

| policy | p50 | p90 | p95 | p99 | mean K |
|---|---|---|---|---|---|
| Oracle (all 5, take min) | 30.42 | 47.77 | 49.53 | **52.45** | 5.00 |
| **MinProp-2** | 29.93 | 46.63 | 49.08 | **53.37** | 2.00 |
| MinProp-3 | 29.88 | 46.68 | 49.18 | 56.31 | 3.00 |
| MinLoad-3 | 31.88 | 49.93 | 54.29 | 81.67 | 3.00 |
| SafeTail 2.0 | 39.88 | 59.88 | 75.66 | 115.64 | 3.55 |
| SafeTail 1.0 | 41.67 | 69.48 | 89.72 | 122.59 | 2.55 |

**MinProp-2 — pick the two lowest-propagation servers — is within 2% of the
Oracle at p99 while using 2 servers instead of 5**, and beats SafeTail 2.0 by
**54%** at p99 (53.37 vs 115.64). This reproduces audit finding **D-10** exactly.

The mechanism is visible in the latency decomposition:

| policy | computation | propagation | transmission |
|---|---|---|---|
| Oracle | 8.85 ms | **5.86 ms** | 20.11 ms |
| SafeTail 2.0 | 8.64 ms | **15.52 ms** | 20.20 ms |
| SafeTail 1.0 | 9.40 ms | **15.89 ms** | 20.20 ms |

Propagation is the only strongly server-discriminating signal, and **neither
learned policy exploits it** — both sit at ~15.5–15.9 ms while the oracle reaches
5.86 ms. Meanwhile transmission (20.2 ms, a 5-value coin flip in this
configuration — **D-12**) is ~45% of the metric and *no policy can influence it*.

Root cause is **D-02**: in the code that produced these results, all 15 regressor
wrappers loaded `models/server1/` + `dataset/server1.csv`, so computation delay
was identical across servers. With computation carrying no signal and
transmission being noise, propagation was the *only* thing left to learn — and
MinProp optimises it directly, by construction.

---

## 3. Is SafeTail 1.0 under-trained?

No — it is **unstable**, and the two are distinguishable from the data.

**Training budget** (identical data on both sides, 1,015 episodes / 15,225 requests):

| | replay cadence | gradient updates |
|---|---|---|
| SafeTail 2.0 | once per **episode**, `epochs=1` | **4,024** |
| SafeTail 1.0 | once per **request**, `epochs=2` | **120,776** |

SafeTail 1.0 already receives **30× more gradient updates**. Its ε reached the
0.1 floor in all three seeds (15,097 replays each; overall explore fraction 31%),
so it spent the majority of the run exploiting a trained policy.

**p95 within each decile of the run** (figure `F5_convergence`):

| | d1 | d3 | d5 | d7 | d8 | d9 | d10 |
|---|---|---|---|---|---|---|---|
| SafeTail 2.0 | 109.3 | 87.7 | 63.2 | 62.0 | 64.1 | 63.8 | **62.3** |
| SafeTail 1.0 s0 | 103.2 | 91.3 | 84.2 | **80.3** | 84.4 | 87.9 | 84.9 |
| SafeTail 1.0 s2 | 104.3 | 102.8 | 74.6 | **74.2** | 77.3 | 85.2 | 78.3 |

SafeTail 2.0 converges by decile 5 and holds flat. SafeTail 1.0 reaches its best
around decile 7 and then **degrades** on every seed. More training makes it worse.

**Probable cause — architectural, not budgetary.** SafeTail 1.0's network ends in
`softmax` + `categorical_crossentropy` (kept verbatim; changing it would make this
"SafeTail 1.5"). Softmax forces the 31 action-values to sum to 1, and CCE expects
a probability-distribution target — but the Bellman targets are negative real
Q-values. The head **structurally cannot represent** the quantity being
regressed, so additional updates drift rather than converge.

If that holds, it is the most interesting result here: **SafeTail 2.0's switch to
a linear head with MSE loss is not incidental — it is what makes the Q-learning
work at all.**

> **Status:** a 3× budget run (45,675 requests, 3,045 episodes, ε schedule
> rescaled, both policies, 3 seeds each) is in progress to test this directly.
> If 1.0 improves, the budget explanation wins; if it degrades or plateaus above
> 2.0, the architectural explanation does. **This section will be updated with
> that evidence.**

---

## 4. Secondary observations

* **Deadline satisfaction is 100% for every policy** and is therefore useless as
  a discriminator here: D2 is 200 ms (d/p) / 400 ms (s), while p99 service
  latency is 50–123 ms. Any claim resting on deadline satisfaction in this
  configuration is uninformative.
* **The Oracle is not a clean lower bound.** Dispatching to all 5 servers
  congests them, so MinProp-2 (K=2) actually edges it at p50 (29.93 vs 30.42).
  Redundancy has a real cost that an "all servers" oracle hides.
* **SafeTail 2.0 seed instability.** A re-run of 2.0 under identical conditions
  gave p50 = 76.14 on seed 0 versus 31.84 / 31.52 on seeds 1–2 — a >2× spread.
  Single-seed results for 2.0 are not trustworthy.
* **Fixing 2.0's reward helps the median, not the tail.** The re-run (which
  carries the D-04…D-07 / D-16 / D-20 repairs) improves p50 39.88 → 31.84 and
  mean 44.44 → 39.60, but p99 is unchanged-to-worse (115.64 → 119.63).

---

## 5. What can and cannot be claimed

**Supported.**
1. SafeTail 2.0 beats SafeTail 1.0 at every percentile on identical data
   (p95: 75.66 vs 89.72, −15.7%).
2. SafeTail 1.0 is not under-trained at this budget; it is unstable, and the
   likely cause is its softmax+CCE head.
3. Both learned policies lose heavily to MinProp at every percentile.

**Not supported.**
1. That 2.0's advantage is a *scheduling-quality* result — it is confounded with
   a 39% larger replication budget (**D-09**). A K-matched comparison is required.
2. Any tail-latency claim over MinProp (**D-10**, **S-17**). BTP §6.2's headline
   ("SafeTail has lower latency than MinLoad, MinProp, and Random at all
   percentiles") is **false on this data**.
3. Anything about heterogeneity-aware scheduling from these numbers: **D-02**
   means computation delay was identical across all five servers in the code that
   produced them.

---

## 6. Recommended next steps

1. **Re-run the comparison K-matched** — `SAFETAIL_MATCH_K=2.55` caps 2.0 to
   1.0's budget. Until then the headline is confounded. *(Implemented, not yet run.)*
2. **Re-run on the D-02-fixed environment**, where computation is genuinely
   per-server (server 3 ≈ 388 ms vs server 1 ≈ 12 ms). Only there does
   "heterogeneity-aware scheduling" mean anything, and it is the one condition
   under which a learned policy could plausibly beat MinProp.
3. **Decide the MinProp story with your advisor before writing up.** An honest
   negative result — *a learned scheduler does not beat a propagation heuristic
   in this environment, and here is why* — is publishable and is supported by
   the decomposition in §2. Claiming a win is not supported.

---

## 7. Reproduce

```bash
SAFETAIL_LEGACY_ENV=1 python tools/run_matrix.py \
    --policies safetail_v1 oracle native --seeds 0 1 2 \
    --chunks 3045 --episodes 1015 --legacy
python tools/make_figures.py
python tools/verify_figures.py       # gate G6
python tools/package_baseline.py
```

Figures: `F1` tail bars · `F2` CCDF · `F5` convergence · `F6` per-request-type ·
`F7` decomposition. Each ships a companion `.csv` of the exact numbers plotted.
Limitations and provenance: `RUN_PROVENANCE.md`. Every deviation from SafeTail
1.0: `FAITHFULNESS_REGISTER.md`.
