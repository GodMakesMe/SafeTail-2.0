# SafeTail 1.0 baseline (`safetail_v1`) — faithfulness register

This is a **port** of the SafeTail 1.0 *algorithm* onto the SafeTail 2.0
*environment*, driven through the seam (`src/policy_registry.Policy`). It never
imports `src/agent`, never loads 1.0's pickles, never touches `_spec_source/`.
Spec: `plan.md` §14.1. Frozen source to read while porting:
`baselines/safetail_v1/_spec_source/`.

Run it:

```bash
python baselines/run_baseline.py --policy safetail_v1 --smoke
POLICY=safetail_v1 SAFETAIL_SEED=1 python baselines/run_baseline.py --policy safetail_v1
```

## Every deviation from SafeTail 1.0 (plan.md §8.6)

| 1.0 element | Port | Rationale |
|---|---|---|
| 5-case τ reward (Eq. 5) | **kept exactly** — `reward_v1.py` | it is the baseline's identity; also the reference impl for B4 (M-03) |
| `softmax` head + `categorical_crossentropy` | **kept** (`model_v1.py`) | faithful even though wrong for Q-regression; changing it → "SafeTail 1.5" |
| 2 hidden layers, sigmoid, BN | **kept** | ST §IV |
| `2^β − 1` subset action space (β=5 ⇒ 31) | **kept** | matches 2.0 (`policy_registry.index_to_subset`) |
| ε-greedy, `ε -= gamma_decay` per replay | **kept** (`train_v1.py`) | ST Eq. 3, subtractive |
| replay `deque(maxlen=2500)`, batch 128, epochs 2, val_split 0.2 | **kept** | 1.0 `instance` values |
| no target network (max over online net) | **kept** | faithful |
| `experience_replay` called **inside** `reward()` | **changed** → called from `finish_episode()` | 1.0's placement is a structural accident; keeping it would couple the baseline to 2.0's reward call sites |
| `reward()` returns `None` when `abs(λ) ≥ 1000` | **fixed** → returns `0.0`, counted as `V1_BUG_01_out_of_band` | latent 1.0 bug: `None` reached the replay buffer. **V1-BUG-01** |
| `nS = 2β + 3` (yolo) | **changed to `2β + 2`** | **A-4** — no RESOLUTION attribute in 2.0; matches 1.0's `instance`/`noise` shape |
| `LOG_COMPRESS`: `node_latency = log(1+node_latency)` before the min | **dropped** (raw latency; `LOG_COMPRESS=0`) | **V1-DEV-01** — changes τ's meaning; τ set for raw latency. Flip `SAFETAIL_V1_LOG_COMPRESS=1` to study. |
| reward graded on **realised** min latency | **changed** → graded on `min(ctx.est_delay over A)` | **V1-DEV-02** — the 2.0 seam deliberately exposes only the pre-action estimate (the single D-18 draw) so no policy can see its own outcome |
| task-specific MLP latency regressors | **dropped** | the 2.0 environment provides latency; a baseline carrying its own simulator makes the comparison meaningless |
| homogeneous `LOAD` abstraction | **kept** — **A-1** | this *is* the thing being tested. `LOAD[i] = clip(MAX_LOAD − free_slots[i], 1, MAX_LOAD)`. Never the heterogeneous hardware vector — giving 1.0 more information than it had would flatter the baseline. |
| `PROPOGATION[i] = random.choice(ping_data[...])` | **A-2** → `ctx.est_components[i].prop` | the same single draw the environment used (D-18) |
| `MESSAGE_SIZE`, `BANDWIDTH` per request | **A-3** → `ctx.message_size`, `ctx.bandwidth` | constant today (D-34); meaningful after B5 |
| `RESOLUTION` | **A-4** → omitted | 2.0 has no such attribute |
| τ = per-task constant (yolo 1.2 s, instance 2.5 s, noise 0.3 s) | **changed** → per **request type** from `reference_v0` service-latency medians: `s 51.8 ms · d 34.5 ms · p 35.5 ms` | a single global τ is meaningless — types differ by >1 order of magnitude (S-14). Override `SAFETAIL_V1_TAU_{S,D,P}`. Documented in `config_v1.py`; **decision 13.1(1)**. |
| online training for the whole run | **configurable** — `SAFETAIL_V1_FREEZE_AFTER=<ep>` freezes to argmax-only | **decision 13.1(4)** train-then-freeze, so v1 and 2.0 are both evaluated in inference mode |

## Isolation

Deleting `baselines/` must leave a working repo (R3, gate **G4**). This package
is imported only via `baselines/register.py` → `src/policy_registry`. `src/`
never imports it.
