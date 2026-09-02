## Summary

Audit-driven bug fixes to SafeTail 2.0's `src/`. Every change carries a
`[SAFETAIL][<component>][<kind>][D-xx]` tag at the fix site, matching the
defect IDs from the codebase-analysis report.

**Scope of this PR:** `src/` correctness only — plus `requirements.txt`
(`D-01`), `README.md` / `run_all_baselines.sh` (`D-13` / `D-14`). No new
top-level directories. The audit write-up, the verification gate scripts, the
regression-test suite, and the SafeTail-1.0 baseline seam live in a separate
working tree and are **not** part of this PR.

**Backwards compatibility:** every new toggle defaults to the pre-fix
behaviour — `REWARD_MODE="headroom"` and `C_RED=0.0` reproduce the old reward
byte-for-byte; `POLICY="native"` keeps the existing scheduler; `--smoke` is
opt-in. The one unavoidable behavioural change is `D-02` (see below): the
computation-latency predictions are now per-server instead of server 1's for
everyone, which moves every downstream number.

## Correctness fixes

| ID | File(s) | What was wrong → fix |
|---|---|---|
| **D-01** | `requirements.txt` | UTF-16LE/CRLF and **no scikit-learn**, so a clean install per the repo's instructions could not unpickle `models/*/*.pkl` and the failure was swallowed (`D-02c`). Re-encoded UTF-8/LF; added `scikit-learn==1.4.2` (the version the pickles were saved with), `scipy`, `seaborn`, `joblib`, `threadpoolctl`. |
| **D-02 / D-02b / D-02c** | `src/regressors.py` (new), `src/servers.py`, `src/_legacy_regressor_wrappers/` (moved) | All 15 per-server wrappers hardcoded `models/server1/` + `dataset/server1.csv`; identical module names made `sys.modules` hand back server 1's class regardless of `sys.path`; and every load failure fell through to a **contention-free single-letter CSV lookup**. Replaced by one `TracePredictor(server_index, task)` — server *i* loads `models/server{i}/` + `dataset/server{i}.csv`, ports the three feature schemas the shipped models actually use (selected by each bundle's `feature_columns`), and **raises** on a load failure or a missing contention row unless `constants.ALLOW_DEGRADED_PREDICTORS` (default `False`). server 2 aliases server 1 (byte-identical CSV; its model needs columns that CSV lacks). Mean computation delay is now server1/2 ≈ 12 ms · server3 ≈ 388 ms · server4 ≈ 266 ms · server5 ≈ 13 ms (was ≈ 12 ms for all five). |
| **D-04 / D-05 / D-06** | `src/controller.py`, `src/agent.py` | Replay stored the **live `Request`** and flattened it only at episode end, by which point it held the realised latencies / contention string / queue-wait of its own action; `state` and `next_state` were the same object; and every step reward was overwritten with the single episodic reward. Now `s_t` is snapshotted as an array **at action time** (pre-schedule), `s_{t+1}` after; `DQNAgent.store_arrays()` takes the arrays; the per-step reward is kept and the episodic reward is added as a broadcast `R_ep/N` bonus. |
| **D-07** | `src/controller.py` (`_collapse_step_reward`) | Step reward was `np.mean()` over all 6 `server_dict` slots → every extra server added a non-negative term to a fixed denominator; **nothing priced redundancy**. Now: mean over the **selected** servers, minus `c_red·(|A|−1)/(β−1)` (`constants.C_RED`, default `0.0`; `C_RED_SWEEP` for an ablation). |
| **D-16** | `src/controller.py` (`compute_step_reward`) | CPU-only servers (no GPU columns → servers 3, 4) got `(1−0)(1−0)=1` for the two GPU headroom factors, structurally inflating their reward. GPU factors are now **dropped** for such servers and the product renormalised as a geometric mean of the available factors. Range stays `[0, log 2]`; docstring corrected (`D-27`). |
| **D-17** | `src/controller.py` | Failure path wrote `rewards[server_idx − 1] = 0.0` — corrupted a neighbour's slot, left server 0 at its initialised value. Now `rewards[server_idx]`. |
| **D-19** | `src/user.py`, `src/servers.py`, `src/controller.py` | `schedule_request()` overwrote `request.combination` with the contention string (`"d"` → `"dps"`), so `latency_log.csv`'s `request_type` column held 192 distinct values. New immutable `Request.contention_str`; `combination` stays the type letter; `latency_log.csv` gains a `contention_str` column. |
| **D-20** | `src/controller.py` | `request_*_done` counters were incremented **unconditionally**, so the completion ratio in the episodic wait-time denominator was identically `1.0`. Now incremented only when `T ≤ D2`, at the one site `T` is known. |
| **D-21** | `src/controller.py` (`process_step`) | Unbounded self-recursion under saturation (`sleep 0.1` + recurse). Replaced with a bounded retry loop (6×, exponential backoff, cap 0.8 s) + an explicit drop and `Controller.dropped_requests` counter. |
| **D-28** | `src/controller.py` | Deleted `assign_request()` — unpacked 3 values from `schedule_request()`'s 7-tuple, raised on any call, unreachable. `dispatch_to_agent()` (unused) removed with it. |
| **D-31** | `src/receiver.py` | `np.load(allow_pickle=True)` on socket payloads = RCE if the port ever leaves localhost. Now refuses any non-loopback bind and caps the payload at 8 MiB. |

## Documentation / hygiene

| ID | File | What |
|---|---|---|
| **D-13** | `README.md` | The "Queueing Model" section claimed an **M/M/1 queue** with `W = λ/(μ(μ−λ))`. There is no queue, no `λ`, no `μ`. Replaced with the reality — an `M/M/c/c` Erlang-B **loss** system (`c = 4`, rejects when full), **uniform** (not Poisson) arrivals, wall-clock wait time. Correction banner added at the top. |
| **D-14** | `run_all_baselines.sh` | `$SAFETAIL_PY` (defaults to a repo venv) instead of a hardcoded `/home/jyoti/miniconda3/bin/python`; `K` ranges `1..β`; a seed loop; output to `results/<mode>_k<K>_s<seed>_<sha>/`. |
| **D-29 / D-30** | `src/controller.py`, `src/agent.py`, `src/constants.py` | Dead methods / constants tagged `[SAFETAIL][DEAD][D-xx]` with the reason (`get_queue_lengths`, `save_checkpoint`, `export_training_data`, `generate_testing_plots`, `DQNAgent.get_min_delay`; `nS`, `episode_size`, `max_load`). |
| **D-33 / D-35** | `src/constants.py` | Annotations: the run processes **~15,225** requests, not `total_no_request` (500,000); `nS` is dead — the encoder input is variable-length from hardware heterogeneity, not a fixed `(β+1)` vector. |

## New capability (small; needed to run / verify the above)

- `src/constants.py`: `SMOKE` / `SEED` / `POLICY` / `REWARD_MODE` / `C_RED` / `MATCH_K` / `TAU_BY_TYPE` / `ALLOW_DEGRADED_PREDICTORS` — all read from env with backwards-compatible defaults.
- `src/main.py`: `--smoke` — a short, socket-free, seeded run (~20 episodes, ~300 requests, no plots) that feeds chunks straight to `Controller.send_to_server`. Also forces UTF-8 on `stdout`/`stderr` so the emoji `print()`s don't crash a Windows console (`D-36`).
- `src/_seeding.py`: `seed_everything()` → `random` / `numpy` / `tensorflow`.
- `src/rewards.py`: `tau_reward_5case()` — SafeTail 1.0's τ-referenced 5-case tail-latency reward, selectable via `REWARD_MODE ∈ {headroom, tau, headroom+tau}`. This is the only place tail latency enters the 2.0 reward (`M-03`); `"headroom"` (default) is unchanged.
- `src/policy_registry.py`, `src/_safetail_log.py`: stdlib-only support modules (a subset↔action-index helper used by the reward/mode changes; a tagged-logging + degradation-ledger helper). Inert unless used.

## How to verify

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # or bin/pip on POSIX
SAFETAIL_SMOKE=1 SAFETAIL_SEED=0 python src/main.py --smoke
```

Expect: 20 episodes, ~300 requests, exit 0, no traceback; per-server
computation delays visibly differ (server 3/4 ≈ 20–30× server 1/5).

## Not in this PR (separate working tree)

Verification gate scripts (`tools/verify_*.py`, `tools/audit_*.py`), the
regression suite, the pluggable-policy seam + SafeTail-1.0 baseline, the
per-defect `CHANGELOG.md`, and the specification-errata register
(`audit/ERRATA.md`, HED-vs-BTP corrections). Happy to open follow-ups.
