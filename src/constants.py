import os


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


############################### RUN CONTROL (plan.md 9.4 / B7) ###########

# [SAFETAIL][MAIN][smoke] Short deterministic run for gates G2/G3/G4 and CI.
# Enabled by `--smoke` (main.py sets SAFETAIL_SMOKE=1) or the env var directly.
SMOKE = _env_flag("SAFETAIL_SMOKE")

# [SAFETAIL][MAIN][B7] Master seed threaded to random / numpy / tensorflow.
# `None` (env unset / "none") keeps legacy non-deterministic behaviour.
_seed_raw = os.environ.get("SAFETAIL_SEED", "").strip().lower()
SEED = None if _seed_raw in ("", "none") else _env_int("SAFETAIL_SEED", 0)

# [SAFETAIL][SEAM] External-policy selector (plan.md 8.3). "native" => built-in
# DQN / heuristic path; any other value is resolved via src/policy_registry.get().
POLICY = os.environ.get("POLICY", "native").strip() or "native"

# Smoke-scale overrides: ~20 episodes, ~300 requests, no plots.
SMOKE_CHUNKS = _env_int("SAFETAIL_SMOKE_CHUNKS", 60)   # 60 * chunk_size(5) = 300 requests
SMOKE_EPISODES = _env_int("SAFETAIL_SMOKE_EPISODES", 20)

# [SAFETAIL][REGRESSOR][D-02c] When False (default), a regressor that fails to
# load or a missing contention trace row is a HARD error -- the collapse is no
# longer silent. When True, degradation is allowed but logged [DEGRADED][D-02c]
# and counted in the run manifest; a run with any [DEGRADED] count > 0 is not
# publishable (gate G5).
ALLOW_DEGRADED_PREDICTORS = _env_flag("SAFETAIL_ALLOW_DEGRADED_PREDICTORS")

# [SAFETAIL][REWARD][B3][M-04][S-02] Redundancy price. The step reward gets an
# explicit cost  c_red * (|A|-1)/(beta-1)  subtracted, so replicating to more
# servers is no longer free (root cause of D-07 over-replication). c_red = 0.0
# reproduces the pre-B3 (un-priced) reward exactly.
# Source: "chosen, see plan.md B3". Unit: reward points (reward is in [0, log2]).
# SWEEP for the publishable K-vs-c_red ablation (plan.md B3 accept / figure F8):
C_RED = float(os.environ.get("SAFETAIL_C_RED", "0.0"))
C_RED_SWEEP = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]

# [SAFETAIL][CONTROLLER][B6][D-09] Budget-controlled evaluation. When set (e.g.
# 3.0), SafeTail's per-request subset is CAPPED (keeping the lowest-estimate
# servers) so its running mean |A| never exceeds this target -- the headline
# comparison then runs at a stated replication budget instead of SafeTail
# running free over all 31 subsets at mean K ~ 3.55 while the baselines are
# capped at K in {1,2,3}. The realised mean K is reported alongside (plan 9.3).
# None => SafeTail unconstrained.
_mk = os.environ.get("SAFETAIL_MATCH_K", "").strip()
MATCH_K = float(_mk) if _mk else None

# [SAFETAIL][REWARD][B4][M-03] Which step reward the 2.0 agent optimises:
#   "headroom"       -- BTP's resource-headroom product + B3 c_red cost (default;
#                       reproduces the pre-B4 reward exactly)
#   "tau"            -- SafeTail 1.0's tau-referenced 5-case tail-latency reward
#                       ONLY (src/rewards.tau_reward_5case)
#   "headroom+tau"   -- their sum
# M-03: tail latency -- the stated objective of ST/HED/BTP -- otherwise never
# appears in the 2.0 reward.
REWARD_MODE = os.environ.get("SAFETAIL_REWARD_MODE", "headroom").strip() or "headroom"

# tau per request type (seconds). Median service latency (comp+prop+trans) per
# type from results/reference_v0/safetail_training_logs/latency_log.csv
# (d 34.5 ms, p 35.5 ms, s 51.8 ms). Same source as baselines/safetail_v1
# config; a single global tau is meaningless (S-14). plan.md 13.1(1).
TAU_BY_TYPE = {
    "s": float(os.environ.get("SAFETAIL_TAU_S", "0.0518")),
    "d": float(os.environ.get("SAFETAIL_TAU_D", "0.0345")),
    "p": float(os.environ.get("SAFETAIL_TAU_P", "0.0355")),
}

############################### HYPERPARAMETERS ##########################
median_computation_delay = 0.05
# [SAFETAIL][DEAD][D-33] total_no_request only SIZES the pre-generated request
# pool. The run stops after `no_of_burst` bursts -> ~15,225 requests actually
# processed, not 500,000 (BTP Table 4.2 overstates by ~33x).
total_no_request = 500000
chunk_size = 5
no_of_chunk = int(total_no_request / chunk_size)
# [SAFETAIL][D-30][D-43] episode_size is vestigial: the controller's episode
# length is `chunks_per_episode = 3`, not this.
episode_size = 4
no_of_burst = 1000
# Burst shape. Defined HERE, above the no_of_episodes derivation that reads
# max_burst -- it used to be declared further down, so the derivation hardcoded
# a literal 4 and changing max_burst silently desynchronised the episode target.
# Flagged by the Claude Code verification session, 8 Sep 2026.
min_burst = 2
max_burst = 4

# [SAFETAIL][MAIN][FIX][D-43] `no_of_episodes` is the run's termination target,
# and it used to be `no_of_chunk / episode_size` = 100000/4 = 25,000 -- wrong
# twice over:
#   * it divides by episode_size (4) while the controller ends an episode every
#     chunks_per_episode (3) chunks, and
#   * it counts chunks the sender can never deliver: the run stops after
#     no_of_burst bursts of at most max_burst chunks, i.e. 1000*4 = 4,000 chunks,
#     not 100,000.
# The target was therefore unreachable by ~19x, which is why the socket entry
# point blocked forever (D-42).
# It is now DERIVED from what can actually be delivered. Override with
# SAFETAIL_NO_OF_EPISODES if you want the old (unreachable) behaviour back.
# Raised independently by the external defect dossier (Uttam) as its D-14.
CHUNKS_PER_EPISODE = _env_int("SAFETAIL_CHUNKS_PER_EPISODE", 3)
_deliverable_chunks = min(no_of_chunk, no_of_burst * max_burst)
no_of_episodes = _env_int("SAFETAIL_NO_OF_EPISODES",
                          max(1, _deliverable_chunks // max(1, CHUNKS_PER_EPISODE)))
learning_rate = 1e-6
# [SAFETAIL][AGENT][FIX][S-16] epsilon schedule -- ONE statement, correctly named.
# BTP 4.4 states a MULTIPLICATIVE decay `eps <- max(eps_min, eps*(1-gamma_eps))`;
# the code has always done a SUBTRACTIVE step `eps -= gamma_decay`; and the
# constant was named `gamma_decay`, which matches neither.
# DECISION (plan.md 13.1): keep the SUBTRACTIVE step. It is what produced every
# run in results/, so switching to multiplicative would silently invalidate the
# whole comparison for a cosmetic agreement with the report. The report text is
# what is wrong (audit/ERRATA.md S-16), not the code.
# `epsilon_decay_step` is the name that describes it. `gamma_decay` is kept as a
# deprecated alias so external scripts and baselines/safetail_v1 keep working.
epsilon_decay_step = 0.002        # subtractive: eps <- max(eps_min, eps - this)
gamma_decay = epsilon_decay_step  # [DEPRECATED][S-16] alias; use epsilon_decay_step
epsilon_min = 0.1
# (min_burst / max_burst are declared above, before the no_of_episodes derivation)
min_interval = 0.2
max_interval = 0.8
jitter = 0.02
lr_decay_rate = 0.999
lr_min = 1e-5
epochs = 1
# [SAFETAIL][DEAD][D-30] max_load is unused; servers.py hardcodes
# MAX_CONCURRENT_REQUESTS = 4.
max_load = 5
beta = 5  # Number of edge servers.
alpha = 0.005  # Reward scaling factor.
discount_rate = 0.9
batch_size = 128
# [SAFETAIL][DEAD][D-30][D-35] nS is dead: the encoder input is shape=(None,1)
# (variable-length, per S-10 -- hardware heterogeneity, not temporal). The agent
# stores `states=constants.nS` but never uses it. Kept only so the existing
# DQNAgent(states=constants.nS) call still resolves.
nS = 1 * beta + 1
nA = 2 ** beta - 1  # Number of possible actions (subsets of servers).
post_epsilon_steps = 8000

############################### LEGACY ENVIRONMENT #######################
# [SAFETAIL][LEGACY] Reproduce the PRE-FIX environment physics, so a policy can
# be evaluated under exactly the conditions that produced
# results/reference_v0/ (and the heterogeneous results the mentor already has).
#
# SAFETAIL_LEGACY_ENV=1 restores the three fixes that changed the LATENCY MODEL:
#   D-02  every server predicts computation with models/server1 + server1.csv
#   D-12/D-34  transmission = random.choice([18.5,19.2,20,21.5,22])/1000,
#              independent of payload size and bandwidth
#   D-18  phase 2 and phase 7 each draw their own propagation/transmission
#   D-23  queue wait = wall-clock elapsed (not the simulated quantity)
#
# It deliberately does NOT restore the reward-side fixes (D-04..D-07, D-16,
# D-20, B3, B4): those change only the SafeTail-2.0 learned policy, never the
# latency a request experiences, so they cannot affect a baseline comparison.
LEGACY_ENV = _env_flag("SAFETAIL_LEGACY_ENV")

LEGACY_REGRESSORS = _env_flag("SAFETAIL_LEGACY_REGRESSORS") or LEGACY_ENV   # D-02
LEGACY_TRANSMISSION = _env_flag("SAFETAIL_LEGACY_TRANSMISSION") or LEGACY_ENV  # D-12/D-34
LEGACY_DOUBLE_DRAW = _env_flag("SAFETAIL_LEGACY_DOUBLE_DRAW") or LEGACY_ENV  # D-18
LEGACY_QUEUE_WAIT = _env_flag("SAFETAIL_LEGACY_QUEUE_WAIT") or LEGACY_ENV    # D-23

############################### METRIC SEMANTICS (B5) ####################
# [SAFETAIL][METRIC][B5] Decisions 13.1(2) and D-11/D-12/D-18/D-23/D-34.

# D-11: which latency the figures plot. "service" = computation+propagation+
# transmission (the quantity the min-over-subset and the reward effectively
# optimise). "end_to_end" = service + simulated queue wait. Both are always
# logged to latency_log.csv; this only selects the headline column.
# Consumed by tools/make_figures.py (_default_metric); --metric overrides it.
LATENCY_METRIC = os.environ.get("SAFETAIL_LATENCY_METRIC", "service").strip() or "service"

# D-12 / D-34: transmission delay is now f(message_size, bandwidth), not a
# 5-value coin flip. Model (SafeTail 1.0, ST IV): t = 8*KB/up_kbps + 8*KB/dn_kbps
# with up = dn = bandwidth. Plus a small per-server multiplicative link jitter.
DEFAULT_MESSAGE_SIZE_KB = 1024      # fallback when a request carries no size
DEFAULT_BANDWIDTH_MBPS = 20
LINK_JITTER_FRAC = float(os.environ.get("SAFETAIL_LINK_JITTER", "0.05"))

# D-34: real per-request payload-size variation (KB), by request type. Speech
# (audio clip) payloads are larger than a single vision frame.
# Calibrated so transmission stays the same ORDER as the legacy constant (~20 ms
# at 20 Mbps two-way) while now varying with type and payload:
#   s 16-64 KB -> 12.8-51 ms | d,p 4-24 KB -> 3.2-19 ms
MESSAGE_SIZE_KB_BY_TYPE = {"s": (16, 64), "d": (4, 24), "p": (4, 24)}

# [SAFETAIL][REGRESSOR][D-40] Which measurement to use when a trace carries more
# than one row for the same contention string.
#   "sample" -- draw uniformly among the repeats (DEFAULT). This is how measured
#               computation-time variance enters the simulator. On the current
#               dataset -- 363 rows / 363 unique combinations / Iteration == [1],
#               i.e. exactly one measurement per scenario, already averaged over
#               ~500 files at collection time -- it is a no-op, so today's numbers
#               are unchanged.
#   "first"  -- always row 0 (the pre-fix behaviour; reproduces an old run
#               against a new dataset).
#   "mean"   -- average the numeric columns across repeats, i.e. reproduce what
#               the current averaged dataset already is. Use this to isolate
#               "what did the extra rows buy us?" against an otherwise identical
#               configuration.
# Sampling uses `random`, which src/_seeding.py seeds, so runs stay reproducible.
# NOTE: this is the INFERENCE-side half of D-40. The other half is the retrain
# (B1b): with repeats available, the regressors should be fitted on all of them.
#
# ⚠️ The current dataset is NOT quite uniform: server5.csv has 369 rows for 363
# combinations -- 's', 'd', 'p', 'ss', 'sd' and 'sp' carry TWO genuine
# measurements each (different times AND different telemetry, not copy-paste;
# e.g. 'sd' = 0.033799 s vs 0.023559 s, a 30% spread). Servers 1-4 are exactly
# 363/363. So on today's data "sample" is a no-op everywhere EXCEPT those six
# contention strings on server 5.
# LEGACY_ENV therefore pins "first", so SAFETAIL_LEGACY_ENV=1 still reproduces
# results/reference_v0 exactly, consistent with the other legacy switches above.
TRACE_SAMPLING = ("first" if LEGACY_ENV else
                  (os.environ.get("SAFETAIL_TRACE_SAMPLING", "sample").strip().lower()
                   or "sample"))

# [SAFETAIL][REGRESSOR][D-40] Where computation time comes from.
#   "model"     -- the fitted regressor (DEFAULT; today's behaviour).
#   "empirical" -- sample among the MEASURED times for that contention string,
#                  falling back to the model for unmeasured strings.
#
# ⚠️ WHY THIS EXISTS. The shipped regressors are POINT predictors fitted on
# squared error, so they return E[y|x] -- the conditional MEAN. Demonstrated on
# the only repeats the current data happens to contain (server5, 'sd'):
#     measured  33.7991 ms  and  23.5592 ms
#     model     20.9059 ms  for BOTH
# A distributional dataset fed to such a model does not yield a distribution; it
# yields a better-estimated mean. So collecting repeated measurements is
# necessary but NOT sufficient for D-40 -- the inference path has to stop
# averaging them too. That is what "empirical" is for.
#
# Recommended sequence once the new dataset lands:
#   1. python tools/verify_dataset.py          (gate G8 -- is the data usable?)
#   2. SAFETAIL_COMPUTATION_SOURCE=empirical   (measured variance, no retrain)
#   3. B1b retrain, ideally quantile / residual-aware, to generalise to
#      contention strings the trace does not cover.
COMPUTATION_SOURCE = os.environ.get("SAFETAIL_COMPUTATION_SOURCE", "model").strip().lower() or "model"

# D-23: the queue wait fed to P(T) and the episodic penalty is SIMULATED, not
# wall-clock. The servers are an M/M/c/c loss system (D-24) -- no real queue --
# so the only genuine wait is the D-21 saturation backoff a request incurred,
# plus a small fixed controller-dispatch cost (ms).
DISPATCH_COST_MS = float(os.environ.get("SAFETAIL_DISPATCH_COST_MS", "1.0"))

############################### TRAINING LOGS ##########################

training_log_folder = os.environ.get("TRAINING_LOG_FOLDER", "training_logs_1")

# references the results where the results were stored when first run was performed, WHILE TESTING PHASE IS RUNNING,
# leave same as `training_log_folder` if training for first time
original_training_log_folder = os.environ.get("TRAINING_LOG_FOLDER", "training_logs_1")

############################### DEADLINES ##########################

# Deadlines in ms: [[D1_s, D2_s], [D1_dp, D2_dp]]
# "s" (speech) type:      D1=100ms (soft), D2=400ms (hard)
# "d"/"p" type:           D1=30ms  (soft), D2=200ms (hard)

DEADLINE_SCALE = 1  # change accordingly to make deadlines 80% or 150% of original

ORIGINAL_DEADLINES = [[100, 400], [30, 200]]
deadlines = [[value * DEADLINE_SCALE for value in pair] for pair in ORIGINAL_DEADLINES]

############################### TESTING ##########################

# true when need to run testing phase
testing_phase_active = False

# which model to use while testing
saved_model_path = ""

############################### ADDRESS CONFIGURATION ##########################

receiver_host = "127.0.0.1"
receiver_port = int(os.environ.get("RECEIVER_PORT", 6001))

############################### BASELINE COMPARISON ##########################

BASELINE_MODE = os.environ.get("BASELINE_MODE", "safetail")
# Options:
#   "safetail"                      -- the DQN policy (or an external one via POLICY)
#   "{minload,minprop,rand}_{K}"    -- fixed-K redundant dispatcher; [SAFETAIL][B6]
#                                      K now ranges 1..beta (was hardcoded 1..3, D-09).
#                                      e.g. minload_4, minprop_5, rand_2
# NOTE the _{K} suffix means the REPLICATION BUDGET, not a seed. Seeds are
# SAFETAIL_SEED (plan.md B7). Results dirs are {policy}_{k}_{seed}_{gitsha7}.
