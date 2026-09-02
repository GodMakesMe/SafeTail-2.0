"""
[SAFETAIL][POLICY][M-02] SafeTail 1.0 baseline -- hyperparameters.

These are SafeTail 1.0's OWN values (the `instance` per-task config, beta=5),
transcribed from baselines/safetail_v1/_spec_source/v1_constants_instance.py and
v1_agent.py. plan.md section 14.1.

The four adaptation decisions (plan.md section 8.5) are recorded here as the
A-1..A-4 constants so a reviewer can see exactly what changed and why.
"""
from __future__ import annotations

import os

BETA = 5
ALPHA = 0.001                 # instance config
LR = 1e-6
GAMMA = 0.95
EPS_START = 1.0
EPS_MIN = 0.1

# ---------------------------------------------------------------------------
# V1-DEV-03: exploration schedule scaled to the RUN BUDGET.
#
# SafeTail 1.0's own value is GAMMA_DECAY_NATIVE = 5.5e-6, subtractive per
# replay (ST Eq. 3). 1.0 called experience_replay from inside reward(), i.e.
# once per request, over 6000 episodes x 30 steps ~= 180,000 steps -- so
# 180000 * 5.5e-6 = 0.99 and epsilon decayed fully.
#
# Our comparison budget is one pass over the SAME data as the heterogeneous run:
# 15,225 requests. At the native rate epsilon would fall by 15225*5.5e-6 = 0.084
# -- it would end at ~0.92 and the "baseline" would be a uniformly random
# scheduler for the entire run (observed: mean K = 2.58, exactly the uniform
# mean subset size over the 31 subsets). That measures nothing.
#
# So the decay is scaled so epsilon reaches EPS_MIN after EPS_DECAY_FRACTION of
# the run's replay calls. This is an explicit, recorded adaptation of the same
# class as tau (plan.md 8.5) -- 1.0's ALGORITHM is preserved; only the schedule
# constant is rescaled to the data budget it is given. Set
# SAFETAIL_V1_GAMMA_DECAY to override, or =5.5e-06 to force the native value.
# ---------------------------------------------------------------------------
GAMMA_DECAY_NATIVE = 5.5e-6
TOTAL_STEPS = int(os.environ.get("SAFETAIL_V1_TOTAL_STEPS", "15225"))
# 0.45 matches the exploration profile of the reference_v0 SafeTail-2.0 run it is
# compared against: that run used gamma_decay=0.002 subtractive with one replay
# per episode over 1,015 episodes, so it reached epsilon_min at episode ~459 --
# 45% of the way through. Equal exploration budget on both sides.
EPS_DECAY_FRACTION = float(os.environ.get("SAFETAIL_V1_EPS_DECAY_FRAC", "0.45"))
# 1.0 replayed once per request; keep that cadence (see policy_v1.observe).
REPLAY_EVERY = int(os.environ.get("SAFETAIL_V1_REPLAY_EVERY", "1"))
_n_replays = max(1.0, EPS_DECAY_FRACTION * TOTAL_STEPS / max(1, REPLAY_EVERY))
GAMMA_DECAY = float(os.environ.get("SAFETAIL_V1_GAMMA_DECAY",
                                   str((EPS_START - EPS_MIN) / _n_replays)))
BATCH = 128
REPLAY_MAXLEN = 2500
EPOCHS = 2
VAL_SPLIT = 0.2
# Keras fit()'s default minibatch size. 1.0 called fit(x, y, epochs=2) without a
# batch_size, so it took ceil(102/32)*2 = 8 gradient updates of 32 per replay.
# The fast path replicates that exactly (see train_v1.experience_replay).
KERAS_BATCH = 32

# A-4: RESOLUTION is omitted (2.0 has no resolution attribute; the 1.0
# instance/noise configs also omit it). => nS = 2*beta + 2, not 2*beta + 3.
NS = 2 * BETA + 2
NA = 2 ** BETA - 1

# tau (median_computation_delay) PER REQUEST TYPE, seconds.
# plan.md 8.5 / decision 13.1(1): derived as the median service latency
# (computation + propagation + transmission) per type from
# results/reference_v0/safetail_training_logs/latency_log.csv:
#     d: 34.46 ms   p: 35.47 ms   s: 51.80 ms
# A single global tau would be meaningless -- the types differ by >1 order of
# magnitude in service time (S-14). Override with SAFETAIL_V1_TAU_{S,D,P} (sec).
TAU_BY_TYPE = {
    "s": float(os.environ.get("SAFETAIL_V1_TAU_S", "0.0518")),
    "d": float(os.environ.get("SAFETAIL_V1_TAU_D", "0.0345")),
    "p": float(os.environ.get("SAFETAIL_V1_TAU_P", "0.0355")),
}

# V1-DEV-01 (plan.md 14.1): 1.0 applies node_latency = log(1 + node_latency)
# before the min for instance/noise. We DROP that compression and use raw
# latency, with tau set accordingly. Flip to True only to study the effect.
LOG_COMPRESS = os.environ.get("SAFETAIL_V1_LOG_COMPRESS", "0") not in ("0", "", "false")

# V1-DEV-02: the 1.0 reward graded itself on the REALISED min latency. The 2.0
# seam deliberately exposes only the pre-action estimate (ctx.est_delay, the
# single D-18 draw) so no policy can see its own outcome. The port therefore
# grades on min(ctx.est_delay over A). Documented deviation, not a bug.

# A-1: LOAD[i] is clipped to [1, MAX_CONCURRENT]. Never feed the heterogeneous
# hardware vector -- 1.0 by construction cannot see it.
MAX_LOAD = 4  # == servers.MAX_CONCURRENT_REQUESTS

# train-then-freeze split (decision 13.1(4)). Episodes before freezing to
# argmax-only inference. 0 / unset => train online for the whole run (1.0's own
# v1_agent.py behaviour).
FREEZE_AFTER_EPISODES = int(os.environ.get("SAFETAIL_V1_FREEZE_AFTER", "0") or "0")
