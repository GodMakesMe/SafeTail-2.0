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
GAMMA_DECAY = 5.5e-6          # instance config; SUBTRACTIVE per replay (ST Eq. 3)
BATCH = 128
REPLAY_MAXLEN = 2500
EPOCHS = 2
VAL_SPLIT = 0.2

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
