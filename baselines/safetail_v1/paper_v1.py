"""
[SAFETAIL][POLICY][M-02][S-18] SafeTail 1.0 **as published in the paper**.

`policy_v1.py` is faithful to the SafeTail 1.0 *GitHub code*
(`_spec_source/v1_agent.py`). This module is faithful to the SafeTail 1.0
*camera-ready paper* (`audit/SafeTail_Camera_Ready.pdf`, §IV, Eq. 5 and Eq. 6).
**They are not the same algorithm.** Three divergences, all verified against the
paper text (see audit/ERRATA.md S-18..S-20):

  S-18  TARGET CONSTRUCTION.
        Paper: the reward is translated into a target vector V_t that is a
        PROBABILITY DISTRIBUTION -- "we ensure that the sum of all elements in
        the target vector equals 1". No bootstrapping, no max over the next
        state, no gamma. That is coherent with a softmax head + categorical
        cross-entropy.
        Code: targets[arange, actions] = rewards + gamma*amax(next_q), then
        categorical_crossentropy. Bellman Q-targets fed to a distribution loss --
        incoherent, and the reason the ported policy destabilises with training.

  S-19  REWARD (Eq. 5).
        Paper, late  (L_R > tau, |E_k| < n): -delta * e^(n - |E_k|)
              -- depends ONLY on remaining redundancy headroom.
        Code:                                -alpha * e^(n-|E_k|) * e^(L_R-tau)
              -- an extra lateness factor the paper does not have.
        Paper, early (L_R < tau, |E_k| > 1):  -delta * e^(L_R - tau)
              -- DECAYS as the request finishes earlier.
        Code:                                 -alpha * e^(|E_k|-1) * e^(tau-L_R)
              -- exponent sign flipped, so it GROWS; plus a redundancy factor.
        Paper, early with |E_k| == 1: 0. The code has no such case.

  S-20  ARCHITECTURE.
        Paper §IV: "The FNN comprises 5 hidden layers with ReLU activations and
        a Softmax output layer... optimized using Adam with categorical
        cross-entropy". Code: 2 hidden layers, sigmoid, + BatchNormalization.

Run it with `--policy safetail_v1_paper`; `--policy safetail_v1` remains the
code-faithful port. Reporting both is the honest thing to do: it separates
"what SafeTail 1.0 proposed" from "what SafeTail 1.0 shipped".
"""
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

from policy_registry import BasePolicy, PolicyContext, index_to_subset

from . import config_v1 as cfg
from .state_v1 import build_state


# --------------------------------------------------------------------------- #
# Eq. 5 -- the paper's reward
# --------------------------------------------------------------------------- #
def reward_paper(latency: float, tau: float, k_size: int, n: int, delta: float) -> float:
    """
    SafeTail 1.0 camera-ready, Definition 4.3 / Eq. (5), verbatim.

        0                       if L_R == tau
        0                       if L_R <  tau and |E_k| == 1
        0                       if L_R >  tau and |E_k| == n
        -delta * e^(n - |E_k|)  if L_R >  tau and |E_k| <  n
        -delta * e^(L_R - tau)  if L_R <  tau and |E_k| >  1

    Upper bound 0 (Property 4.1). Note the asymmetry the paper intends
    (Property 4.4): missing the target latency when redundancy could still have
    been increased is penalised far more heavily than meeting it wastefully --
    e^(n-|E_k|) is O(1..e^4) whereas e^(L_R-tau) < 1 for an early finish.
    """
    if latency == tau:
        return 0.0
    if latency < tau:
        if k_size == 1:
            return 0.0
        return -delta * math.exp(latency - tau)          # < 0, decays with earliness
    # latency > tau
    if k_size >= n:
        return 0.0
    return -delta * math.exp(n - k_size)                  # depends only on headroom


# --------------------------------------------------------------------------- #
# Eq. 6 -- the paper's target vector
# --------------------------------------------------------------------------- #
def target_vector(action_idx: int, reward: float, beta: int, n_actions: int,
                  subset_cache: dict | None = None) -> np.ndarray:
    """
    SafeTail 1.0 camera-ready, Eq. (6) + surrounding text.

      * length 2^n - 1, and SUMS TO 1 (it is a distribution, not a Q-vector).
      * R == 0        -> one-hot on the chosen action A_k.
      * R <  0        -> start every element at 1/(2^n - 1); for A_k and every
                         A_j whose server set E_j is a SUBSET of E_k, set
                             V(j) = max(0, 1/(2^n - 1) + R)
                         then distribute the remaining mass
                             1 - sum_{j: E_j subset of E_k} V(j)
                         equally among the remaining elements.

    So a bad outcome pushes probability mass off the chosen action *and every
    action dominated by it*, and spreads it over the alternatives. This is
    policy improvement over a distribution -- exactly what softmax + CCE wants.
    """
    v = np.zeros(n_actions, dtype="float32")
    if reward == 0.0:
        v[action_idx] = 1.0
        return v

    base = 1.0 / n_actions
    chosen = frozenset(index_to_subset(action_idx, beta))

    if subset_cache is not None and chosen in subset_cache:
        dominated = subset_cache[chosen]
    else:
        dominated = [j for j in range(n_actions)
                     if frozenset(index_to_subset(j, beta)) <= chosen]
        if subset_cache is not None:
            subset_cache[chosen] = dominated

    val = max(0.0, base + reward)
    v[dominated] = val
    used = val * len(dominated)
    rest = [j for j in range(n_actions) if j not in set(dominated)]
    if rest:
        v[rest] = max(0.0, (1.0 - used)) / len(rest)
    s = v.sum()
    if s > 0:
        v /= s                       # guard: the paper asserts sum == 1
    else:
        v[action_idx] = 1.0
    return v


# --------------------------------------------------------------------------- #
# Paper architecture: 5 hidden ReLU layers + softmax
# --------------------------------------------------------------------------- #
def build_model_paper(n_s: int, n_a: int, lr: float, width: int | None = None):
    w = width or cfg.PAPER_HIDDEN_WIDTH
    layers = [keras.layers.Input(shape=(n_s,))]
    for _ in range(cfg.PAPER_HIDDEN_LAYERS):
        layers.append(keras.layers.Dense(w, activation="relu"))
    layers.append(keras.layers.Dense(n_a, activation="softmax"))
    m = keras.Sequential(layers, name="safetail_v1_paper_fnn")
    m.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=lr))
    return m


# --------------------------------------------------------------------------- #
class SafeTailV1PaperPolicy(BasePolicy):
    name = "safetail_v1_paper"

    def __init__(self):
        self.model = build_model_paper(cfg.NS, cfg.NA, cfg.LR)
        self.memory: deque = deque([], maxlen=cfg.REPLAY_MAXLEN)
        self.epsilon = cfg.EPS_START
        self.frozen = False
        self.n_select = self.n_explore = self.n_replay = self._step = 0
        self.losses: list[float] = []
        self.access_rates: list[float] = []
        self._pending = None
        self._subset_cache: dict = {}
        self._train_step = None

    # -- Policy protocol --------------------------------------------------
    def select(self, ctx: PolicyContext):
        state = build_state(ctx)
        self.n_select += 1
        if (not self.frozen) and random.random() <= self.epsilon:
            action_idx = random.randrange(cfg.NA)
            self.n_explore += 1
        else:
            p = self.model(state.reshape(1, -1).astype("float32"), training=False).numpy()[0]
            action_idx = int(np.argmax(p))
        subset = index_to_subset(action_idx, ctx.beta)
        self.access_rates.append(len(subset) / ctx.beta)
        self._pending = (state, action_idx, subset, ctx)
        return subset

    def observe(self, ctx: PolicyContext, action, reward: float) -> None:
        if self._pending is None or self.frozen:
            return
        state, action_idx, subset, sel_ctx = self._pending
        self._pending = None
        sel = [int(i) for i in action] or subset
        est = [sel_ctx.est_delay[i] for i in sel if 0 <= i < len(sel_ctx.est_delay)]
        if not est:
            return
        latency = float(min(est))
        rtype = sel_ctx.request_type if sel_ctx.request_type in cfg.TAU_BY_TYPE else "d"
        r = reward_paper(latency, cfg.TAU_BY_TYPE[rtype], len(sel), sel_ctx.beta, cfg.PAPER_DELTA)
        # Eq. 6: store the TARGET DISTRIBUTION, not a scalar Q-target.
        v = target_vector(action_idx, r, sel_ctx.beta, cfg.NA, self._subset_cache)
        self.memory.append((state, v))

        self._step += 1
        if (not self.frozen) and self._step % max(1, cfg.REPLAY_EVERY) == 0:
            self._train()

    def _train(self):
        if len(self.memory) < cfg.BATCH:
            return
        mb = random.sample(self.memory, cfg.BATCH)
        xs = np.asarray([m[0] for m in mb], dtype="float32")
        ys = np.asarray([m[1] for m in mb], dtype="float32")
        if self._train_step is None:
            model = self.model

            @tf.function(reduce_retracing=True)
            def step(x, y):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(
                        tf.keras.losses.categorical_crossentropy(y, model(x, training=True)))
                g = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(g, model.trainable_variables))
                return loss
            self._train_step = step

        n_train = int(cfg.BATCH * (1.0 - cfg.VAL_SPLIT))
        for _ in range(cfg.EPOCHS):
            for i in range(0, n_train, cfg.KERAS_BATCH):
                self.losses.append(float(self._train_step(
                    tf.constant(xs[i:i + cfg.KERAS_BATCH]),
                    tf.constant(ys[i:i + cfg.KERAS_BATCH])).numpy()))
        self.n_replay += 1
        if self.epsilon > cfg.EPS_MIN:
            self.epsilon = max(cfg.EPS_MIN, self.epsilon - cfg.GAMMA_DECAY)

    def finish_episode(self, episode_index: int) -> None:
        if cfg.FREEZE_AFTER_EPISODES and episode_index >= cfg.FREEZE_AFTER_EPISODES:
            self.frozen = True
            self.epsilon = 0.0

    # -- reporting --------------------------------------------------------
    def report(self) -> dict:
        return {
            "policy": self.name, "variant": "paper-faithful (Eq. 5 + Eq. 6)",
            "selects": self.n_select,
            "explore_frac": (self.n_explore / self.n_select) if self.n_select else 0.0,
            "replay_steps": self.n_replay, "epsilon": self.epsilon, "frozen": self.frozen,
            "mean_access_rate": float(np.mean(self.access_rates)) if self.access_rates else 0.0,
            "final_loss": self.losses[-1] if self.losses else None,
            "tau_by_type": cfg.TAU_BY_TYPE, "delta": cfg.PAPER_DELTA,
            "hidden_layers": cfg.PAPER_HIDDEN_LAYERS, "hidden_width": cfg.PAPER_HIDDEN_WIDTH,
        }

    def save_snapshot(self, out_dir, tag: str = "final"):
        import json
        from pathlib import Path
        d = Path(out_dir) / "snapshots"; d.mkdir(parents=True, exist_ok=True)
        p = d / f"safetail_v1_paper_{tag}.keras"
        try:
            self.model.save(p)
        except Exception as e:  # noqa: BLE001
            print(f"[SAFETAIL][POLICY][SNAPSHOT] failed: {e}"); return None
        (d / f"safetail_v1_paper_{tag}.json").write_text(
            json.dumps(self.report(), indent=2, default=str), encoding="utf-8")
        print(f"[SAFETAIL][POLICY][SNAPSHOT] saved -> {p}")
        return str(p)


def factory() -> SafeTailV1PaperPolicy:
    return SafeTailV1PaperPolicy()
