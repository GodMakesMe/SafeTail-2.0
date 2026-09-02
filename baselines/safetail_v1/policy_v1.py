"""
[SAFETAIL][POLICY][M-02] SafeTailV1Policy -- the SafeTail 1.0 algorithm driving
the SafeTail 2.0 environment through the seam (src/policy_registry.Policy).

plan.md section 8: this is a PORT, not a bridge. It reimplements the 1.0
algorithm (state abstraction + network + tau-referenced 5-case reward + eps-greedy
replay) against the 2.0 environment API. It never imports src/agent, never loads
1.0's pickles, never touches _spec_source/.

Transition assembly is online with a one-step delay: at select() step t we
finalise the transition from step t-1 using step t's state as next_state, then
push it to replay. finish_episode() runs one replay step (1.0's cadence, moved
to the episode boundary -- section 8.6).
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np

from policy_registry import BasePolicy, PolicyContext, index_to_subset, subset_to_index

from . import config_v1 as cfg
from .model_v1 import build_model
from .reward_v1 import reward_v1
from .state_v1 import build_state
from .train_v1 import experience_replay


class SafeTailV1Policy(BasePolicy):
    name = "safetail_v1"

    def __init__(self):
        self.model = build_model(cfg.NS, cfg.NA, cfg.LR)
        self.memory: deque = deque([], maxlen=cfg.REPLAY_MAXLEN)
        self.epsilon = cfg.EPS_START
        self.frozen = False

        # counters for the faithfulness / manifest report
        self.n_select = 0
        self.n_explore = 0
        self.n_replay = 0
        self._step = 0
        self.v1_bug01_out_of_band = 0          # V1-BUG-01 events
        self.losses: list[float] = []
        self.access_rates: list[float] = []

        self._pending = None                  # (state, action_idx, subset, ctx)
        self._prev = None                     # (state, action_idx, reward) awaiting next_state

    # -- Policy protocol ---------------------------------------------------
    def select(self, ctx: PolicyContext):
        state = build_state(ctx)
        self.n_select += 1

        if (not self.frozen) and random.random() <= self.epsilon:
            action_idx = random.randrange(cfg.NA)
            self.n_explore += 1
        else:
            # [SAFETAIL][POLICY][PERF] direct __call__, not predict(): identical
            # maths, without the per-call data-pipeline setup that dominates a
            # 2k-parameter net (this runs once per request, 15,225 times).
            q = self.model(state.reshape(1, -1).astype("float32"), training=False).numpy()[0]
            action_idx = int(np.argmax(q))

        subset = index_to_subset(action_idx, ctx.beta)
        self.access_rates.append(len(subset) / ctx.beta)

        # close out the previous transition now that we have its next_state
        if self._prev is not None and not self.frozen:
            s_prev, a_prev, r_prev = self._prev
            self.memory.append((s_prev, a_prev, r_prev, state))
        self._pending = (state, action_idx, subset, ctx)
        return subset

    def observe(self, ctx: PolicyContext, action, reward: float) -> None:
        # IGNORE `reward` -- that is the 2.0 headroom reward. Grade with the 1.0
        # tau-referenced 5-case reward on the pre-action estimate (V1-DEV-02).
        if self._pending is None or self.frozen:
            return
        state, action_idx, subset, sel_ctx = self._pending
        sel = [int(i) for i in action] or subset
        est = [sel_ctx.est_delay[i] for i in sel if 0 <= i < len(sel_ctx.est_delay)]
        if not est:
            return
        node_lat = [np.log1p(e) for e in est] if cfg.LOG_COMPRESS else est
        obs_latency = float(min(node_lat))
        rtype = sel_ctx.request_type if sel_ctx.request_type in cfg.TAU_BY_TYPE else "d"
        tau = cfg.TAU_BY_TYPE[rtype]
        r, out_of_band = reward_v1(obs_latency, tau, len(sel), sel_ctx.beta, cfg.ALPHA)
        if out_of_band:
            self.v1_bug01_out_of_band += 1
        self._prev = (state, action_idx, r)
        self._pending = None

        # [SAFETAIL][POLICY][V1-DEV-03] replay at 1.0's own cadence -- once per
        # request (1.0 called experience_replay from inside reward()). Driven
        # from the step boundary here rather than from inside the reward
        # function, which is the only structural change (plan.md 8.6).
        # Replaying once per EPISODE instead gave 1,015 gradient steps and 1,015
        # epsilon decrements over the run: the policy never left exploration.
        self._step += 1
        if (not self.frozen) and self._step % max(1, cfg.REPLAY_EVERY) == 0:
            loss, _val, self.epsilon = experience_replay(self.model, self.memory, self.epsilon)
            if loss == loss:  # not NaN
                self.n_replay += 1
                self.losses.append(loss)

    def finish_episode(self, episode_index: int) -> None:
        if cfg.FREEZE_AFTER_EPISODES and episode_index >= cfg.FREEZE_AFTER_EPISODES and not self.frozen:
            self.freeze()

    # -- snapshots -------------------------------------------------------
    def save_snapshot(self, out_dir, tag: str = "final") -> str | None:
        """
        [SAFETAIL][POLICY][SNAPSHOT] Persist the trained 1.0 network + the run
        state needed to interpret or resume it.

        Writes <out_dir>/snapshots/safetail_v1_<tag>.keras plus a sibling
        .json with epsilon, replay/step counts, tau, and the V1-DEV-03 schedule
        that produced it -- a checkpoint without its schedule is not reproducible.
        """
        import json
        from pathlib import Path
        d = Path(out_dir) / "snapshots"
        d.mkdir(parents=True, exist_ok=True)
        model_path = d / f"safetail_v1_{tag}.keras"
        try:
            self.model.save(model_path)
        except Exception as e:  # noqa: BLE001
            print(f"[SAFETAIL][POLICY][SNAPSHOT] model save failed: {type(e).__name__} - {e}")
            return None
        meta = self.report()
        meta.update({
            "tag": tag,
            "gamma_decay": cfg.GAMMA_DECAY,
            "gamma_decay_native": cfg.GAMMA_DECAY_NATIVE,
            "eps_decay_fraction": cfg.EPS_DECAY_FRACTION,
            "replay_every": cfg.REPLAY_EVERY,
            "total_steps_assumed": cfg.TOTAL_STEPS,
            "batch": cfg.BATCH, "gamma": cfg.GAMMA, "lr": cfg.LR,
            "nS": cfg.NS, "nA": cfg.NA, "beta": cfg.BETA,
            "model_path": str(model_path),
        })
        (d / f"safetail_v1_{tag}.json").write_text(json.dumps(meta, indent=2, default=str),
                                                   encoding="utf-8")
        print(f"[SAFETAIL][POLICY][SNAPSHOT] saved -> {model_path}")
        return str(model_path)

    # -- helpers ---------------------------------------------------------
    def freeze(self) -> None:
        self.frozen = True
        self.epsilon = 0.0
        print(f"[SAFETAIL][POLICY][safetail_v1] frozen -> argmax-only inference "
              f"(after {self.n_replay} replay steps, eps was {self.epsilon})")

    def report(self) -> dict:
        return {
            "policy": self.name,
            "selects": self.n_select,
            "explore_frac": (self.n_explore / self.n_select) if self.n_select else 0.0,
            "replay_steps": self.n_replay,
            "epsilon": self.epsilon,
            "frozen": self.frozen,
            "V1_BUG_01_out_of_band": self.v1_bug01_out_of_band,
            "mean_access_rate": float(np.mean(self.access_rates)) if self.access_rates else 0.0,
            "final_loss": self.losses[-1] if self.losses else None,
            "tau_by_type": cfg.TAU_BY_TYPE,
        }


def factory() -> SafeTailV1Policy:
    return SafeTailV1Policy()
