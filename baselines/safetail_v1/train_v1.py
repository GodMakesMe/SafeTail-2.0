"""
[SAFETAIL][POLICY][M-02] SafeTail 1.0 replay trainer.

plan.md section 14.1 "Training" -- experience_replay(batch):

    minibatch = random.sample(memory, batch)
    targets = model.predict(states)
    targets[arange(batch), actions] = rewards + gamma * max(model.predict(next_states), axis=1)
    model.fit(states, targets, epochs=epochs, validation_split=0.2)
    if eps > eps_min: eps -= gamma_decay

No target network (faithful -- 1.0 takes the max over the online net).
plan.md 8.6: replay is called from the EPISODE BOUNDARY here, not from inside
reward() as in 1.0 (that placement was a structural accident).
"""
from __future__ import annotations

import random

import numpy as np

from . import config_v1 as cfg


def experience_replay(model, memory, epsilon: float) -> tuple[float, float, float]:
    """One replay step. Returns (loss, val_loss, new_epsilon)."""
    if len(memory) < cfg.BATCH:
        return float("nan"), float("nan"), epsilon

    minibatch = random.sample(memory, cfg.BATCH)
    states = np.array([m[0] for m in minibatch], dtype=float)
    actions = np.array([m[1] for m in minibatch], dtype=int)
    rewards = np.array([m[2] for m in minibatch], dtype=float)
    next_states = np.array([m[3] for m in minibatch], dtype=float)

    current_q = model.predict(states, verbose=0)
    next_q = model.predict(next_states, verbose=0)
    targets = current_q.copy()
    targets[np.arange(cfg.BATCH), actions] = rewards + cfg.GAMMA * np.amax(next_q, axis=1)

    hist = model.fit(states, targets, epochs=cfg.EPOCHS, verbose=0,
                     validation_split=cfg.VAL_SPLIT)

    if epsilon > cfg.EPS_MIN:
        epsilon = max(cfg.EPS_MIN, epsilon - cfg.GAMMA_DECAY)  # subtractive (ST Eq. 3)

    loss = float(hist.history.get("loss", [float("nan")])[0])
    val = float(hist.history.get("val_loss", [float("nan")])[0])
    return loss, val, epsilon
