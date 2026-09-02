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
import tensorflow as tf

from . import config_v1 as cfg


# [SAFETAIL][POLICY][PERF] Compiled train step, built once per model.
#
# Keras `model.predict()` / `model.fit()` are designed for large datasets: each
# call rebuilds a data pipeline, runs callbacks and (for fit) re-splits for
# validation. On a 2k-parameter net at batch 128 that framework overhead is
# 10-30 ms and completely dominates the ~microsecond of actual arithmetic. With
# one replay per request (V1-DEV-03) that overhead is paid 15,225 times.
#
# Calling the model directly and driving one gradient step inside a tf.function
# keeps the arithmetic identical and removes the per-call machinery.
_TRAIN_STEPS: dict[int, object] = {}


def _get_train_step(model):
    key = id(model)
    fn = _TRAIN_STEPS.get(key)
    if fn is not None:
        return fn

    @tf.function(reduce_retracing=True)
    def train_step(states, targets):
        with tf.GradientTape() as tape:
            preds = model(states, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(targets, preds))
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    _TRAIN_STEPS[key] = train_step
    return train_step


def experience_replay(model, memory, epsilon: float) -> tuple[float, float, float]:
    """One replay step. Returns (loss, val_loss, new_epsilon)."""
    if len(memory) < cfg.BATCH:
        return float("nan"), float("nan"), epsilon

    minibatch = random.sample(memory, cfg.BATCH)
    states = np.asarray([m[0] for m in minibatch], dtype="float32")
    actions = np.asarray([m[1] for m in minibatch], dtype=int)
    rewards = np.asarray([m[2] for m in minibatch], dtype="float32")
    next_states = np.asarray([m[3] for m in minibatch], dtype="float32")

    # direct __call__ instead of predict(): same maths, no data-pipeline setup
    current_q = model(states, training=False).numpy()
    next_q = model(next_states, training=False).numpy()
    targets = current_q.copy()
    targets[np.arange(cfg.BATCH), actions] = rewards + cfg.GAMMA * np.amax(next_q, axis=1)

    # [SAFETAIL][POLICY][PERF] SafeTail 1.0 called
    #   model.fit(states, targets, epochs=EPOCHS, validation_split=0.2)
    # Keras' validation_split holds out the LAST 20% of the batch, so 1.0 took
    # its gradient on only the first 80% (102 of 128). Replicate that exactly --
    # training on all 128 here would make this a faster but DIFFERENT algorithm.
    n_train = int(cfg.BATCH * (1.0 - cfg.VAL_SPLIT))
    xs_all = states[:n_train]
    ys_all = targets[:n_train].astype("float32")
    xv = tf.constant(states[n_train:])
    yv = tf.constant(targets[n_train:], dtype="float32")

    # ...and Keras fit() defaults to batch_size=32, so 1.0 took
    # ceil(102/32) * EPOCHS = 8 gradient updates of 32 per replay, NOT 2
    # full-batch updates of 102. Replicate the minibatching too -- the update
    # COUNT and SIZE change the optimisation dynamics, not just the wall time.
    step = _get_train_step(model)
    losses = []
    for _ in range(cfg.EPOCHS):
        for i in range(0, n_train, cfg.KERAS_BATCH):
            xb = tf.constant(xs_all[i:i + cfg.KERAS_BATCH])
            yb = tf.constant(ys_all[i:i + cfg.KERAS_BATCH])
            losses.append(float(step(xb, yb).numpy()))

    val = float(tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(yv, model(xv, training=False))).numpy())

    if epsilon > cfg.EPS_MIN:
        epsilon = max(cfg.EPS_MIN, epsilon - cfg.GAMMA_DECAY)  # subtractive (ST Eq. 3)

    return losses[0], val, epsilon
