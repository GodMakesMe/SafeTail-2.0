"""
[SAFETAIL][POLICY][M-02] SafeTail 1.0's exact network.

plan.md section 14.1 "Network" + section 8.6 faithfulness register:

    Dense(2*nS, input_dim=nS, activation='sigmoid')
    BatchNormalization()
    Dense(4*nS, activation='sigmoid')
    BatchNormalization()
    Dense(nA, activation='softmax')
    compile(loss='categorical_crossentropy', optimizer=Adam(lr))

Kept verbatim -- including softmax + CCE, which are wrong for Q-regression.
Changing them would make this "SafeTail 1.5", not SafeTail 1.0. The oddity is
noted in baselines/safetail_v1/README.md.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def build_model(n_s: int, n_a: int, lr: float):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_s,)),
        keras.layers.Dense(2 * n_s, activation="sigmoid"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(4 * n_s, activation="sigmoid"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(n_a, activation="softmax"),
    ], name="safetail_v1_dqn")
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model
