from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def make_lstm(input_shape, n_classes: int) -> keras.Model:
    
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.30),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.30),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.50),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    return model