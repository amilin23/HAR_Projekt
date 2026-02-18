from tensorflow import keras
from tensorflow.keras import layers

def make_cnn(input_shape, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, 7, padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(),
        layers.MaxPool1D(2),

        layers.Conv1D(64, 7, padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(),
        layers.MaxPool1D(2),

        layers.Conv1D(64, 7, padding="same"),
        layers.ReLU(),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
