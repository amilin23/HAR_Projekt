from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.config import Config, default_session_to_activity
from src.model.model_lstm import make_lstm
from src.preprocessing.dataset import build_dataset

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def _encode_labels(y_str: np.ndarray) -> Tuple[List[str], np.ndarray]:
    classes = sorted(np.unique(y_str))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)
    return classes, y


def _split_data(X: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int, test_size: float):
    # Stratify by (label + subject)
    strat = [f"{label}|{group}" for label, group in zip(y, g)]

    X_train_val, X_test, y_train_val, y_test, g_train_val, g_test = train_test_split(
        X, y, g,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    strat2 = [f"{label}|{group}" for label, group in zip(y_train_val, g_train_val)]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.20,   # 20% of train_val -> validation
        random_state=seed,
        stratify=strat2,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _fit_normalizer(X_train: np.ndarray):
    # Global normalization using TRAIN only
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return mean, std


def _apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def _save_classes(path: str, classes: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")


def main():
    cfg = Config()
    sessions = list(cfg.sessions)
    session_to_activity = default_session_to_activity()

    X, y_str, g, _s = build_dataset(
        sessions=sessions,
        fs=cfg.fs,
        win_sec=cfg.win_sec,
        hop_sec=cfg.hop_sec,
        session_to_activity=session_to_activity,
    )

    classes, y = _encode_labels(y_str)

    print("\nClasses:", classes)
    print("Total windows:", len(X))

    Xtr, Xval, Xte, ytr, yval, yte = _split_data(
        X, y, g, seed=cfg.random_state, test_size=cfg.test_size
    )
    print(f"Train: {len(Xtr)} | Val: {len(Xval)} | Test: {len(Xte)}")

    mean, std = _fit_normalizer(Xtr)
    Xtr = _apply_normalizer(Xtr, mean, std)
    Xval = _apply_normalizer(Xval, mean, std)
    Xte = _apply_normalizer(Xte, mean, std)

    model = make_lstm(input_shape=Xtr.shape[1:], n_classes=len(classes))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
    ]

    print("START TRAINING...")
    
    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # eval
    probs = model.predict(Xte, verbose=0)
    yp = probs.argmax(axis=1)

    print("\n=== TEST REPORT (Random Shuffle Split) ===")
    print(classification_report(yte, yp, target_names=classes, zero_division=0))
    cm = confusion_matrix(yte, yp)
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes))

    os.makedirs(cfg.out_dir, exist_ok=True)
    model.save(os.path.join(cfg.out_dir, "ax6_lstm_independent_wrist.keras"))
    np.save(os.path.join(cfg.out_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(cfg.out_dir, "norm_std.npy"), std)
    _save_classes(os.path.join(cfg.out_dir, "classes.txt"), classes)

    print(f"\nSaved model + normalization + classes to: {cfg.out_dir}")

if __name__ == "__main__":
    main()