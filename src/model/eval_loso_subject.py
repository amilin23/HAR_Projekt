from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras

from src.config import Config, default_session_to_activity
from src.model.model_lstm import make_lstm
from src.preprocessing.dataset import build_dataset


tf.random.set_seed(42)
np.random.seed(42)


def _encode_labels(y_str: np.ndarray):
    classes = sorted(np.unique(y_str))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)
    return classes, class_to_idx, y


def _fit_normalizer(X_train: np.ndarray):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return mean, std


def _apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def main():
    cfg = Config()
    session_to_activity = default_session_to_activity()

    X, y_str, g, _s = build_dataset(
        sessions=list(cfg.sessions),
        fs=cfg.fs,
        win_sec=cfg.win_sec,
        hop_sec=cfg.hop_sec,
        session_to_activity=session_to_activity,
    )

    classes, class_to_idx, y = _encode_labels(y_str)
    subjects = sorted(np.unique(g).tolist())

    print("\nClasses:", classes)
    print("Subjects:", subjects)
    print("Total windows:", len(X))

    all_true = []
    all_pred = []

    for test_subj in subjects:
        print(f"\n=== LOSO Fold: Test subject = {test_subj} ===")

        tr_idx = np.where(g != test_subj)[0]
        te_idx = np.where(g == test_subj)[0]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        mean, std = _fit_normalizer(Xtr)
        Xtr = _apply_normalizer(Xtr, mean, std)
        Xte = _apply_normalizer(Xte, mean, std)

        model = make_lstm(input_shape=Xtr.shape[1:], n_classes=len(classes))
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            Xtr, ytr,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5),
            ],
        )

        yp = model.predict(Xte, verbose=0).argmax(axis=1)
        acc = accuracy_score(yte, yp)
        print(f"Fold WINDOW accuracy : {acc:.4f}")

        all_true.append(yte)
        all_pred.append(yp)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    print("\n" + "=" * 60)
    print("LOSO FINAL RESULTS")
    print("=" * 60)
    print(f"\nOverall WINDOW Accuracy: {accuracy_score(all_true, all_pred):.4f}")
    print(classification_report(all_true, all_pred, target_names=classes, zero_division=0))
    cm = confusion_matrix(all_true, all_pred)
    print("\nConfusion Matrix:")
    print(cm)



if __name__ == "__main__":
    main()