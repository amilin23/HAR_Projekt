import os
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras

from src.config import Config, default_session_to_activity, session_to_subject
from src.preprocessing.dataset import build_dataset
from src.model.model_lstm import make_lstm


def main():
    cfg = Config()
    session_to_activity = default_session_to_activity()
    sessions = list(range(12, 22))

    # Build full dataset once (windows)
    X, y_str, g = build_dataset(
        fs=cfg.fs,
        win_sec=cfg.win_sec,
        hop_sec=cfg.hop_sec,
        sessions=sessions,
        session_to_activity=session_to_activity,
    )

    # Classes in stable order
    classes = sorted(np.unique(y_str))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)

    # Groups are subjects ("amy"/"thasa")
    subjects = sorted(np.unique(g))
    if len(subjects) < 2:
        raise RuntimeError(f"Need at least 2 subjects for LOSO. Found: {subjects}")

    print("\nClasses:", classes)
    print("Subjects:", subjects)
    print("Total windows:", len(X))

    all_true = []
    all_pred = []

    # Evaluate: leave one subject out
    for test_subj in subjects:
        train_mask = (g != test_subj)
        test_mask = (g == test_subj)

        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]

        # Normalize using TRAIN subject only
        mean = Xtr.mean(axis=(0, 1), keepdims=True)
        std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
        Xtr = (Xtr - mean) / std
        Xte = (Xte - mean) / std

        # Fresh model per fold
        model = make_lstm(input_shape=Xtr.shape[1:], n_classes=len(classes))

        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5),
        ]

        model.fit(
            Xtr, ytr,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            validation_split=0.1,
            callbacks=callbacks,
        )

        probs = model.predict(Xte, verbose=0)
        yp = probs.argmax(axis=1)

        fold_acc = accuracy_score(yte, yp)
        print(f"\n=== LOSO Fold: Test subject = {test_subj} ===")
        print(f"Fold window-accuracy: {fold_acc:.4f}")

        all_true.append(yte)
        all_pred.append(yp)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    print("\n" + "=" * 60)
    print("LOSO BY SUBJECT (WINDOW-LEVEL) FINAL RESULTS")
    print("=" * 60)

    acc = accuracy_score(all_true, all_pred)
    print(f"\nOverall Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_true, all_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(all_true, all_pred)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes))


if __name__ == "__main__":
    main()