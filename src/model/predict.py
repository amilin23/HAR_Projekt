from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.config import Config, default_session_to_activity
from src.preprocessing.dataset import build_dataset


def _read_classes(path: str) -> List[str]:
    classes: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                classes.append(s)
    return classes


def main():
    cfg = Config()
    sessions = list(cfg.sessions)
    session_to_activity = default_session_to_activity()

    # Load the same data and reproduce the same split strategy as train.py
    X, y_str, g = build_dataset(
        sessions=sessions,
        fs=cfg.fs,
        win_sec=cfg.win_sec,
        hop_sec=cfg.hop_sec,
        session_to_activity=session_to_activity,
    )

    classes = _read_classes(os.path.join(cfg.out_dir, "classes.csv"))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)

    strat = [f"{label}|{group}" for label, group in zip(y, g)]
    X_train_val, Xtest, y_train_val, ytest, g_train_val, gtest = train_test_split(
        X, y, g,
        test_size=0.50,
        random_state=cfg.random_state,
        stratify=strat,
    )
    strat2 = [f"{label}|{group}" for label, group in zip(y_train_val, g_train_val)]
    Xtr, Xval, ytr, yval = train_test_split(
        X_train_val, y_train_val,
        test_size=0.20,
        random_state=cfg.random_state,
        stratify=strat2,
    )

    model = keras.models.load_model(os.path.join(cfg.out_dir, "ax6_two_wrist_lstm.keras"))
    mean = np.load(os.path.join(cfg.out_dir, "norm_mean.npy"))
    std = np.load(os.path.join(cfg.out_dir, "norm_std.npy"))

    Xte = (Xtest - mean) / (std + 1e-8)
    probs = model.predict(Xte, verbose=0)
    yp = probs.argmax(axis=1)

    print("\n=== TEST REPORT (Random Shuffle Split) ===")
    print(classification_report(ytest, yp, target_names=classes))
    cm = confusion_matrix(ytest, yp)
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes))


if __name__ == "__main__":
    main()