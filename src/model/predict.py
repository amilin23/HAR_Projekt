from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.config import Config, default_session_to_activity
from src.preprocessing.dataset import build_dataset


def _load_classes(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _split_data(X: np.ndarray, y: np.ndarray, g: np.ndarray, s: np.ndarray, seed: int, test_size: float):
    strat = [f"{label}|{group}" for label, group in zip(y, g)]
    X_train_val, X_test, y_train_val, y_test, g_train_val, g_test, s_train_val, s_test = train_test_split(
        X, y, g, s,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    strat2 = [f"{label}|{group}" for label, group in zip(y_train_val, g_train_val)]
    X_train, X_val, y_train, y_val, g_train, g_val, s_train, s_val = train_test_split(
        X_train_val, y_train_val, g_train_val, s_train_val,
        test_size=0.20,
        random_state=seed,
        stratify=strat2,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, s_test


def _majority_vote_per_session(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray):
    sess_true = []
    sess_pred = []
    for sid in sorted(np.unique(s).tolist()):
        idx = np.where(s == sid)[0]
        true_label = Counter(y_true[idx]).most_common(1)[0][0]
        pred_label = Counter(y_pred[idx]).most_common(1)[0][0]
        sess_true.append(true_label)
        sess_pred.append(pred_label)
    return np.array(sess_true), np.array(sess_pred)


def main():
    cfg = Config()

    model_path = os.path.join(cfg.out_dir, "ax6_lstm_independent_wrist.keras")
    mean_path = os.path.join(cfg.out_dir, "norm_mean.npy")
    std_path = os.path.join(cfg.out_dir, "norm_std.npy")
    classes_path = os.path.join(cfg.out_dir, "classes.txt")

    if not all(os.path.exists(p) for p in [model_path, mean_path, std_path, classes_path]):
        raise FileNotFoundError("Missing model artifacts in out/. Run train first.")

    model = keras.models.load_model(model_path)
    mean = np.load(mean_path)
    std = np.load(std_path)
    classes = _load_classes(classes_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    sessions = list(cfg.sessions)
    session_to_activity = default_session_to_activity()

    X, y_str, g, s = build_dataset(
        sessions=sessions,
        fs=cfg.fs,
        win_sec=cfg.win_sec,
        hop_sec=cfg.hop_sec,
        session_to_activity=session_to_activity,
    )
    y = np.array([class_to_idx[v] for v in y_str], dtype=np.int64)

    _, _, Xte, _, _, yte, s_te = _split_data(X, y, g, s, seed=cfg.random_state, test_size=cfg.test_size)

    Xte = (Xte - mean) / (std + 1e-8)
    yp = model.predict(Xte, verbose=0).argmax(axis=1)

    print("\n=== WINDOW-LEVEL REPORT (Random Shuffle Split, saved model) ===")
    print(f"Window Accuracy: {accuracy_score(yte, yp):.4f}")
    print(classification_report(yte, yp, target_names=classes, zero_division=0))
    cm = confusion_matrix(yte, yp)
    print("Window Confusion matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes))

    # Session-level majority voting
    yt_sess, yp_sess = _majority_vote_per_session(yte, yp, s_te)

    print("\n=== SESSION-LEVEL REPORT (Majority Vote over windows) ===")
    print(f"Session Accuracy: {accuracy_score(yt_sess, yp_sess):.4f}")
    print(classification_report(yt_sess, yp_sess, target_names=classes, zero_division=0))
    cm2 = confusion_matrix(yt_sess, yp_sess)
    print("Session Confusion matrix:")
    print(pd.DataFrame(cm2, index=classes, columns=classes))


if __name__ == "__main__":
    main()