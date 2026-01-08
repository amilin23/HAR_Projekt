# src/train_cnn.py
import argparse, os, json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras

from .config import Config, default_session_to_activity
from .dataset import build_dataset
from .model_cnn import make_cnn

def main():
    cfg = Config()
    cfg.sessions = list(range(12, 22))

    session_to_activity = default_session_to_activity()

    X, y_str, groups = build_dataset(
        data_root=cfg.data_root,
        left_id=cfg.left_id,
        right_id=cfg.right_id,
        sessions=cfg.sessions,
        fs=cfg.fs,
        win_sec=cfg.win_sec,
        hop_sec=cfg.hop_sec,
        session_to_activity=session_to_activity,
        clap_search_sec=cfg.clap_search_sec,
        trim_after_clap_sec=cfg.trim_after_clap_sec,
        drop_start_sec=cfg.drop_start_sec,
        drop_end_sec=cfg.drop_end_sec,
        remove_idle=cfg.remove_idle,
        idle_energy_threshold=cfg.idle_energy_threshold
    )

    classes = sorted(np.unique(y_str))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)

    print("\nClasses:", classes)
    print("Total windows:", len(X))

    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # Normalize (train stats)
    mean = Xtr.mean(axis=(0,1), keepdims=True)
    std = Xtr.std(axis=(0,1), keepdims=True) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    model = make_cnn(input_shape=Xtr.shape[1:], n_classes=len(classes))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5)
    ]

    model.fit(
        Xtr, ytr,
        validation_split=0.2,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    probs = model.predict(Xte, verbose=0)
    yp = probs.argmax(axis=1)

    print("\n=== TEST REPORT (split by subject: amy vs thasa) ===")
    print(classification_report(yte, yp, target_names=classes))
    print("Confusion matrix:\n", confusion_matrix(yte, yp))

    os.makedirs(cfg.out_dir, exist_ok=True)
    model_path = os.path.join(cfg.out_dir, "ax6_two_wrist_cnn.keras")
    model.save(model_path)

    np.save(os.path.join(cfg.out_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(cfg.out_dir, "norm_std.npy"), std)
    pd.Series(classes).to_csv(os.path.join(cfg.out_dir, "classes.csv"), index=False)

    print("\nSaved:")
    print(" -", model_path)
    print(" -", os.path.join(cfg.out_dir, "norm_mean.npy"))
    print(" -", os.path.join(cfg.out_dir, "norm_std.npy"))
    print(" -", os.path.join(cfg.out_dir, "classes.csv"))

if __name__ == "__main__":
    main()
