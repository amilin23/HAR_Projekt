import os
import numpy as np
import pandas as pd

from .config import Config, default_session_to_activity
from .io_ax6 import find_session_file, load_ax6_csv, resample_to_fs
from .clap_sync import align_by_clap
from .dataset import build_two_wrist_tensor, trim_after_clap, window_signal, remove_idle_windows

from tensorflow import keras

def predict_session(session_id: int):
    cfg = Config()
    cfg.sessions = list(range(12, 22))

    model = keras.models.load_model(os.path.join(cfg.out_dir, "ax6_two_wrist_cnn.keras"))
    mean = np.load(os.path.join(cfg.out_dir, "norm_mean.npy"))
    std  = np.load(os.path.join(cfg.out_dir, "norm_std.npy"))
    classes = pd.read_csv(os.path.join(cfg.out_dir, "classes.csv"), header=None)[0].tolist()

    pL = find_session_file(cfg.data_root, cfg.left_id, session_id)
    pR = find_session_file(cfg.data_root, cfg.right_id, session_id)

    dfL = resample_to_fs(load_ax6_csv(pL), cfg.fs)
    dfR = resample_to_fs(load_ax6_csv(pR), cfg.fs)

    dfL_al, dfR_al, shift, clap_idx = align_by_clap(dfL, dfR, cfg.fs, cfg.clap_search_sec)
    X = build_two_wrist_tensor(dfL_al, dfR_al)
    X = trim_after_clap(X, clap_idx, cfg.fs, cfg.trim_after_clap_sec, cfg.drop_start_sec, cfg.drop_end_sec)

    win = int(cfg.win_sec * cfg.fs)
    hop = int(cfg.hop_sec * cfg.fs)
    W = window_signal(X, win, hop)
    if cfg.remove_idle:
        W = remove_idle_windows(W, cfg.idle_energy_threshold)

    if len(W) == 0:
        print("No windows produced (too short or trimmed too much).")
        return

    Wn = (W - mean) / std
    probs = model.predict(Wn, verbose=0)
    yp = probs.argmax(axis=1)

    # Majority vote
    counts = np.bincount(yp, minlength=len(classes))
    top = int(np.argmax(counts))
    print(f"Session {session_id:02d} predicted activity: {classes[top]}")
    print("Window vote distribution:")
    for i,c in enumerate(classes):
        print(f"  {c:14s}: {counts[i]}")

if __name__ == "__main__":
    for sid in range(12, 22):
        print("\n" + "="*40)
        predict_session(sid)
