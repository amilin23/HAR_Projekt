import numpy as np
from typing import Dict, Tuple

from .io_ax6 import find_session_file, load_ax6_csv, resample_to_fs
from .clap_sync import align_by_clap
from .config import session_to_subject

def build_two_wrist_tensor(dfL, dfR) -> np.ndarray:
    # [T, 12]: L(6) + R(6)
    return np.column_stack([
        dfL[["ax","ay","az","gx","gy","gz"]].values,
        dfR[["ax","ay","az","gx","gy","gz"]].values
    ])

def trim_after_clap(X: np.ndarray, clap_idx: int, fs: int,
                    trim_after_clap_sec: float, drop_start_sec: float, drop_end_sec: float) -> np.ndarray:
    start = clap_idx + int(trim_after_clap_sec*fs) + int(drop_start_sec*fs)
    end = len(X) - int(drop_end_sec*fs)
    if start >= end:
        return X
    return X[start:end]

def window_signal(X: np.ndarray, win: int, hop: int) -> np.ndarray:
    T, C = X.shape
    out = []
    for s in range(0, T - win + 1, hop):
        out.append(X[s:s+win])
    if not out:
        return np.zeros((0, win, C), dtype=np.float32)
    return np.stack(out).astype(np.float32)

def remove_idle_windows(W: np.ndarray, idle_energy_threshold: float) -> np.ndarray:
    if W.shape[0] == 0:
        return W
    # accel channels: L 0..2, R 6..8
    L = W[:, :, 0:3]
    R = W[:, :, 6:9]
    Lmag = np.sqrt((L**2).sum(axis=2))
    Rmag = np.sqrt((R**2).sum(axis=2))
    energy = (Lmag.std(axis=1) + Rmag.std(axis=1)) / 2.0
    keep = energy > idle_energy_threshold
    return W[keep]

def build_dataset(
    data_root: str,
    left_id: str,
    right_id: str,
    sessions,
    fs: int,
    win_sec: float,
    hop_sec: float,
    session_to_activity: Dict[int, str],
    clap_search_sec: float,
    trim_after_clap_sec: float,
    drop_start_sec: float,
    drop_end_sec: float,
    remove_idle: bool,
    idle_energy_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: [N, win, 12]
      y_str: [N] activity string
      groups: [N] subject string (amy/thasa)
    """
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)

    X_all, y_all, g_all = [], [], []

    for sess in sessions:
        activity = session_to_activity.get(sess)
        if activity is None:
            continue

        pL = find_session_file(data_root, left_id, sess)
        pR = find_session_file(data_root, right_id, sess)

        dfL = resample_to_fs(load_ax6_csv(pL), fs)
        dfR = resample_to_fs(load_ax6_csv(pR), fs)

        dfL_al, dfR_al, shift, clap_idx = align_by_clap(dfL, dfR, fs, clap_search_sec)

        X = build_two_wrist_tensor(dfL_al, dfR_al)
        X = trim_after_clap(X, clap_idx, fs, trim_after_clap_sec, drop_start_sec, drop_end_sec)

        W = window_signal(X, win, hop)
        if remove_idle:
            W = remove_idle_windows(W, idle_energy_threshold)

        subj = session_to_subject(sess)
        y = np.array([activity] * len(W))
        g = np.array([subj] * len(W))

        X_all.append(W)
        y_all.append(y)
        g_all.append(g)

        print(f"Session {sess:02d} | {activity:14s} | subj={subj:5s} | windows={len(W):4d} | shift(iR-iL)={shift}")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    g_all = np.concatenate(g_all, axis=0)
    return X_all, y_all, g_all
