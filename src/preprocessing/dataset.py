from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.config import session_to_subject
from src.preprocessing.io_ax6 import load_ax6_csv, resample_to_fs

DEFAULT_ACTIVE_ROOT = os.path.join("ax6_cnn_project", "data", "active_sections_tensors")
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]  # 6 per wrist


def _windowize(X: np.ndarray, win: int, hop: int) -> np.ndarray:
    """X: [T, C] -> windows: [N, win, C]"""
    T = X.shape[0]
    if T < win:
        return np.empty((0, win, X.shape[1]), dtype=np.float32)
    starts = np.arange(0, T - win + 1, hop, dtype=np.int64)
    out = np.stack([X[s:s + win] for s in starts], axis=0)
    return out.astype(np.float32)


def _to_sensor_matrix(df: pd.DataFrame, fs: int) -> np.ndarray:
    """Return [T,6] in SENSOR_COLS order."""
    df = resample_to_fs(df, fs)
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")
    return df[SENSOR_COLS].to_numpy(dtype=np.float32)


def build_dataset(
    sessions: Iterable[int],
    fs: int,
    win_sec: float,
    hop_sec: float,
    session_to_activity: Dict[int, str],
    active_root: str = DEFAULT_ACTIVE_ROOT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    g_list: List[str] = []
    s_list: List[int] = []

    for sid in sessions:
        if sid not in session_to_activity:
            continue

        pL = os.path.join(active_root, f"{sid}part1.csv")
        pR = os.path.join(active_root, f"{sid}part2.csv")
        if not os.path.exists(pL) or not os.path.exists(pR):
            raise FileNotFoundError(f"Missing active tensors for session {sid}: {pL} / {pR}")

        dfL = load_ax6_csv(pL)
        dfR = load_ax6_csv(pR)

        XL = _to_sensor_matrix(dfL, fs)  # [T,6]
        XR = _to_sensor_matrix(dfR, fs)  # [T,6]

        WL = _windowize(XL, win, hop)  # [NL, win, 6]
        WR = _windowize(XR, win, hop)  # [NR, win, 6]

        if WL.shape[0] == 0 and WR.shape[0] == 0:
            print(f"Session {sid:02d} produced 0 windows. Skipping.")
            continue

        activity = session_to_activity[sid]
        subject = session_to_subject(sid)

        # Add left windows
        if WL.shape[0] > 0:
            X_list.append(WL)
            y_list.extend([activity] * WL.shape[0])
            g_list.extend([subject] * WL.shape[0])
            s_list.extend([sid] * WL.shape[0])

        # Add right windows
        if WR.shape[0] > 0:
            X_list.append(WR)
            y_list.extend([activity] * WR.shape[0])
            g_list.extend([subject] * WR.shape[0])
            s_list.extend([sid] * WR.shape[0])

        print(f"Session {sid:02d} | {activity:<14} | subj={subject:<6} | windows={WL.shape[0]:4d} + {WR.shape[0]:4d}")

    if not X_list:
        raise RuntimeError("No data loaded. Check DEFAULT_ACTIVE_ROOT and input files.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y_str = np.asarray(y_list, dtype=object)
    g = np.asarray(g_list, dtype=object)
    s = np.asarray(s_list, dtype=np.int64)
    return X, y_str, g, s
