from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from src.config import ACTIVE_SECTIONS_DATA_ROOT, session_to_subject
from src.preprocessing.io_ax6 import load_ax6_csv, resample_to_fs


def window_signal(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """
    Sliding window segmentation.

    Args:
      x: [T, C]
      win: window length in samples
      hop: hop length in samples

    Returns:
      windows: [N, win, C]
    """
    if win <= 0 or hop <= 0:
        raise ValueError("win and hop must be > 0")

    T, C = x.shape
    if T < win:
        return np.zeros((0, win, C), dtype=np.float32)

    out = []
    for start in range(0, T - win + 1, hop):
        out.append(x[start : start + win])

    return np.stack(out).astype(np.float32) if out else np.zeros((0, win, C), dtype=np.float32)


def build_dataset(
    fs: int,
    win_sec: float,
    hop_sec: float,
    sessions: List[int],
    session_to_activity: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build window dataset from already prepared ACTIVE sections.

    File naming convention:
      <ACTIVE_SECTIONS_DATA_ROOT>/<session>part1.csv
      <ACTIVE_SECTIONS_DATA_ROOT>/<session>part2.csv

    Returns:
      X: [N, win, C] (each wrist contributes samples)
      y_str: [N] string labels
      g: [N] subject id
    """
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    g_all: List[np.ndarray] = []

    for sess in sessions:
        label = session_to_activity.get(sess)
        if label is None:
            continue

        base = os.path.join(ACTIVE_SECTIONS_DATA_ROOT, f"{sess}part")
        p1 = base + "1.csv"
        p2 = base + "2.csv"
        if not (os.path.exists(p1) and os.path.exists(p2)):
            raise FileNotFoundError(f"Missing active-section files: {p1} / {p2}")

        t1 = resample_to_fs(load_ax6_csv(p1), fs).values.astype(np.float32)
        t2 = resample_to_fs(load_ax6_csv(p2), fs).values.astype(np.float32)

        W1 = window_signal(t1, win, hop)
        W2 = window_signal(t2, win, hop)

        subj = session_to_subject(sess)

        # ✅ FIX: y1 must match len(W1), y2 must match len(W2)
        y1 = np.array([label] * len(W1), dtype=object)
        y2 = np.array([label] * len(W2), dtype=object)
        g1 = np.array([subj] * len(W1), dtype=object)
        g2 = np.array([subj] * len(W2), dtype=object)

        X_all.extend([W1, W2])
        y_all.extend([y1, y2])
        g_all.extend([g1, g2])

        print(
            f"Session {sess:02d} | {label:14s} | subj={subj:5s} | windows={len(W1):4d} / {len(W2):4d}"
        )

    if not X_all:
        raise RuntimeError("No data loaded. Check ACTIVE_SECTIONS_DATA_ROOT and session mapping.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    g = np.concatenate(g_all, axis=0)
    return X, y, g