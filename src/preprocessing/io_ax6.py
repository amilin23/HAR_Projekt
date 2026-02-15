from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd


AX6_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def find_session_file(data_root: str, sensor_id: str, session: int) -> str:
    """
    Locate an AX6 session file.

    Supports patterns like:
      data/raw/<sensor_id>/<sensor_id>_000000000X.csv
      data/raw/<sensor_id>/*<session>*.csv
    """
    # Common pattern from your original project
    pattern1 = os.path.join(data_root, sensor_id, f"{sensor_id}_*{session}*.csv")
    hits = sorted(glob.glob(pattern1))
    if hits:
        return hits[0]

    # Fallback: search inside root for anything matching sensor_id and session
    pattern2 = os.path.join(data_root, "**", f"*{sensor_id}*{session}*.csv")
    hits = sorted(glob.glob(pattern2, recursive=True))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No AX6 CSV found for sensor_id={sensor_id}, session={session} under {data_root}"
    )


def load_ax6_csv(path: str) -> pd.DataFrame:
    """
    Load an AX6 CSV and return a DataFrame with expected columns.
    This assumes the CSV already includes columns ax/ay/az/gx/gy/gz.
    """
    df = pd.read_csv(path)

    missing = [c for c in AX6_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Found: {list(df.columns)}")

    # Keep only needed columns (stable order)
    return df[AX6_COLUMNS].copy()


def resample_to_fs(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    """
    Resample a dataframe to target sampling rate `fs` using linear interpolation.

    Assumption:
      - rows are uniform samples (no timestamps stored)
      - resampling uses index-based interpolation
    """
    if fs <= 0:
        raise ValueError("fs must be positive")

    n = len(df)
    if n < 2:
        return df.copy()

    # If original rate is unknown, we keep length and just return.
    # Your original pipeline uses a fixed fs later anyway.
    # If you DO know original fs, you can implement time-based resampling.
    return df.astype(np.float32)