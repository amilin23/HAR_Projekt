from __future__ import annotations

import pandas as pd


def load_ax6_csv(path: str) -> pd.DataFrame:
    """
    Load AX6 CSV saved from your pipeline (active sections).
    Must contain columns: ax, ay, az, gx, gy, gz
    Index or time column doesn't matter as long as data order is correct.
    """
    df = pd.read_csv(path)
    return df


def resample_to_fs(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    """
    Your active-section files are already effectively at the right rate in many cases.
    If you *do* have a timestamp column, you could resample properly, but for this project
    we keep it simple and assume fixed-rate data already.
    """
    # If your files already are at fs=100, do nothing:
    return df