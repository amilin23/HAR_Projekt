from __future__ import annotations

import pandas as pd


def load_ax6_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def resample_to_fs(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    return df