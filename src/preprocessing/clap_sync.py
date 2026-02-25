from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd


def accel_magnitude(df: pd.DataFrame) -> np.ndarray:
    """Compute acceleration magnitude from ax/ay/az."""
    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy()
    return np.sqrt(ax * ax + ay * ay + az * az)


def simple_clap_sync(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    search_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Align left/right streams by finding a large accel peak in the first `search_samples`.

    Returns:
      (df_left_aligned, df_right_aligned, shift)
    where shift > 0 means right is shifted forward.
    """
    L = accel_magnitude(df_left)[:search_samples]
    R = accel_magnitude(df_right)[:search_samples]

    if len(L) == 0 or len(R) == 0:
        return df_left.copy(), df_right.copy(), 0

    iL = int(np.argmax(L))
    iR = int(np.argmax(R))
    shift = iR - iL

    if shift > 0:
        # right starts later -> drop first 'shift' from right
        df_right_al = df_right.iloc[shift:].reset_index(drop=True)
        df_left_al = df_left.iloc[: len(df_right_al)].reset_index(drop=True)
    elif shift < 0:
        # left starts later -> drop first '-shift' from left
        df_left_al = df_left.iloc[-shift:].reset_index(drop=True)
        df_right_al = df_right.iloc[: len(df_left_al)].reset_index(drop=True)
    else:
        df_left_al = df_left.copy().reset_index(drop=True)
        df_right_al = df_right.copy().reset_index(drop=True)

    return df_left_al, df_right_al, shift
