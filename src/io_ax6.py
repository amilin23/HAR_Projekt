import os, glob
import pandas as pd
import numpy as np

def find_session_file(data_root: str, sensor_id: str, session: int) -> str:
    """
    Supports:
      data/raw/<sensor_id>/<sensor_id>_0000000012.csv
    or:
      data/raw/<sensor_id>_0000000012.csv
    """
    p1 = os.path.join(data_root, sensor_id, f"{sensor_id}_00000000{session:02d}.csv")
    p2 = os.path.join(data_root, f"{sensor_id}_00000000{session:02d}.csv")

    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2

    g1 = glob.glob(os.path.join(data_root, sensor_id, f"{sensor_id}_*{session:02d}.csv"))
    g2 = glob.glob(os.path.join(data_root, f"{sensor_id}_*{session:02d}.csv"))
    cand = sorted(g1 + g2)
    if not cand:
        raise FileNotFoundError(f"Missing file for sensor={sensor_id}, session={session}")
    return cand[0]

def load_ax6_csv(path: str) -> pd.DataFrame:
    """
    Normalizes to columns:
      t, ax, ay, az, gx, gy, gz
    Assumes first column is time and next 6 are sensor channels.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 7:
        raise ValueError(f"{path} has {df.shape[1]} columns; expected >= 7")

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).copy()

    cols = list(df.columns)
    sensor_cols = cols[1:7]
    out = df[[time_col] + sensor_cols].copy()
    out.columns = ["t", "ax", "ay", "az", "gx", "gy", "gz"]

    for c in ["ax","ay","az","gx","gy","gz"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out

def resample_to_fs(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    """
    AX6 exports often have timestamps that are not unique / low resolution.
    Resampling by timestamps can collapse the whole signal -> windows=0.

    So we keep the original sample order and create a synthetic uniform time axis.
    """
    df = df.copy().reset_index(drop=True)
    # synthetic time: 0, 1/fs, 2/fs, ...
    df["t"] = pd.to_timedelta(np.arange(len(df)) / fs, unit="s")
    return df

