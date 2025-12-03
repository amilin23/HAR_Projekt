# src/sync.py
import numpy as np
import pandas as pd
from scipy.signal import correlate

def smv(df):  # signal magnitude vector
    ax, ay, az = [df[c].to_numpy(float) for c in ("ax","ay","az")]
    return np.sqrt(ax**2 + ay**2 + az**2)

def align_left_right(df_left, df_right, fs=100, max_shift_s=2.0):
    a = smv(df_left) - np.mean(smv(df_left))
    b = smv(df_right) - np.mean(smv(df_right))
    max_shift = int(max_shift_s * fs)
    c = correlate(a, b, mode="full")
    lags = np.arange(-len(b)+1, len(a))
    mask = (lags >= -max_shift) & (lags <= max_shift)
    best_lag = lags[mask][np.argmax(c[mask])]
    # positive best_lag => left is ahead; shift left forward
    if best_lag > 0:
        df_left = df_left.iloc[best_lag:].copy()
        df_right = df_right.iloc[:len(df_left)].copy()
    elif best_lag < 0:
        df_right = df_right.iloc[-best_lag:].copy()
        df_left = df_left.iloc[:len(df_right)].copy()
    n = min(len(df_left), len(df_right))
    return df_left.iloc[:n].copy(), df_right.iloc[:n].copy(), float(best_lag)/fs
