# src/features.py
import numpy as np
import pandas as pd
from scipy.stats import iqr

def window_indices(n, fs, win_s=3.0, step_s=1.5):
    win = int(win_s*fs); step = int(step_s*fs)
    i = 0
    while i + win <= n:
        yield i, i+win
        i += step

def time_feats(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "ptp": float(np.ptp(x)),
        "iqr": float(iqr(x)),
        "zcr": int(np.sum(np.diff(np.signbit(x)) != 0))
    }

def fft_bandpower(x, fs, bands=((0.2,0.6),(0.6,1.5),(1.5,3.0))):
    # simple rectangular-band integration
    n = len(x)
    X = np.fft.rfft(x * np.hanning(n))
    f = np.fft.rfftfreq(n, d=1.0/fs)
    p = np.abs(X)**2
    out = {}
    for (lo,hi) in bands:
        m = (f>=lo) & (f<=hi)
        out[f"bp_{lo:.1f}_{hi:.1f}"] = float(np.sum(p[m]) / (np.sum(p)+1e-9))
    return out

def orientation_feats(ax, ay, az):
    g = np.sqrt(np.maximum(ay**2 + az**2, 1e-12))
    pitch = np.degrees(np.arctan2(-ax, g))
    roll  = np.degrees(np.arctan2( ay, (az + 1e-9)))
    return {
        "pitch_med": float(np.median(pitch)),
        "pitch_std": float(np.std(pitch)),
        "roll_med":  float(np.median(roll)),
        "roll_std":  float(np.std(roll))
    }

def pair_diff_feats(L, R):
    # symmetry between wrists (e.g., running vs push-up patterns differ)
    feats = {}
    for c in ["ax","ay","az","gx","gy","gz"]:
        d = L[c].values - R[c].values
        feats[f"{c}_diff_mean"] = float(np.mean(d))
        feats[f"{c}_diff_std"]  = float(np.std(d))
    return feats

def extract_features_dual(dfL, dfR, fs, win_s=3.0, step_s=1.5):
    rows = []
    n = len(dfL)
    for i0, i1 in window_indices(n, fs, win_s, step_s):
        wL = dfL.iloc[i0:i1]; wR = dfR.iloc[i0:i1]
        row = {"t0": float(wL["t"].iloc[0]), "t1": float(wL["t"].iloc[-1])}

        for side, w in [("L", wL), ("R", wR)]:
            for c in ["ax","ay","az","gx","gy","gz"]:
                feats = time_feats(w[c].values)
                for k,v in feats.items(): row[f"{side}_{c}_{k}"] = v
                row.update({f"{side}_{c}_{bk}": bv for bk,bv in fft_bandpower(w[c].values, fs).items()})
            row.update({f"{side}_{k}": v for k,v in orientation_feats(w["ax"].values, w["ay"].values, w["az"].values).items()})

        row.update({f: v for f,v in pair_diff_feats(wL, wR).items()})
        rows.append(row)
    return pd.DataFrame(rows)
