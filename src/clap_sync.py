import numpy as np
from scipy.signal import find_peaks

def accel_mag_from_df(df) -> np.ndarray:
    return np.sqrt(df["ax"].values**2 + df["ay"].values**2 + df["az"].values**2)

def find_clap_index(mag: np.ndarray, fs: int, search_sec: float) -> int:
    """
    Find strongest peak in first search_sec seconds.
    Robust threshold using MAD.
    """
    n = min(len(mag), int(search_sec * fs))
    x = mag[:n].astype(float)
    x = x - np.median(x)

    mad = np.median(np.abs(x - np.median(x))) + 1e-9
    height = 6.0 * mad

    peaks, props = find_peaks(np.abs(x), height=height, distance=int(0.15*fs))
    if len(peaks) == 0:
        return int(np.argmax(np.abs(x)))
    return int(peaks[np.argmax(props["peak_heights"])])

def align_by_clap(dfL, dfR, fs: int, search_sec: float):
    """
    Align by shifting start so clap indices match.
    Returns dfL_aligned, dfR_aligned, shift (iR - iL), clap_idx_after (approx).
    """
    mL = accel_mag_from_df(dfL)
    mR = accel_mag_from_df(dfR)
    iL = find_clap_index(mL, fs, search_sec)
    iR = find_clap_index(mR, fs, search_sec)

    shift = iR - iL
    if shift > 0:
        dfL2 = dfL.iloc[shift:].reset_index(drop=True)
        dfR2 = dfR.reset_index(drop=True)
    elif shift < 0:
        dfR2 = dfR.iloc[-shift:].reset_index(drop=True)
        dfL2 = dfL.reset_index(drop=True)
    else:
        dfL2 = dfL.reset_index(drop=True)
        dfR2 = dfR.reset_index(drop=True)

    n = min(len(dfL2), len(dfR2))
    dfL2 = dfL2.iloc[:n].reset_index(drop=True)
    dfR2 = dfR2.iloc[:n].reset_index(drop=True)

    # clap index in aligned data (use left after alignment)
    clap_idx = find_clap_index(accel_mag_from_df(dfL2), fs, search_sec)
    return dfL2, dfR2, shift, clap_idx
