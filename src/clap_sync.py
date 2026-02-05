from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def accel_mag_from_df(df) -> np.ndarray:
    return np.sqrt(df["ax"].values**2 + df["ay"].values**2 + df["az"].values**2)

def simple_clap_sync(dfL: pd.DataFrame, dfR: pd.DataFrame, search_sec: float=6, fs: int=100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simply align left and right sensor signal by searching for biggest in first [search_sec] seconds.
    
    :param dfL: left sensor
    :type dfL: pd.DataFrame
    :param dfR: right sensor
    :type dfR: pd.DataFrame
    :param search_sec: search in the first seconds of signal
    :type search_sec: float
    :param fs: sensor frequenzy
    :type fs: int
    :return: synchronized dataframes
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # calculate acceloration magnitude
    magL = accel_mag_from_df(dfL)
    magR = accel_mag_from_df(dfR)

    # simply search for the biggest peak in first [search_sec] seconds
    search_limit = search_sec * fs
    peakL = np.argmax(magL[:search_limit])
    peakR = np.argmax(magR[:search_limit])

    # calculate offset
    offset = peakR - peakL
    print(f"Offset: {offset}")

    # align data, according to offset
    if offset > 0:
        # dfL starts before dfR
        dfR_final = dfR.iloc[offset:].reset_index(drop=True)
        dfL_final = dfL.iloc[:len(dfR_final)].reset_index(drop=True)
    else:
        # dfL starts after dfR
        dfL_final = dfL.iloc[abs(offset):].reset_index(drop=True)
        dfR_final = dfR.iloc[:len(dfL_final)].reset_index(drop=True)

    # plotting for quality check
    plt.figure(figsize=(12, 4))
    plt.plot(dfL_final['ax'][:800], label="Left (ax)")
    plt.plot(dfR_final['ax'][:800], label="Rechts (ax)", alpha=0.7)
    plt.title("Quality Check (first 8 seconds)")
    plt.legend()
    plt.show()

    return dfL_final, dfR_final

# def find_clap_index(mag: np.ndarray, fs: int, search_sec: float) -> int:
#     """
#     Find strongest peak in first search_sec seconds.
#     Robust threshold using MAD.
#     """
#     n = min(len(mag), int(search_sec * fs))
#     x = mag[:n].astype(float)
#     x = x - np.median(x)

#     mad = np.median(np.abs(x - np.median(x))) + 1e-9
#     height = 6.0 * mad

#     peaks, props = signal.find_peaks(np.abs(x), height=height, distance=int(0.15*fs))
#     if len(peaks) == 0:
#         return int(np.argmax(np.abs(x)))
#     return int(peaks[np.argmax(props["peak_heights"])])

# def align_by_clap(dfL, dfR, fs: int, search_sec: float):
#     """
#     Align by shifting start so clap indices match.
#     Returns dfL_aligned, dfR_aligned, shift (iR - iL), clap_idx_after (approx).
#     """
#     mL = accel_mag_from_df(dfL)
#     mR = accel_mag_from_df(dfR)
#     iL = find_clap_index(mL, fs, search_sec)
#     iR = find_clap_index(mR, fs, search_sec)

#     shift = iR - iL
#     if shift > 0:
#         dfL2 = dfL.iloc[shift:].reset_index(drop=True)
#         dfR2 = dfR.reset_index(drop=True)
#     elif shift < 0:
#         dfR2 = dfR.iloc[-shift:].reset_index(drop=True)
#         dfL2 = dfL.reset_index(drop=True)
#     else:
#         dfL2 = dfL.reset_index(drop=True)
#         dfR2 = dfR.reset_index(drop=True)

#     n = min(len(dfL2), len(dfR2))
#     dfL2 = dfL2.iloc[:n].reset_index(drop=True)
#     dfR2 = dfR2.iloc[:n].reset_index(drop=True)

#     # clap index in aligned data (use left after alignment)
#     clap_idx = find_clap_index(accel_mag_from_df(dfL2), fs, search_sec)
#     return dfL2, dfR2, shift, clap_idx