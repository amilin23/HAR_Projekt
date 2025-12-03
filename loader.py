# src/loader.py
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def _read_any(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl" if path.suffix.lower()==".xlsx" else None)
    else:
        # try robust CSV (semicolon/comma/tab; dot/comma decimals)
        tried = []
        for sep, dec, header in [
            (",", ".", 0), (",", ".", None),
            (";", ",", 0), (";", ",", None),
            ("\t",".", 0), ("\t",".", None),
        ]:
            try:
                df = pd.read_csv(path, sep=sep, decimal=dec, header=header, engine="python")
                if df.shape[1] >= 4 and len(df) > 100:
                    break
            except Exception as e:
                tried.append((sep, dec, header, str(e)))
        else:
            raise ValueError(f"Could not parse {path}; tried: {tried}")
    return df

def _normalize_columns(df):
    # try to map to Time, Ax..Gz; support headerless or weird names
    ren = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["time","timestamp","date/time","datetime"]: ren[c]="Time"
        elif lc in ["ax","acc_x","accx","x"]: ren[c]="Ax"
        elif lc in ["ay","acc_y","accy","y"]: ren[c]="Ay"
        elif lc in ["az","acc_z","accz","z"]: ren[c]="Az"
        elif lc in ["gx","gyr_x","gyrox","gyroscopex","gyroscope_x"]: ren[c]="Gx"
        elif lc in ["gy","gyr_y","gyroy","gyroscopey","gyroscope_y"]: ren[c]="Gy"
        elif lc in ["gz","gyr_z","gyroz","gyroscopez","gyroscope_z"]: ren[c]="Gz"
    if ren:
        df = df.rename(columns=ren)

    # headerless fallback
    if "Time" not in df.columns:
        if df.shape[1] >= 7:
            df = df.iloc[:, :7]
            df.columns = ["Time","Ax","Ay","Az","Gx","Gy","Gz"]
        else:
            df = df.iloc[:, :4]
            df.columns = ["Time","Ax","Ay","Az"]
    for nm in ["Ax","Ay","Az"]:
        if nm not in df.columns:
            raise ValueError("Missing accel columns after normalization.")
    for nm in ["Gx","Gy","Gz"]:
        if nm not in df.columns:
            df[nm] = 0.0
    return df

def load_ax6(path, fs=100):
    df0 = _read_any(path)
    df0 = _normalize_columns(df0)

    # parse time
    tcol = df0["Time"]
    if np.issubdtype(tcol.dtype, np.number):
        t = tcol.astype(float).values
        if (t > 1e10).any(): t = t/1000.0
        idx = pd.to_datetime(t - t[0], unit="s", origin="unix")
    else:
        idx = pd.to_datetime(tcol, errors="coerce")
        if idx.isna().all():
            tt = pd.to_numeric(tcol, errors="coerce").fillna(0).values
            idx = pd.to_datetime(tt - tt[0], unit="s", origin="unix")

    df = pd.DataFrame(index=idx)
    for nm_src, nm_dst in zip(["Ax","Ay","Az","Gx","Gy","Gz"], ["ax","ay","az","gx","gy","gz"]):
        df[nm_dst] = pd.to_numeric(df0[nm_src], errors="coerce")

    # unit heuristics (g->m/s^2; rad/s->deg/s)
    acc_med = df[["ax","ay","az"]].abs().median().median()
    if 0.5 < acc_med < 3.5:
        df[["ax","ay","az"]] *= 9.80665
    gyro_med = df[["gx","gy","gz"]].abs().median().median()
    if 0.5 < gyro_med < 6.0:
        df[["gx","gy","gz"]] *= (180.0/np.pi)

    # resample to fs and add t
    df = df.sort_index().interpolate("time").resample(f"{int(1000/fs)}L").mean().interpolate(limit_direction="both")
    df.insert(0, "t", (df.index - df.index[0]).total_seconds())
    return df
