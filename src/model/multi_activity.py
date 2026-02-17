from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import Config
from src.preprocessing.io_ax6 import load_ax6_csv, resample_to_fs

SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def _load_classes(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _windowize(X: np.ndarray, win: int, hop: int) -> np.ndarray:
    T = X.shape[0]
    if T < win:
        return np.empty((0, win, X.shape[1]), dtype=np.float32)
    starts = np.arange(0, T - win + 1, hop, dtype=np.int64)
    out = np.stack([X[s:s + win] for s in starts], axis=0)
    return out.astype(np.float32), starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="Path to LEFT wrist CSV")
    ap.add_argument("--right", required=True, help="Path to RIGHT wrist CSV")
    ap.add_argument("--out_csv", default="out/multi_activity_predictions.csv", help="Output CSV path")
    ap.add_argument("--smooth_k", type=int, default=1, help="Majority smooth over k windows (odd). 1 = no smoothing.")
    args = ap.parse_args()

    cfg = Config()
    model_path = os.path.join(cfg.out_dir, "ax6_two_wrist_lstm.keras")
    mean_path = os.path.join(cfg.out_dir, "norm_mean.npy")
    std_path = os.path.join(cfg.out_dir, "norm_std.npy")
    classes_path = os.path.join(cfg.out_dir, "classes.csv")

    if not all(os.path.exists(p) for p in [model_path, mean_path, std_path, classes_path]):
        raise FileNotFoundError("Missing model artifacts in out/. Run train first.")

    model = keras.models.load_model(model_path)
    mean = np.load(mean_path)
    std = np.load(std_path)
    classes = _load_classes(classes_path)

    dfL = resample_to_fs(load_ax6_csv(args.left), cfg.fs)
    dfR = resample_to_fs(load_ax6_csv(args.right), cfg.fs)

    XL = dfL[SENSOR_COLS].to_numpy(dtype=np.float32)
    XR = dfR[SENSOR_COLS].to_numpy(dtype=np.float32)

    T = min(XL.shape[0], XR.shape[0])
    XL = XL[:T]; XR = XR[:T]

    X = np.concatenate([XL, XR], axis=-1)  # [T,12]

    win = int(round(cfg.win_sec * cfg.fs))
    hop = int(round(cfg.hop_sec * cfg.fs))

    W, starts = _windowize(X, win, hop)  # [N,win,12], starts in samples

    if W.shape[0] == 0:
        raise RuntimeError("Recording too short for one window.")

    # Normalize with training stats
    Wn = (W - mean) / (std + 1e-8)

    yp = model.predict(Wn, verbose=0).argmax(axis=1)

    # Optional smoothing (majority over +/-k//2)
    k = int(args.smooth_k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1

    if k > 1:
        half = k // 2
        yp_smooth = []
        for i in range(len(yp)):
            lo = max(0, i - half)
            hi = min(len(yp), i + half + 1)
            vals = yp[lo:hi].tolist()
            yp_smooth.append(max(set(vals), key=vals.count))
        yp = np.array(yp_smooth, dtype=np.int64)

    times_sec = starts / float(cfg.fs)
    labels = [classes[i] for i in yp]

    out = pd.DataFrame({
        "t_start_sec": times_sec,
        "t_end_sec": times_sec + cfg.win_sec,
        "pred_class": labels,
        "pred_idx": yp,
    })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved timeline predictions to: {args.out_csv}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()