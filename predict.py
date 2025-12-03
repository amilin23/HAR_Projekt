# src/predict.py
import argparse, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from loader import load_ax6
from sync import align_left_right
from features import extract_features_dual

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left",  required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--model", default="models/rf_dual.pkl")
    ap.add_argument("--featnames", default="models/rf_dual_featnames.csv")
    ap.add_argument("--fs", type=int, default=100)
    ap.add_argument("--win_s", type=float, default=3.0)
    ap.add_argument("--step_s", type=float, default=1.5)
    ap.add_argument("--out", default="data/processed/pred")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    dfL = load_ax6(args.left, fs=args.fs)
    dfR = load_ax6(args.right, fs=args.fs)
    dfL, dfR, lag = align_left_right(dfL, dfR, fs=args.fs)

    feats = extract_features_dual(dfL, dfR, fs=args.fs, win_s=args.win_s, step_s=args.step_s)
    t0 = feats["t0"].values; t1 = feats["t1"].values
    X = feats.drop(columns=["t0","t1"])

    # match training feature order
    featnames = pd.read_csv(args.featnames, header=None)[0].tolist()
    X = X.reindex(columns=featnames, fill_value=0.0)

    clf = load(args.model)
    y = clf.predict(X.values)

    out_df = pd.DataFrame({"t0": t0, "t1": t1, "label": y})
    out_df.to_csv(Path(args.out)/"timeline.csv", index=False)
    with open(Path(args.out)/"timeline.json","w") as f:
        json.dump([{"t0": float(a), "t1": float(b), "label": str(c)} for a,b,c in zip(t0,t1,y)], f, indent=2)

    print("Saved:", Path(args.out)/"timeline.csv")

if __name__ == "__main__":
    main()
