# src/train_rf.py
import argparse, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader import load_ax6
from sync import align_left_right
from features import extract_features_dual

def label_windows_from_intervals(feats_df, intervals_df):
    labels = []
    for r in feats_df.itertuples(index=False):
        mid = 0.5*(r.t0 + r.t1)
        lab = None
        for itv in intervals_df.itertuples(index=False):
            if (mid >= itv.start_s) and (mid <= itv.end_s):
                lab = itv.label; break
        labels.append(lab if lab is not None else "other")
    feats_df["label"] = labels
    return feats_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw", help="folder with session_* subfolders")
    ap.add_argument("--out_dir",   default="models", help="where to save model")
    ap.add_argument("--fs", type=int, default=100)
    ap.add_argument("--win_s", type=float, default=3.0)
    ap.add_argument("--step_s", type=float, default=1.5)
    ap.add_argument("--single_label", type=str, default="", help="if set, assign this label to all windows in this run")
    args = ap.parse_args()

    X_all, y_all = [], []
    sessions = sorted(Path(args.data_root).glob("session_*"))
    assert sessions, f"No sessions in {args.data_root}"

    for sess in sessions:
        left = sess / "Left_10.csv"
        right = sess / "Right_10.csv"
        assert left.exists() and right.exists(), f"Missing left/right in {sess}"

        dfL = load_ax6(left, fs=args.fs)
        dfR = load_ax6(right, fs=args.fs)
        dfL, dfR, lag = align_left_right(dfL, dfR, fs=args.fs)

        feats = extract_features_dual(dfL, dfR, fs=args.fs, win_s=args.win_s, step_s=args.step_s)

        if args.single_label:
            feats["label"] = args.single_label
        else:
            lab_path = sess / "labels.csv"
            assert lab_path.exists(), f"Need labels.csv in {sess} (or use --single_label)"
            intervals = pd.read_csv(lab_path)
            # expected columns: start_s, end_s, label
            feats = label_windows_from_intervals(feats, intervals)

        # accumulate
        y_all.extend(feats["label"].values.tolist())
        feats = feats.drop(columns=["label","t0","t1"])
        X_all.append(feats)

    X = pd.concat(X_all, axis=0).reset_index(drop=True)
    y = np.array(y_all)

    # split & train
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1
    ).fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    print(classification_report(yte, ypred))

    os.makedirs(args.out_dir, exist_ok=True)
    dump(clf, Path(args.out_dir)/"rf_dual.pkl")
    X.columns.to_series().to_csv(Path(args.out_dir)/"rf_dual_featnames.csv", index=False)
    with open(Path(args.out_dir)/"rf_dual_labels.txt","w") as f:
        for c in sorted(set(y)): f.write(str(c)+"\n")
    print("Saved model to", Path(args.out_dir)/"rf_dual.pkl")

if __name__ == "__main__":
    main()
