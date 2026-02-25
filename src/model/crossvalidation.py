from typing import Literal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split

from src.preprocessing.dataset import build_dataset
from src.model.train import _fit_normalizer, _apply_normalizer, _encode_labels
from src.model.model_lstm import make_lstm
from src.config import Config, default_session_to_activity

random_state = 42
tf.random.set_seed(42)
np.random.seed(42)

def create_kfold(X, y, mode: Literal["stratify", "group"], n=3, shuffle=True):
    """Create n splits. Either stratified (equally distributed) or grouped (no overlap) based on y.

    Args:
        X (N, 6|12): Sensor Data
        y (N): Labels to be stratified or grouped by
        mode ("stratify" or "group"): Equally distributed or no overlap between splits
        n (int): Number of splits (only affects stratification)
    """
    if mode == "stratify":
        kfold = StratifiedKFold(n, shuffle=shuffle, random_state=random_state)
    if mode == "group":
        kfold = GroupKFold(len(np.unique(y)), shuffle=True, random_state=random_state)
    
    return [split for split in kfold.split(np.zeros(X.shape[0]), y, y)]

def train_on_split(Xtr, ytr, Xte, yte, classes, Xval=None, yval=None):
    # Global normalization
    mean, std = _fit_normalizer(Xtr)
    Xtr = _apply_normalizer(Xtr, mean, std)
    Xte = _apply_normalizer(Xte, mean, std)
    #Xval = _apply_normalizer(Xval, mean, std)

    model = make_lstm(input_shape=Xtr.shape[1:], n_classes=len(classes))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
    ]

    model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=Config.epochs,
        batch_size=Config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # use same validation data for restored best weights
    probs = model.predict(Xte, verbose=0)
    yp = probs.argmax(axis=1)
    cm_raw = confusion_matrix(yte, yp)
    cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)
    
    return {
        "acc": accuracy_score(yte, yp),
        "f1": f1_score(yte, yp, average="weighted", zero_division=0),
        "cm": cm_norm,
        "cm_raw": cm_raw,
    }


def visualize(results: pd.DataFrame, classes):
    output_path = os.path.join(os.getcwd(), "out")
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "crossvalidition")
    os.makedirs(output_path, exist_ok=True)

    # Strip whitespace/newlines from split labels that came from print-titles
    results = results.copy()
    results["split"] = results["split"].str.strip()

    # Overlap vs Split
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, sensor in zip(axes.flat, ["single", "combined"]):
        sns.boxplot(
            data=results[results["sensor"] == sensor],
            x="split", y="f1", hue="overlap",
            ax=ax, palette="bright"
        )
        ax.set_title(f"{sensor} Wrist – F1 per Split-Strategy & Overlap")
        ax.set_xlabel("Split Strategy")
        ax.set_ylabel("F1")
        ax.tick_params(axis='x', rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha('right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "overlap_vs_split.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Single vs. Combined
    # Align both sensors on same x-axis (all split strategies), combined leaves NaN gaps
    overlaps = results["overlap"].unique()
    all_splits = sorted(results["split"].unique())
    n_overlaps = len(overlaps)
    fig, axes = plt.subplots(1, n_overlaps, figsize=(7 * n_overlaps, 5), sharey=True)
    if n_overlaps == 1:
        axes = [axes]
    for ax, overlap in zip(axes, overlaps):
        sns.boxplot(
            data=results[results["overlap"] == overlap],
            x="split", y="f1", hue="sensor",
            order=all_splits, ax=ax
        )
        ax.set_title(f"Overlap: {overlap}")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha('right')
    # Single shared x-axis label
    fig.text(0.5, -0.04, "Split Strategy", ha='center', va='center', fontsize=11)
    fig.suptitle("Single vs. Combined Wrist – F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "single_vs_combined.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Heatmap – shared color scale for fair comparison 
    pivot = results.groupby(["sensor", "overlap", "split"])["f1"].mean().unstack("split")
    vmin_h = pivot.min().min()
    vmax_h = pivot.max().max()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (sensor, group) in zip(axes.flat, pivot.groupby(level="sensor")):
        sns.heatmap(
            group.droplevel("sensor"), annot=True, fmt=".3f",
            cmap="YlGnBu", ax=ax, vmin=vmin_h, vmax=vmax_h
        )
        ax.set_title(f"{sensor} Wrist")
        ax.tick_params(axis='x', rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha('right')
    plt.suptitle("Mean F1-Score (Overlap × Split-Strategy)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "heatmap_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion Matrices
    # Build mean normalized and summed absolute CMs per configuration
    cm_norm_mean = {}
    cm_abs_sum = {}
    for name, group in results.groupby(["sensor", "overlap", "split"]):
        cm_norm_mean[name] = np.array([row["cm"] for _, row in group.iterrows()]).mean(axis=0)
        cm_abs_sum[name] = np.array([row["cm_raw"] for _, row in group.iterrows()]).sum(axis=0)

    sensors_u = list(results["sensor"].unique())
    overlaps_u = list(results["overlap"].unique())
    splits_u = sorted(results["split"].unique())
    row_keys = [(s, o) for s in sensors_u for o in overlaps_u]
    nrows = len(row_keys)
    ncols = len(splits_u)

    def _draw_cm_figure(cm_dict, fmt, title_suffix, filename, vmin=None, vmax=None):
        cell_size = max(3.0, 22 / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * cell_size, nrows * cell_size * 0.95))
        axes = np.array(axes).reshape(nrows, ncols)
        for ax in axes.flat:
            ax.set_visible(False)
        
        # Dynamic shared color scale
        all_vals = np.concatenate([m.flatten() for m in cm_dict.values()])
        _vmin = all_vals.min() if vmin is None else vmin
        _vmax = all_vals.max() if vmax is None else vmax

        for (sensor, overlap, split), cm in cm_dict.items():
            row_idx = row_keys.index((sensor, overlap))
            col_idx = splits_u.index(split)
            ax = axes[row_idx, col_idx]
            ax.set_visible(True)
            sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                        xticklabels=classes, yticklabels=classes,
                        vmin=_vmin, vmax=_vmax, cbar=False)
            ax.set_title(f"{sensor} | {overlap} | {split}", fontsize=7, pad=3)
            ax.set_xlabel("Predicted", fontsize=7)
            ax.set_ylabel("True", fontsize=7)
            ax.tick_params(axis='x', rotation=30, labelsize=6)
            ax.tick_params(axis='y', rotation=0, labelsize=6)
            for lbl in ax.get_xticklabels():
                lbl.set_ha('right')

        plt.suptitle(f"Confusion Matrices – {title_suffix}", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, filename), dpi=150, bbox_inches='tight')
        plt.close()

    _draw_cm_figure(cm_norm_mean, ".1f", "Mean normalized (avg over folds)",
                    "confusion_matrices_normalized.png", vmin=0, vmax=1)
    _draw_cm_figure(cm_abs_sum,   "d",   "Absolute sample counts (summed over folds)",
                    "confusion_matrices_absolute.png")

def main():
    sess_to_activity = default_session_to_activity()
    sessions = [int(s) for s in sess_to_activity.keys()]
    
    # Building datasets for seperate sensors
    singleWrist_50pOverlap = build_dataset(sessions, Config.fs, 1, 0.5, sess_to_activity, "single")
    singleWrist_25pOverlap = build_dataset(sessions, Config.fs, 1, 0.75, sess_to_activity, "single")
    singleWrist_0pOverlap = build_dataset(sessions, Config.fs, 1, 1, sess_to_activity, "single")
    singleWrist = [singleWrist_50pOverlap, singleWrist_25pOverlap, singleWrist_0pOverlap]
    
    # Building datasets for combined sensors
    combinedWrist_50pOverlap = build_dataset(sessions, Config.fs, 1, 0.5, sess_to_activity, "combined")
    combinedWrist_25pOverlap = build_dataset(sessions, Config.fs, 1, 0.75, sess_to_activity, "combined")
    combinedWrist_0pOverlap = build_dataset(sessions, Config.fs, 1, 1, sess_to_activity, "combined")
    combinedWrist = [combinedWrist_50pOverlap, combinedWrist_25pOverlap, combinedWrist_0pOverlap]
    
    print(f"\n\n{"-" * 50}\n")
    
    # SEPERATE SENSORS
    print(f"{"-" * 20}Single Wrist evaluation{"-" * 20}")
    results = []
    for overlap_label, window_set in zip(["50%", "25%", "0%"], singleWrist):
        print(f"{"-"*15}{overlap_label}{"-"*15}")
        X, y, g, s, w = window_set
        classes, y = _encode_labels(y)
        #Xtr, Xval, ytr, yval, gtr, _, str, _, wtr, _ = train_test_split(X, y, g, s, w, test_size=0.15, stratify=y, random_state=random_state)
        
        splits_subjectsGrouped = create_kfold(X, g, "group")
        splits_subjectsStratified = create_kfold(X, [f"{activity}{s}" for activity, s in zip(y, g)], "stratify")
        splits_wristsGrouped = create_kfold(X, w, "group")
        splits_completlyStratified = create_kfold(X, [f"{activity}{s}{w}" for activity, s, w in zip(y, g, w)], "stratify")
        splits_list = [splits_subjectsGrouped, splits_subjectsStratified, splits_wristsGrouped, splits_completlyStratified]
        
        titles = ["\n\nGrouped Subjects\n\n", "Stratified Subjects\n\n", "\n\nGrouped Wrists\n\n", "\n\nCompletly Stratified\n\n"]
        for split_label, splits in zip(titles, splits_list):
            print(f"{"-"*10}{split_label}{"-"*10}")
            for i, (train_idcs, test_idcs) in enumerate(splits, start=1):
                print(f"{"-"*5}Fold: {i}{"-"*5}")
                result = train_on_split(X[train_idcs], y[train_idcs], X[test_idcs], y[test_idcs], classes)
                results.append({
                    "sensor": "single",
                    "overlap": overlap_label,
                    "split": split_label,
                    "fold": i,
                    "acc": result["acc"],
                    "f1": result["f1"],
                    "cm": result["cm"].tolist(),
                    "cm_raw": result["cm_raw"].tolist(),
                })
        
    print(f"\n{"-" * 50}\n")
    # COMBINED SENSORS
    print(f"{"-" * 20}Combined Wrist evaluation {"-" * 20}")
    for overlap_label, window_set in zip(["50%", "25%", "0%"], combinedWrist):
        print(f"{"-"*15}{overlap_label}{"-"*15}")
        X, y, g, s, w = window_set
        classes, y = _encode_labels(y)
        #Xtr, Xte, ytr, yte, gtr, _, str, _ = train_test_split(X, y, g, s, test_size=0.15, stratify=y, random_state=random_state)
        
        splits_subjectsGrouped = create_kfold(X, g, "group")
        splits_subjectsStratified = create_kfold(X, [f"{activity}{s}" for activity, s in zip(y, g)], "stratify")
        splits_list = [splits_subjectsGrouped, splits_subjectsStratified]
        
        titles = ["\n\nGrouped Subjects\n\n", "\n\nStratified Subjects\n\n"]
        for title, splits in zip(titles, splits_list):
            print(f"{"-"*10}{title}{"-"*10}")
            for i, (train_idcs, test_idcs) in enumerate(splits, start=1):
                print(f"{"-"*5}Fold: {i}{"-"*5}")
                result = train_on_split(X[train_idcs], y[train_idcs], X[test_idcs], y[test_idcs], classes)
                results.append({
                    "sensor": "combined",
                    "overlap": overlap_label,
                    "split": title,
                    "fold": i,
                    "acc": result["acc"],
                    "f1": result["f1"],
                    "cm": result["cm"].tolist(),
                    "cm_raw": result["cm_raw"].tolist(),
                })
                
    results = pd.DataFrame(results)
    visualize(results, classes)
                
if __name__ == "__main__":
    main()