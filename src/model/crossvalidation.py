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
    cm = confusion_matrix(yte, yp)
    
    return {
        "acc": accuracy_score(yte, yp),
        "f1": f1_score(yte, yp, average="weighted", labels=classes, zero_division=np.nan),
        "cm": cm
    }


def visualize(results: pd.DataFrame, classes):
    # Overlap comparison for each strategy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, sensor in zip(axes.flat, ["single", "combined"]):
        sns.boxplot(
            data=results[results["sensor"] == sensor],
            x="split", y="f1", hue="overlap", 
            ax=ax,
            palette="bright"
        )
        ax.set_title(f"{sensor} Wrist – F1 per Split-Strategy & Overlap")
        ax.set_xlabel("Split")
        ax.set_ylabel("F1")
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig("overlap_vs_split.png", dpi=150)

    # Single vs. Combined
    fig, axes = plt.subplots(1, max(2, len(results["overlap"].unique())), figsize=(15, 5), sharey=True)
    for ax, overlap in zip(axes.flat, results["overlap"].unique()):
        sns.boxplot(
            data=results[results["overlap"] == overlap],
            x="split", y="f1", hue="sensor",
            ax=ax
        )
        ax.set_title(f"Overlap: {overlap}")
        ax.tick_params(axis='x', rotation=20)
    plt.suptitle("Single vs. Combined Wrist – F1 Score", y=1.02)
    plt.tight_layout()
    plt.savefig("single_vs_combined.png", dpi=150)

    # Heatmap der mittleren F1-Scores (Übersicht) 
    pivot = results.groupby(["sensor", "overlap", "split"])["f1"].mean().unstack("split")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (sensor, group) in zip(axes.flat, pivot.groupby(level="sensor")):
        sns.heatmap(group.droplevel("sensor"), annot=True, fmt=".3f",
                    cmap="YlGnBu", ax=ax, vmin=0.5, vmax=1.0)
        ax.set_title(f"{sensor} Wrist")
    plt.suptitle("Mittlerer F1-Score (Overlap × Split-Strategie)")
    plt.tight_layout()
    plt.savefig("heatmap_overview.png", dpi=150)
    
    # Confusion Matrices – mean over folds per configuration
    confusion_matrices = {}
    for name, group in results.groupby(["sensor", "overlap", "split"]):
        cms = np.array([row["cm"] for _, row in group.iterrows()])
        confusion_matrices[name] = cms.mean(axis=0)
    
    configs = results.groupby(["sensor", "overlap", "split"])
    ncols = len(results["split"].unique())
    nrows = len(results["sensor"].unique()) * len(results["overlap"].unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), sharey=True)
    for ax, (name, _) in zip(axes.flat, configs):
        sns.heatmap(confusion_matrices[name], annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
        ax.set_title(f"{name[0]} | {name[1]} | {name[2]}", fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis='x', rotation=20)
    plt.suptitle("Confusion Matrices – Mean over Folds")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)

def main():
    sess_to_activity = default_session_to_activity()
    sessions = [int(s) for s in sess_to_activity.keys()]
    
    # Building datasets for seperate sensors
    singleWrist_50pOverlap = build_dataset(sessions, Config.fs, 1, 0.5, sess_to_activity, "single")
    # singleWrist_25pOverlap = build_dataset(sessions, Config.fs, 1, 0.75, sess_to_activity, "single")
    # singleWrist_0pOverlap = build_dataset(sessions, Config.fs, 1, 1, sess_to_activity, "single")
    singleWrist = [singleWrist_50pOverlap]#, singleWrist_25pOverlap, singleWrist_0pOverlap]
    
    # Building datasets for combined sensors
    combinedWrist_50pOverlap = build_dataset(sessions, Config.fs, 1, 0.5, sess_to_activity, "combined")
    #combinedWrist_25pOverlap = build_dataset(sessions, Config.fs, 1, 0.75, sess_to_activity, "combined")
    #combinedWrist_0pOverlap = build_dataset(sessions, Config.fs, 1, 1, sess_to_activity, "combined")
    combinedWrist = [combinedWrist_50pOverlap]#, combinedWrist_25pOverlap, combinedWrist_0pOverlap]
    
    print(f"\n\n{"-" * 50}\n")
    
    # SEPERATE SENSORS
    # print(f"{"-" * 20}Single Wrist evaluation{"-" * 20}")
    results = []
    for overlap_label, window_set in zip(["50%", "25%", "0%"], singleWrist):
        print(f"{"-"*15}{overlap_label}{"-"*15}")
        X, y, g, s, w = window_set
        classes, y = _encode_labels(y)
        #Xtr, Xval, ytr, yval, gtr, _, str, _, wtr, _ = train_test_split(X, y, g, s, w, test_size=0.15, stratify=y, random_state=random_state)
        
        splits_subjectsGrouped = create_kfold(X, g, "group")
        # splits_subjectsStratified = create_kfold(X, [f"{activity}{s}" for activity, s in zip(y, g)], "stratify")
        # splits_wristsGrouped = create_kfold(X, w, "group")
        # splits_completlyStratified = create_kfold(X, [f"{activity}{s}{w}" for activity, s, w in zip(y, g, w)], "stratify")
        splits_list = [splits_subjectsGrouped]#, splits_subjectsStratified, splits_wristsGrouped, splits_completlyStratified]
        
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
                    "cm": result["cm"]
                })
        
    print(f"\n{"-" * 40}\n")
    # COMBINED SENSORS
    print(f"{"-" * 20}Combined Wrist evaluation {"-" * 20}")
    for overlap_label, window_set in zip(["50%", "25%", "0%"], combinedWrist):
        print(f"{"-"*15}{overlap_label}{"-"*15}")
        X, y, g, s, w = window_set
        classes, y = _encode_labels(y)
        #Xtr, Xte, ytr, yte, gtr, _, str, _ = train_test_split(X, y, g, s, test_size=0.15, stratify=y, random_state=random_state)
        
        splits_subjectsGrouped = create_kfold(X, g, "group")
        #splits_subjectsStratified = create_kfold(X, [f"{activity}{s}" for activity, s in zip(y, g)], "stratify")
        splits_list = [splits_subjectsGrouped]#, splits_subjectsStratified]
        
        titles = ["\n\nGrouped Subjects\n\n", "\n\nStratified Subjects\n\n"]
        for title, splits in zip(titles, splits_list):
            print(f"{"-"*10}{title}{"-"*10}")
            for i, (train_idcs, test_idcs) in enumerate(splits):
                print(f"{"-"*5}Fold: {i}{"-"*5}")
                result = train_on_split(X[train_idcs], y[train_idcs], X[test_idcs], y[test_idcs], classes)
                results.append({
                    "sensor": "combined",
                    "overlap": overlap_label,
                    "split": title,
                    "fold": i,
                    "acc": result["acc"],
                    "f1": result["f1"],
                })
                
    results = pd.DataFrame(results)
    visualize(results, classes)
                
if __name__ == "__main__":
    main()