import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from config import Config, default_session_to_activity
from dataset import build_dataset
from model_cnn import make_cnn

def main():
    cfg = Config()
    cfg.sessions = list(range(12, 22))
    session_to_activity = default_session_to_activity()

    # X: sensor data, y_str: labels, g: subjects
    X, y_str, g = build_dataset(
        sessions=cfg.sessions,
        fs=cfg.fs,
        win_sec=cfg.win_sec,
        hop_sec=cfg.hop_sec,
        session_to_activity=session_to_activity
    )

    classes = sorted(np.unique(y_str))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=np.int64)

    print("\nClasses:", classes)
    print("Total windows:", len(X))

    # --- Simple train test split ---
    
    # Train+Val vs. Test (50/50)
    strat_labels = [f"{label}{group}" for label, group in zip(y, g)]
    X_train_val, Xtest, y_train_val, ytest, g_train_val, gtest = train_test_split(
        X, y, g,
        test_size=0.50, 
        random_state=cfg.random_state,
        stratify=strat_labels  # distribute labels and subjects equally
    )
    
    # Train vs. Val (80/20 from remaining)
    strat_labels = [f"{label}{group}" for label, group in zip(y_train_val, g_train_val)]
    Xtr, Xval, ytr, yval = train_test_split(
        X_train_val, y_train_val, 
        test_size=0.2, 
        random_state=cfg.random_state,
        stratify=strat_labels # distribute labels and subjects equally
    )

    print(f"Train: {len(Xtr)} | Val: {len(Xval)} | Test: {len(Xtest)}")

    # normalization
    mean = Xtr.mean(axis=(0, 1), keepdims=True)
    std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8

    Xtr = (Xtr - mean) / std
    Xval = (Xval - mean) / std 
    Xte = (Xtest - mean) / std

    # training
    model = make_cnn(input_shape=Xtr.shape[1:], n_classes=len(classes))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
    ]

    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # eval
    probs = model.predict(Xte, verbose=0)
    yp = probs.argmax(axis=1)

    print("\n=== TEST REPORT (Random Shuffle Split) ===")
    print(classification_report(ytest, yp, target_names=classes))
    cm = confusion_matrix(ytest, yp)
    print("Confusion matrix:\n", pd.DataFrame(cm, index=classes, columns=classes))

    # save
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.save(os.path.join(cfg.out_dir, "ax6_two_wrist_cnn.keras"))
    np.save(os.path.join(cfg.out_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(cfg.out_dir, "norm_std.npy"), std)
    pd.Series(classes).to_csv(os.path.join(cfg.out_dir, "classes.csv"), index=False)

    print(f"\nModell und Normierungswerte in {cfg.out_dir} gespeichert.")

if __name__ == "__main__":
    main()