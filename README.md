# Human Activity Recognition using Bidirectional LSTM (AX6 IMU)

End-to-end Human Activity Recognition (HAR) pipeline using wrist-worn inertial sensors and deep learning.  
This project includes preprocessing, sliding-window dataset generation, Bidirectional LSTM training, multiple evaluation strategies, and inference for activity timeline prediction. Bidirectional LSTM processes the sequence in both forward and backward directions, allowing the model to use past and future context within each window. This improves recognition of motion patterns that depend on the full temporal structure.

Why LSTM? 
- Learns temporal dynamics (periodicity, peaks, transitions)
- Bidirectional context improves within-window pattern recognition

---

## Project Overview

The objective is to classify physical activities from wearable IMU signals recorded from wrist-mounted AX6 devices.

### Activities (8 Classes)
- walk
- jogging
- situp
- squat
- jump_squat
- jumping_jacks
- lunge
- pumping

### Sensors
- Device: AX6
- Location: Left wrist and Right wrist
- Signals per wrist:
  - Accelerometer: ax, ay, az
  - Gyroscope: gx, gy, gz
- Sampling rate: **100 Hz**

---

## Independent Wrist Training

This project uses **independent-wrist learning**:

- Left wrist windows are treated as samples
- Right wrist windows are treated as samples
- No sensor fusion
- Model input shape: **[window, 6 channels]**

**Why this design?**
- Doubles dataset size
- More stable training with limited data

---

## Dataset

- Sessions: **12 – 27**
- Subjects: **2 (Amilin, Thasa)**
- Total windows: **1398**

### Data Split (Random Shuffle)
| Split | Windows |
|---|---|
Train | 838 |
Validation | 210 |
Test | 350 |

### Window Configuration
- Window length: **1.0 sec (100 samples)**
- Hop size: **0.5 sec (50 samples)**
- 50% overlap

---

## Processing Pipeline

Raw AX6 CSV  
↓  
Active segment extraction  
↓  
Resample to 100 Hz  
↓  
Sliding window (1 s, 0.5 s hop)  
↓  
Global normalization (train-only)  
↓  
Bidirectional LSTM  
↓  
Window-level prediction  
↓  
Session-level majority voting

---

## Repository Structure


```
HAR_Projekt/
│
├── src/
│ ├── config.py
│ ├── preprocessing/
│ │ ├── dataset.py
│ │ └── io_ax6.py
│ │
│ └── model/
│ ├── model_lstm.py
│ ├── train.py
│ ├── predict.py
│ └── eval_loso_subject.py
│
├── notebooks/
│ ├── sync_and_extract.ipynb
│ └── trim_data.ipynb
│
├── out/ # Saved model and normalization

```

## Normalization
Global normalization is computed using training data only

## Model Architecture
Bidirectional LSTM for temporal sequence modeling.
Input: [100 timesteps, 6 channels]
Architecture:

- Bidirectional LSTM (128 units)
- Dropout (0.30)
- Bidirectional LSTM (64 units)
- Dropout (0.30)
- Dense (64, ReLU)
- Dropout (0.50)
- Dense (8, Softmax)

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 40 |
| Batch size | 32 |
| Optimizer | Adam (lr = 1e-3) |
| Loss | Sparse categorical crossentropy |

**Callbacks**
- EarlyStopping(patience=10, restore_best_weights=True)
- ReduceLROnPlateau(patience=5, factor=0.5)

---

## Evaluation

### 1. Random Shuffle Split

Stratified by:
- Activity label
- Subject

**Results**

| Metric | Value |
|---|---|
| Window Accuracy | 55% |
| Session Accuracy | 87.5% |

Session-level prediction uses majority voting:


### 2. Leave-One-Subject-Out (LOSO)

Train on one subject and evaluate on the unseen subject.

**Result**

Window Accuracy: **9.6%**

This indicates poor cross-subject generalization due to limited subject diversity.

---

## Performance Analysis

**Best performing class**
- Situp (F1 ≈ 0.76)

**Common confusions**
- Squat ↔ Lunge
- Jump Squat ↔ Lunge
- Jogging ↔ Jumping Jacks

**Reason**
Similar lower-body motion dynamics across activities.
Wrist-only sensors make some leg-dominant activities ambiguous (lunge vs squat)
Lunge → often predicted as squat (similar movement dynamics at wrist)
Jumping_jacks ↔ jogging (both are rhythmic, high-energy patterns)
Some pumping windows predicted as lunge when arm motion overlaps


---

## Training

Run training:

```bash
python -m src.model.train
```

**Saved artifacts**

```
out/
├── ax6_lstm_independent_wrist.keras
├── norm_mean.npy
├── norm_std.npy
├── classes.txt
```

---

## Model Evaluation

Evaluate the trained model on the test split:

```bash
python -m src.model.predict
```

This script:

- Loads the saved model and normalization parameters
- Rebuilds the dataset
- Performs the same stratified test split
- Reports:
  - Window-level accuracy
  - Classification report
  - Confusion matrix
  - Session-level accuracy (majority vote)


## LOSO Evaluation

```bash
python -m src.model.eval_loso_subject
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Main libraries**
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn

---

## Limitations

- Only 2 subjects
- Poor cross-subject generalization
- Similar activities cause confusion
- Independent wrist training (no sensor fusion)

---

## Future Work

- Collect data from more subjects
- Two-wrist sensor fusion (12-channel model)
- CNN-LSTM hybrid architecture
- Data augmentation
---

## Key Results

- Window Accuracy: **55%**
- Session Accuracy: **87.5%**
- LOSO Accuracy: **9.6%**
- Main limitation: **dataset size and subject diversity**

