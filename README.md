# Human Activity Recognition using Bidirectional LSTM  
**Two-Wrist AX6 Sensor System**

## Overview

This project implements a Human Activity Recognition (HAR) system using motion data recorded from two AX6 wearable sensors placed on the left and right wrists.

Each sensor provides:
- Tri-axial accelerometer: `ax, ay, az`
- Tri-axial gyroscope: `gx, gy, gz`

The original pipeline used a Convolutional Neural Network (CNN).  
This version replaces the CNN with a **Bidirectional LSTM** to better model temporal dependencies in the motion signals.

Recognized activities:
- Walking
- Jogging
- Jumping Jacks
- Sit-ups
- Squats

---

## Measurement Setup (Messprotokoll)

**Sensor configuration**
- Sampling frequency: **100 Hz**
- Accelerometer range: **±8 g**
- Gyroscope range: **±250 °/s**

**Device IDs**
- Left wrist: `6033722`
- Right wrist: `6032357`

### Recording procedure
1. Clap hands **3 times** (synchronization signal)
2. Start recording
3. Perform activity
4. Clap 3 times again
5. Stop recording

The clap events are used to:
- Synchronize left and right sensors
- Extract the active segment of the recording

---

## Dataset

Sessions used: **12–21**

| Activity | Sessions |
|---|---|
| Walk | 12, 13 |
| Jogging | 14, 15 |
| Sit-ups | 16, 17 |
| Jumping Jacks | 18, 19 |
| Squats | 20, 21 |

Subjects:
- Even sessions → **Amy**
- Odd sessions → **Thasa**

Active segments are stored in:

data/active_sections_tensors/

Files per session:

<session>part1.csv (left wrist)
<session>part2.csv (right wrist)


---

## Data Processing Pipeline

### 1. Resampling
All signals are resampled to:

fs = 100 Hz


### 2. Sliding Window Segmentation

- Window length: **1.0 second** (100 samples)
- Hop size: **0.5 seconds** (50% overlap)

Each window becomes one sample:

Input shape: (100 timesteps, 6 features)


Both wrists are treated as independent samples.

Dataset output:

X : [N, 100, 6]
y : [N]
g : [N] # subject id


---

## Model Architecture

File: `src/model/model_lstm.py`

Input (100 × 6)
↓
Bidirectional LSTM (128)
↓
Dropout (0.3)
↓
Bidirectional LSTM (64)
↓
Dropout (0.3)
↓
Dense (64, ReLU)
↓
Dropout (0.5)
↓
Softmax (5 classes)


### Why LSTM?
Human motion is sequential. LSTM networks:
- Capture temporal dependencies
- Model movement dynamics over time
- Use context within the full time window

---

## Training

Run:

python -m src.model.train


Steps:
1. Build dataset
2. Encode labels
3. Stratified random split (train / validation / test)
4. Normalize using training mean and std
5. Train using:
   - Adam optimizer
   - Cross-entropy loss
   - Early stopping
   - Learning rate scheduling

Saved outputs:

out/
├── ax6_two_wrist_lstm.keras
├── norm_mean.npy
├── norm_std.npy
└── classes.csv


---

## Evaluation

Run:

python -m src.model.predict


Metrics:
- Accuracy
- Precision / Recall / F1-score
- Confusion matrix

Typical performance:

Accuracy ≈ 0.96 – 0.97


Most confusion occurs between **Sit-ups** and **Squats** due to similar motion patterns.

---

## Important Note

Evaluation uses **random window splitting**.

Since windows from the same recording may appear in both training and testing sets:
- Results reflect **within-subject performance**
- Performance may be optimistic

Cross-subject evaluation (Leave-One-Subject-Out) can be performed to assess generalization.

---

## Project Structure

src/
├── config.py
├── preprocessing/
│ ├── dataset.py
│ ├── io_ax6.py
│ └── clap_sync.py
└── model/
├── model_lstm.py
├── train.py
└── predict.py
data/
notebooks/
out/


---

## Requirements

Install dependencies:
pip install -r requirements.txt




