# Deepfake Audio Detection using CNN

##  Overview

Deepfake audio generated using modern AI voice synthesis systems can closely mimic human speech, making it increasingly difficult to distinguish between genuine and artificial recordings.

This project presents a Deep Learning based Deepfake Audio Detection system that classifies audio recordings as:

-  Genuine (Human Voice)
-  Deepfake (AI Generated Voice)

The model converts audio recordings into Mel Spectrogram representations and uses a Convolutional Neural Network (CNN) to learn discriminative patterns and artifacts associated with synthetic speech.

---

##  Features

- Deepfake vs Genuine audio classification
- Audio preprocessing and normalization
- Mel Spectrogram feature extraction
- CNN-based classification model
- Class imbalance handling using class weights
- Automatic model checkpointing
- Early stopping and learning rate scheduling
- Equal Error Rate (EER) evaluation
- Confusion Matrix visualization
- Performance report generation
- Python inference script for testing new audio samples
- Streamlit web app for interactive demo

---

##  Dataset

### Fake-or-Real (FoR) Dataset

**Dataset Link:** [kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

We use the **for-norm** split (LA Norm directory) as recommended.

Dataset Structure:
```
for-norm/
├── training/
│   ├── fake/
│   └── real/
├── validation/
│   ├── fake/
│   └── real/
└── testing/
    ├── fake/
    └── real/
```

| Split | Fake | Real | Total |
|-------|------|------|-------|
| Training | 26,927 | 26,941 | 53,868 |
| Validation | 5,398 | 5,400 | 10,798 |
| Testing | 2,370 | 2,264 | 4,634 |

---

##  Project Pipeline

### 1. Audio Preprocessing

- Load `.wav` audio files
- Convert audio to mono channel
- Resample audio to 16 kHz
- Fix audio length to 4 seconds
- Pad shorter samples
- Trim longer samples

### 2. Feature Extraction

Each audio file is transformed into a Mel Spectrogram.

```
Audio Signal
      ↓
Mel Spectrogram
      ↓
Normalized Feature Matrix
```

Configuration:
- Sampling Rate: 16,000 Hz
- Audio Length: 4 Seconds
- Number of Mel Bands: 64

### 3. CNN Architecture

The model consists of:

```
Input (64 × 126 × 1)
↓ Conv2D (32) + Batch Normalization + Max Pooling + Dropout(0.2)
↓ Conv2D (64) + Batch Normalization + Max Pooling + Dropout(0.2)
↓ Conv2D (128) + Batch Normalization + Max Pooling + Dropout(0.2)
↓ Conv2D (256) + Batch Normalization + Global Average Pooling
↓ Dense (512) + Dropout(0.5)
↓ Dense (2) + Softmax
```

Total Parameters: **522,370**

---

##  Training Strategy

### Optimizer
Adam Optimizer
```python
Learning Rate = 0.001
```

### Loss Function
```python
Sparse Categorical Crossentropy
```

### Regularization
- Batch Normalization
- Dropout Layers

### Callbacks
- ModelCheckpoint — saves best model based on val_accuracy
- ReduceLROnPlateau — reduces LR when accuracy plateaus
- EarlyStopping — stops training when no improvement (patience=7)

### Class Imbalance Handling
Class Weights computed using:
```python
compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
```

---

##  Results

| Metric | Value | Required | Status |
|--------|-------|----------|--------|
| Test Accuracy | **92.88%** | ≥ 80% |  PASS |
| F1 Score | **0.9279** | ≥ 0.80 |  PASS |
| Equal Error Rate (EER) | **7.10%** | ≤ 12% |  PASS |
| Fake Class Accuracy | **91.94%** | ≥ 75% |  PASS |
| Real Class Accuracy | **93.86%** | ≥ 75% |  PASS |

### Confusion Matrix

|  | Predicted Fake | Predicted Real |
|--|----------------|----------------|
| **Actual Fake** | 2,179  | 191 |
| **Actual Real** | 138  | 2,126  |

---

##  Training Visualization

The project generates:
- Training & Validation Accuracy Curve
- Training & Validation Loss Curve
- Confusion Matrix Heatmap

Generated file:
```
results.png
```

---

##  Repository Structure

```
deepfake-audio-detection/
├── final_notebook-2.ipynb       # Training notebook with full code
├── predict.py                 # Python inference script
├── app.py                     # Streamlit web application
├── best_model.keras           # Trained model
├── performance_report.json    # Metrics in JSON format
├── performance_report.pdf     # Detailed performance report
├── results.png                # Training plots and confusion matrix
└── README.md
```

---

##  Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Librosa
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

---

##  Running the Project

### Install Dependencies
```bash
pip install tensorflow librosa numpy scikit-learn matplotlib seaborn streamlit
```

### Run Training Notebook
```bash
jupyter notebook
```
Open `final_notebook-2.ipynb` and run all cells sequentially.

### Test on New Audio Sample (predict.py)
```bash
python predict.py --audio path/to/audio.wav --model best_model.keras
```

Example output:
```
==================================================
DEEPFAKE AUDIO DETECTION RESULT
==================================================
Audio File  : sample.wav
Prediction  : Deepfake (AI-Generated)
Confidence  : 96.40%
Fake Prob   : 96.40%
Real Prob   : 3.60%
==================================================
```

### Run Streamlit Web App
```bash
streamlit run app.py
```
- Upload a `.wav` audio file
- Get instant prediction: Genuine or Deepfake
- View confidence score

---

##  Future Improvements

- EfficientNet-based audio spectrogram classification
- Transformer-based architectures
- ASVspoof benchmark integration
- Real-time audio detection
- Explainable AI (XAI) visualizations

---

##  Author

Sakshi  
Electrical Engineering  
Indian Institute of Technology Roorkee

---

##  License

This project is developed for educational, research, and deepfake detection purposes.