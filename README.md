Deepfake Audio Detection using CNN

What is this project?
AI voice synthesis has gotten scarily good. Modern text-to-speech systems can now mimic real human voices so convincingly that it's genuinely hard to tell the difference just by listening. This project is my attempt at tackling that problem — a deep learning system that listens to an audio clip and tells you whether it's a real human voice or an AI-generated one.
Under the hood, it converts audio into Mel Spectrograms (basically a visual fingerprint of sound) and runs them through a CNN trained to spot the subtle artifacts that synthetic voices leave behind.

Features

Binary classification: Genuine vs. Deepfake audio
Audio preprocessing pipeline (mono, 16kHz, fixed length)
Mel Spectrogram feature extraction
CNN model with Batch Norm + Dropout regularization
Handles class imbalance using computed class weights
Auto saves the best model during training
Early stopping + learning rate scheduling
Equal Error Rate (EER) evaluation
Confusion Matrix and training curves visualization
predict.py for quick command-line inference
Streamlit web app for an interactive demo


Dataset
I used the Fake-or-Real (FoR) Dataset — specifically the for-norm split.
Link: kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
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
SplitFakeRealTotalTraining26,92726,94153,868Validation5,3985,40010,798Testing2,3702,2644,634

How it works
Step 1 — Audio Preprocessing
Every audio file goes through the same pipeline:

Load .wav → convert to mono → resample to 16kHz
Fix length to exactly 4 seconds (pad if shorter, trim if longer)

Step 2 — Feature Extraction
Each clip is converted into a Mel Spectrogram — a 2D representation of how frequency content changes over time. This is what the CNN actually "sees."
Config: SR = 16000 Hz | Length = 4s | Mel Bands = 64
Step 3 — CNN Model
Input (64 × 126 × 1)
  ↓ Conv2D (32) → BatchNorm → MaxPool → Dropout(0.2)
  ↓ Conv2D (64) → BatchNorm → MaxPool → Dropout(0.2)
  ↓ Conv2D (128) → BatchNorm → MaxPool → Dropout(0.2)
  ↓ Conv2D (256) → BatchNorm → GlobalAvgPool
  ↓ Dense (512) → Dropout(0.5)
  ↓ Dense (2) → Softmax
Total Parameters: 522,370

Training Details

Optimizer: Adam (lr = 0.001)
Loss: Sparse Categorical Crossentropy
Regularization: Batch Normalization + Dropout
Callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping (patience=7)
Class imbalance: Handled via compute_class_weight('balanced', ...)


Results
MetricValueRequiredStatusTest Accuracy88.95%≥ 80%✅ PASSF1 Score0.8873≥ 0.80✅ PASSEqual Error Rate (EER)11.05%≤ 12%✅ PASSFake Class Accuracy88.86%≥ 75%✅ PASSReal Class Accuracy89.05%≥ 75%✅ PASS
Confusion Matrix
Predicted FakePredicted RealActual Fake2,106 ✅264 ❌Actual Real248 ❌2,016 ✅

Repo Structure
deepfake-audio-detection/
├── final_notebook.ipynb    # Full training code
├── predict.py              # Command-line inference
├── app.py                  # Streamlit web app
├── best_model.keras        # Saved model weights
├── performance_report.json
├── performance_report.pdf
├── results.png             # Training plots + confusion matrix
└── README.md

Running the Project
Install dependencies
bashpip install tensorflow librosa numpy scikit-learn matplotlib seaborn streamlit
Train the model
Open final_notebook.ipynb in Jupyter and run all cells.
Test on a new audio file
bashpython predict.py --audio path/to/audio.wav --model best_model.keras
Sample output:
==================================================
DEEPFAKE AUDIO DETECTION RESULT
==================================================
Audio File  : sample.wav
Prediction  : Deepfake (AI-Generated)
Confidence  : 96.40%
Fake Prob   : 96.40%
Real Prob   : 3.60%
==================================================
Run the web app
bashstreamlit run app.py
Upload a .wav file and get an instant result with a confidence score.

What's next?
A few things I'd like to explore going forward:

Swapping CNN for EfficientNet on spectrograms
Trying Transformer-based audio models
Testing against the ASVspoof benchmark
Real-time detection from microphone input
Adding XAI visualizations (Grad-CAM on spectrograms)


Author
Sakshi (24115129)

Electrical Engineering

IIT Roorkee

Built for educational and research purposes — specifically, to make AI-generated audio a little harder to fake undetected.