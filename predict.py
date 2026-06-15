"""
Deepfake Audio Detection - Inference Script
Usage: python predict.py --audio path/to/audio.wav --model path/to/best_model.keras
"""

import argparse
import numpy as np
import librosa
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Same constants as notebook
SAMPLE_RATE = 16000
MAX_LEN     = 64000  # 4 seconds
N_MELS      = 64
BATCH_SIZE  = 32


def preprocess_audio(audio_path):
    """
    Preprocess a single audio file — same pipeline as training notebook:
    1. Load audio at 16kHz mono
    2. Pad/truncate to 4 seconds (64000 samples)
    3. Extract Mel Spectrogram (64 mels)
    4. Convert to dB scale
    5. Normalize
    """
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = np.pad(audio, (0, max(0, MAX_LEN - len(audio))))[:MAX_LEN]

    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    # Shape: (1, 64, 126, 1) — batch of 1
    mel = mel.astype(np.float32)
    mel = mel[np.newaxis, ..., np.newaxis]
    return mel


def predict(audio_path, model_path='best_model.keras'):
    """
    Predict whether an audio file is Genuine (Human) or Deepfake (AI-Generated).

    Args:
        audio_path: Path to .wav audio file
        model_path: Path to trained .keras model

    Returns:
        label: 'Genuine (Human)' or 'Deepfake (AI-Generated)'
        confidence: confidence score (0-100%)
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Preprocess audio
    print(f"Processing audio: {audio_path}")
    mel = preprocess_audio(audio_path)

    # Predict
    probs = model.predict(mel, verbose=0)[0]  # [fake_prob, real_prob]
    fake_prob = probs[0]
    real_prob = probs[1]

    if real_prob >= fake_prob:
        label = "Genuine (Human)"
        confidence = real_prob * 100
    else:
        label = "Deepfake (AI-Generated)"
        confidence = fake_prob * 100

    return label, confidence, fake_prob, real_prob


def main():
    parser = argparse.ArgumentParser(description='Deepfake Audio Detection')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to input audio file (.wav)')
    parser.add_argument('--model', type=str, default='best_model.keras',
                        help='Path to trained model file (default: best_model.keras)')
    args = parser.parse_args()

    # Run prediction
    label, confidence, fake_prob, real_prob = predict(args.audio, args.model)

    # Print results
    print("\n" + "="*50)
    print("DEEPFAKE AUDIO DETECTION RESULT")
    print("="*50)
    print(f"Audio File  : {args.audio}")
    print(f"Prediction  : {label}")
    print(f"Confidence  : {confidence:.2f}%")
    print(f"Fake Prob   : {fake_prob*100:.2f}%")
    print(f"Real Prob   : {real_prob*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()