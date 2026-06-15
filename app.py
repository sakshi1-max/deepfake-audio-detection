import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile

# Configuration
SAMPLE_RATE = 16000
MAX_LEN = 64000
N_MELS = 64

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

st.title("🎙 Deepfake Audio Detection")

st.write(
    "Upload a WAV audio file and the model will predict whether it is Genuine or Deepfake."
)

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = np.pad(
        audio,
        (0, max(0, MAX_LEN - len(audio)))
    )[:MAX_LEN]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-9
    )

    mel = mel[np.newaxis, ..., np.newaxis]

    probs = model.predict(
        mel,
        verbose=0
    )[0]

    pred_class = np.argmax(probs)

    confidence = float(
        np.max(probs) * 100
    )

    if pred_class == 1:
        label = "✅ Genuine (Human)"
    else:
        label = "❌ Deepfake (AI Generated)"

    st.subheader("Prediction")

    st.success(label)

    st.write(
        f"Confidence: {confidence:.2f}%"
    )

    st.write(
        f"Fake Probability: {probs[0]*100:.2f}%"
    )

    st.write(
        f"Real Probability: {probs[1]*100:.2f}%"
    )