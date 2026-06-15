import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import matplotlib.pyplot as plt

# ─── Configuration — must match training exactly ───────────────────────────────
SAMPLE_RATE  = 16000
MAX_LEN      = 64000   # 4 seconds
N_MELS       = 64
HOP_LENGTH   = 512

# EER from performance_report = 11.05%  →  threshold 0.50 is correct
# "Threshold Used: 1.0000" in your report was a bug in notebook saving
THRESHOLD    = 0.50
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Deepfake Audio Detection", page_icon="🎙")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center;'>🎙 Deepfake Audio Detection</h1>
    <p style='text-align:center; font-size:18px;'>
        AI-powered detection of synthetic and genuine speech
    </p>
""", unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.markdown("""
### Deepfake Audio Detection System
- TensorFlow CNN Model
- Mel Spectrogram Features (64×126)
- Binary Classification: Genuine vs Deepfake
- FOR-Norm Dataset Trained
""")
st.sidebar.markdown(f"**Decision Threshold:** `{THRESHOLD}`")
st.sidebar.markdown(f"**Model Accuracy:** `88.95%`")
st.sidebar.markdown(f"**EER:** `11.05%`")

# ─── Upload ────────────────────────────────────────────────────────────────────
st.write("Upload a `.wav` audio file — model will predict Genuine or Deepfake.")
uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # ─── Preprocess — exactly like training ────────────────────────────────────
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Pad or truncate to exactly 4 seconds
    audio = np.pad(audio, (0, max(0, MAX_LEN - len(audio))))[:MAX_LEN]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    mel = mel[np.newaxis, ..., np.newaxis]   # shape: (1, 64, 126, 1)

    # ─── Debug info (remove after testing) ────────────────────────────────────
    with st.expander("🔍 Debug Info"):
        st.write("Mel Shape:", mel.shape)
        st.write("Expected :", model.input_shape)
        st.write("Audio Duration:", f"{len(audio)/SAMPLE_RATE:.2f} sec")
        st.write("Mel Mean:", f"{float(mel.mean()):.4f}")
        st.write("Mel Std :", f"{float(mel.std()):.4f}")

    # ─── Mel Spectrogram plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(mel[0, :, :, 0], aspect='auto', origin='lower', cmap='magma')
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bins")
    st.pyplot(fig)

    # ─── Prediction ───────────────────────────────────────────────────────────
    probs     = model.predict(mel, verbose=0)[0]
    real_prob = float(probs[1])
    fake_prob = float(probs[0])

    # Use threshold (NOT argmax)
    pred_class = 1 if real_prob >= THRESHOLD else 0
    confidence = real_prob * 100 if pred_class == 1 else fake_prob * 100

    # ─── Result display ───────────────────────────────────────────────────────
    st.subheader("Prediction")
    if pred_class == 1:
        st.success(" Genuine Human Voice Detected")
    else:
        st.error(" AI Generated Deepfake Detected")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{confidence:.2f}%")
    col2.metric("Fake Prob",  f"{fake_prob * 100:.2f}%")
    col3.metric("Real Prob",  f"{real_prob * 100:.2f}%")

    st.subheader("Probability Distribution")
    st.progress(real_prob)
    st.write(f"Real: {real_prob * 100:.2f}%")
    st.progress(fake_prob)
    st.write(f"Fake: {fake_prob * 100:.2f}%")

    st.markdown("---")
    st.caption("Built using TensorFlow · Librosa · Streamlit")