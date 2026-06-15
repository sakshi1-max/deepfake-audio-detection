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

st.markdown(
    """
    <h1 style='text-align:center;'>
    🎙 Deepfake Audio Detection
    </h1>
    <p style='text-align:center; font-size:18px;'>
    AI-powered detection of synthetic and genuine speech
    </p>
    """,
    unsafe_allow_html=True
)

st.write("Model Loaded Successfully ✅")
st.write("Input Shape:", model.input_shape)
st.write("Output Shape:", model.output_shape)

st.write(
    "Upload a WAV audio file and the model will predict whether it is Genuine or Deepfake."
)

uploaded_file = st.file_uploader(
    "Upload Audio in .wav ",
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


    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8,3))
    ax.imshow(mel[0,:,:,0], aspect='auto', origin='lower')
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)


    probs = model.predict(
    mel,
    verbose=0
    )[0]

    confidence = float(
    np.max(probs) * 100

    )
    st.write("Mel Shape:", mel.shape)
    st.write("Mel Mean:", float(mel.mean()))
    st.write("Mel Std:", float(mel.std()))
    st.write("Audio Length:", len(audio)/16000)
    pred_class = np.argmax(probs)
    col1, col2, col3 = st.columns(3)

    st.write("Predicted Class:", pred_class)
    st.write("Raw Probabilities:", probs)

    

    col1.metric("Confidence", f"{confidence:.2f}%")
    col2.metric("Fake", f"{probs[0]*100:.2f}%")
    col3.metric("Real", f"{probs[1]*100:.2f}%")

    st.subheader("Probability Distribution")

    st.progress(float(probs[1]))
    st.write(f"Real Probability: {probs[1]*100:.2f}%")

    st.progress(float(probs[0]))
    st.write(f"Fake Probability: {probs[0]*100:.2f}%")

    if pred_class == 1:
        label = " ✅ Genuine (Human)"
    else:
        label = "❌ Deepfake (AI Generated)"

    st.subheader("Prediction")


    if pred_class == 1:
        st.success("✅ Genuine Human Voice Detected")
    else:
        st.error("🚨 AI Generated Deepfake Detected")


    st.write(
        f"Confidence: {confidence:.2f}%"
    )

    st.write(
        f"Fake Probability: {probs[0]*100:.2f}%"
    )

    st.write(
        f"Real Probability: {probs[1]*100:.2f}%"
    )
    st.write(model.input_shape)
st.write(model.output_shape)
model = load_model()



st.sidebar.title("About")

st.sidebar.markdown("""
### Deepfake Audio Detection System

- TensorFlow CNN Model
- Mel Spectrogram Features
- Binary Classification
- Genuine vs Deepfake
""")


# col1, col2, col3 = st.columns(3)

# col1.metric("Confidence", f"{confidence:.2f}%")
# col2.metric("Fake", f"{probs[0]*100:.2f}%")
# col3.metric("Real", f"{probs[1]*100:.2f}%")
# st.subheader("Probability Distribution")

# st.progress(float(probs[1]))
# st.write(f"Real Probability: {probs[1]*100:.2f}%")

# st.progress(float(probs[0]))
# st.write(f"Fake Probability: {probs[0]*100:.2f}%")

# st.markdown("---")
# st.caption(
#     "Built using TensorFlow, Librosa and Streamlit"
# )