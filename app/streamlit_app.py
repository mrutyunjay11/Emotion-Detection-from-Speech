# =============================================================================
# Human Emotion Detection from Voice
# Streamlit Web Application
# =============================================================================
# A beautiful, interactive Streamlit app that lets users:
#   1. Upload an audio file (.wav)
#   2. Record voice from microphone
#   3. Detect the emotion expressed in the audio
# =============================================================================

import os
import sys
import json
import tempfile
import numpy as np
import joblib
import librosa
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import extract_features

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Emotion → emoji mapping for a richer UI
EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "neutral": "😐",
    "fearful": "😨",
}

# Emotion → color mapping for visual feedback
EMOTION_COLORS = {
    "happy": "#FFD700",
    "sad": "#4A90D9",
    "angry": "#FF4444",
    "neutral": "#AAAAAA",
    "fearful": "#9B59B6",
}


# ── Helper functions ─────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    """
    Load the trained model, scaler, and metadata from disk.
    Uses Streamlit's caching so files are loaded only once.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


def predict_emotion(audio_path: str, model, scaler, class_names: list[str]):
    """
    Extract features from an audio file and predict the emotion.

    Returns:
        emotion_label: Predicted emotion string.
        probabilities: Dict mapping each emotion to its probability.
    """
    features = extract_features(audio_path)
    features_scaled = scaler.transform(features.reshape(1, -1))

    prediction = model.predict(features_scaled)[0]
    emotion_label = class_names[prediction]

    # Get probability scores if the model supports it
    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
        probabilities = {class_names[i]: float(proba[i]) for i in range(len(class_names))}

    return emotion_label, probabilities


def plot_waveform(audio_path: str):
    """Plot the waveform of an audio file."""
    signal, sr = librosa.load(audio_path, sr=22050)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.linspace(0, len(signal) / sr, len(signal)), signal, color="#4A90D9", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    return fig


def plot_probability_bars(probabilities: dict):
    """Plot a horizontal bar chart of emotion probabilities."""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [EMOTION_COLORS.get(e, "#4A90D9") for e in emotions]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(emotions, probs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Probabilities")
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            color="white",
            fontweight="bold",
        )
    plt.tight_layout()
    return fig


# ── Streamlit App ─────────────────────────────────────────────────────────────

def main():
    # ── Page config ───────────────────────────────────────────────────────
    st.set_page_config(
        page_title="🎙️ Voice Emotion Detection",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS for premium look ───────────────────────────────────────
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .main { font-family: 'Inter', sans-serif; }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: #888;
            text-align: center;
            margin-top: 0;
            margin-bottom: 2rem;
        }

        .emotion-result {
            text-align: center;
            padding: 2rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #333;
            margin: 1rem 0;
        }
        .emotion-emoji {
            font-size: 4rem;
        }
        .emotion-label {
            font-size: 2rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-top: 0.5rem;
        }

        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.6rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: transform 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
        }

        .info-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #333;
            border-radius: 0.8rem;
            padding: 1.2rem;
            margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown('<h1 class="hero-title">🎙️ Voice Emotion Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Detect human emotions from voice using Machine Learning</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📋 About")
        st.markdown(
            """
            This app uses **Machine Learning** to detect emotions from voice recordings.

            **Supported Emotions:**
            - 😊 Happy
            - 😢 Sad
            - 😡 Angry
            - 😐 Neutral
            - 😨 Fearful

            **How it works:**
            1. Audio features (MFCC, Chroma, Spectral Contrast) are extracted
            2. A trained ML model predicts the emotion
            3. Confidence scores are displayed
            """
        )

        st.markdown("---")
        st.markdown("### 🛠️ Model Info")
        model, scaler, metadata = load_model()
        if metadata:
            st.markdown(f"**Model:** {metadata.get('best_model', 'N/A')}")
            st.markdown(f"**Accuracy:** {metadata.get('accuracy', 0):.2%}")
            st.markdown(f"**Features:** {metadata.get('feature_dim', 'N/A')} dimensions")
        else:
            st.warning("No trained model found. Please train a model first.")

    # ── Check if a model is loaded ────────────────────────────────────────
    model, scaler, metadata = load_model()

    if model is None:
        st.error(
            "⚠️ **No trained model found!**\n\n"
            "Please train a model first by running:\n"
            "```bash\n"
            "python src/train_model.py\n"
            "```\n"
            "Make sure the RAVDESS dataset is in the `data/` folder."
        )
        return

    class_names = metadata["classes"]

    # ── Input tabs ────────────────────────────────────────────────────────
    tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Voice"])

    # ── Tab 1: Upload Audio ───────────────────────────────────────────────
    with tab_upload:
        st.markdown("### Upload an audio file")
        st.markdown("Supported formats: **WAV, MP3, OGG, FLAC**")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "flac"],
            key="upload",
        )

        if uploaded_file is not None:
            # Save to a temporary file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Play the audio
            st.audio(uploaded_file, format=f"audio/{suffix.strip('.')}")

            # Detect emotion
            if st.button("🔍 Detect Emotion", key="detect_upload"):
                with st.spinner("Analyzing audio..."):
                    try:
                        emotion, probs = predict_emotion(tmp_path, model, scaler, class_names)

                        # Display result
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            emoji = EMOTION_EMOJIS.get(emotion, "🎵")
                            color = EMOTION_COLORS.get(emotion, "#4A90D9")
                            st.markdown(
                                f"""
                                <div class="emotion-result">
                                    <div class="emotion-emoji">{emoji}</div>
                                    <div class="emotion-label" style="color: {color};">{emotion}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with col2:
                            if probs:
                                st.markdown("#### Confidence Scores")
                                fig = plot_probability_bars(probs)
                                st.pyplot(fig)
                                plt.close(fig)

                        # Waveform
                        st.markdown("#### Waveform")
                        fig_wave = plot_waveform(tmp_path)
                        st.pyplot(fig_wave)
                        plt.close(fig_wave)

                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                    finally:
                        os.unlink(tmp_path)

    # ── Tab 2: Record Voice ───────────────────────────────────────────────
    with tab_record:
        st.markdown("### Record your voice")
        st.markdown("Click the microphone button below to record audio.")

        # Use Streamlit's built-in audio input
        audio_recording = st.audio_input("🎤 Click to record", key="recorder")

        if audio_recording is not None:
            # Save recording to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_recording.read())
                tmp_path = tmp.name

            # Play back the recording
            st.audio(audio_recording, format="audio/wav")

            if st.button("🔍 Detect Emotion", key="detect_record"):
                with st.spinner("Analyzing recording..."):
                    try:
                        emotion, probs = predict_emotion(tmp_path, model, scaler, class_names)

                        # Display result
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            emoji = EMOTION_EMOJIS.get(emotion, "🎵")
                            color = EMOTION_COLORS.get(emotion, "#4A90D9")
                            st.markdown(
                                f"""
                                <div class="emotion-result">
                                    <div class="emotion-emoji">{emoji}</div>
                                    <div class="emotion-label" style="color: {color};">{emotion}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with col2:
                            if probs:
                                st.markdown("#### Confidence Scores")
                                fig = plot_probability_bars(probs)
                                st.pyplot(fig)
                                plt.close(fig)

                        # Waveform
                        st.markdown("#### Waveform")
                        fig_wave = plot_waveform(tmp_path)
                        st.pyplot(fig_wave)
                        plt.close(fig_wave)

                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                    finally:
                        os.unlink(tmp_path)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.85rem;">
            Built with ❤️ using Python, Librosa, Scikit-learn & Streamlit
            &nbsp;|&nbsp; RAVDESS Dataset
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
