# =============================================================================
# Human Emotion Detection from Voice
# Data Loader Module
# =============================================================================
# Loads the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and
# Song) dataset and maps file names to emotion labels.
#
# RAVDESS filename convention:
#   Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
#   e.g. 03-01-05-01-01-01-12.wav
#
# Emotion codes in RAVDESS:
#   01 = neutral, 02 = calm, 03 = happy, 04 = sad,
#   05 = angry,   06 = fearful, 07 = disgust, 08 = surprised
# =============================================================================

import os
import glob
import numpy as np
from tqdm import tqdm

# Import the feature extraction module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_extraction import extract_features


# ---------------------------------------------------------------------------
# RAVDESS emotion code → label mapping
# We only keep the 5 target emotions; others are skipped.
# ---------------------------------------------------------------------------
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
}

# The emotions we want to classify
TARGET_EMOTIONS = ["happy", "sad", "angry", "neutral", "fearful"]


def get_emotion_from_filename(filename: str) -> str | None:
    """
    Parse the RAVDESS filename and return the emotion label.

    Args:
        filename: Name of the audio file (e.g. '03-01-05-01-01-01-12.wav').

    Returns:
        Emotion label string, or None if the emotion is not in our target set.
    """
    # The emotion code is the 3rd component (index 2) of the filename
    parts = os.path.basename(filename).split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return RAVDESS_EMOTION_MAP.get(emotion_code, None)


def load_ravdess_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the RAVDESS dataset from disk.

    Expects the data directory to contain actor sub-folders:
        data/Actor_01/, data/Actor_02/, …, data/Actor_24/

    Each sub-folder contains .wav files following the RAVDESS naming convention.

    Args:
        data_dir: Path to the root data directory containing Actor_* folders.

    Returns:
        features:  NumPy array of shape (N, 59) — extracted feature vectors.
        labels:    NumPy array of shape (N,) — integer-encoded emotion labels.
        label_names: List of emotion class names in label-index order.
    """
    features = []
    labels = []
    skipped = 0

    # Locate all .wav files inside Actor_* sub-folders
    audio_files = sorted(glob.glob(os.path.join(data_dir, "Actor_*", "*.wav")))

    if not audio_files:
        # Fallback: try flat directory of .wav files
        audio_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))

    if not audio_files:
        raise FileNotFoundError(
            f"No .wav files found in '{data_dir}'. "
            "Make sure the RAVDESS dataset is placed inside the 'data/' folder "
            "with sub-folders like Actor_01/, Actor_02/, etc."
        )

    print(f"Found {len(audio_files)} audio files. Extracting features...")

    for filepath in tqdm(audio_files, desc="Processing audio files"):
        emotion = get_emotion_from_filename(filepath)

        # Skip emotions not in our target set (e.g. calm, disgust, surprised)
        if emotion is None:
            skipped += 1
            continue

        try:
            feature_vector = extract_features(filepath)
            features.append(feature_vector)
            labels.append(TARGET_EMOTIONS.index(emotion))
        except Exception as e:
            print(f"  ⚠ Skipping {filepath}: {e}")
            skipped += 1

    print(f"✔ Extracted features from {len(features)} files ({skipped} skipped)")

    features = np.array(features)
    labels = np.array(labels)

    return features, labels, TARGET_EMOTIONS


if __name__ == "__main__":
    # Quick test: load dataset from 'data/' folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data")

    X, y, class_names = load_ravdess_dataset(data_path)
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_names}")
    print(f"Samples per class: {dict(zip(class_names, np.bincount(y)))}")
