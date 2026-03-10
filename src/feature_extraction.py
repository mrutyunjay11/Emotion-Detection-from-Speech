# =============================================================================
# Human Emotion Detection from Voice
# Feature Extraction Module
# =============================================================================
# Extracts audio features (MFCC, Chroma, Spectral Contrast) from audio files
# using the Librosa library.
# =============================================================================

import numpy as np
import librosa


def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCC).

    MFCCs capture the timbral/textural aspects of sound and are widely used
    in speech and emotion recognition tasks.

    Args:
        signal: Audio time-series signal.
        sr: Sampling rate of the audio.
        n_mfcc: Number of MFCCs to extract (default: 40).

    Returns:
        Mean MFCC values across time frames (shape: [n_mfcc]).
    """
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def extract_chroma(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract Chroma (pitch class) features.

    Chroma features represent the 12 different pitch classes and capture
    harmonic and melodic characteristics of the audio signal.

    Args:
        signal: Audio time-series signal.
        sr: Sampling rate of the audio.

    Returns:
        Mean Chroma values across time frames (shape: [12]).
    """
    # Compute Short-Time Fourier Transform
    stft = np.abs(librosa.stft(signal))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    return np.mean(chroma.T, axis=0)


def extract_spectral_contrast(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract Spectral Contrast features.

    Spectral contrast measures the difference in amplitude between peaks and
    valleys in the sound spectrum across several frequency sub-bands.

    Args:
        signal: Audio time-series signal.
        sr: Sampling rate of the audio.

    Returns:
        Mean Spectral Contrast values across time frames (shape: [7]).
    """
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    return np.mean(spectral_contrast.T, axis=0)


def extract_features(file_path: str, sr: int = 22050) -> np.ndarray:
    """
    Extract and combine all audio features into a single feature vector.

    Loads the audio file and extracts MFCC, Chroma, and Spectral Contrast
    features, then concatenates them into one feature vector.

    Feature vector composition:
        - MFCC:              40 values
        - Chroma:            12 values
        - Spectral Contrast:  7 values
        ──────────────────────────────
        Total:               59 values

    Args:
        file_path: Path to the audio file (.wav).
        sr: Target sampling rate (default: 22050 Hz).

    Returns:
        Combined feature vector (shape: [59]).

    Raises:
        Exception: If the audio file cannot be loaded or processed.
    """
    # Load audio file with a consistent sampling rate
    signal, sample_rate = librosa.load(file_path, sr=sr, duration=3)

    # Extract individual feature sets
    mfcc = extract_mfcc(signal, sample_rate)
    chroma = extract_chroma(signal, sample_rate)
    spectral_contrast = extract_spectral_contrast(signal, sample_rate)

    # Concatenate all features into a single vector
    combined_features = np.concatenate([mfcc, chroma, spectral_contrast])

    return combined_features
