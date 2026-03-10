# 🎙️ Human Emotion Detection from Voice

A machine learning project that detects human emotions from voice/speech audio using audio feature extraction and classification models.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)

---

## 📋 Overview

This project uses audio processing and machine learning to classify emotions expressed in speech. It extracts acoustic features (MFCC, Chroma, Spectral Contrast) from audio recordings and trains SVM and Random Forest classifiers to predict one of five emotions:

| Emotion  | Emoji |
|----------|-------|
| Happy    | 😊    |
| Sad      | 😢    |
| Angry    | 😡    |
| Neutral  | 😐    |
| Fearful  | 😨    |

---

## 🗂️ Project Structure

```
voice-emotion-detection-ai/
├── data/                    # RAVDESS dataset (Actor_01/ ... Actor_24/)
├── models/                  # Saved trained models & scaler
│   ├── emotion_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Core source code
│   ├── __init__.py
│   ├── feature_extraction.py   # MFCC, Chroma, Spectral Contrast
│   ├── data_loader.py          # RAVDESS dataset loader
│   └── train_model.py          # Training & evaluation pipeline
├── app/                     # Streamlit web application
│   └── streamlit_app.py
├── outputs/                 # Evaluation outputs (plots, reports)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd voice-emotion-detection-ai
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the RAVDESS Dataset

1. Download from: [RAVDESS on Zenodo](https://zenodo.org/record/1188976)
2. Extract the **Audio_Speech_Actors_01-24** folder
3. Place the `Actor_01/` through `Actor_24/` folders inside the `data/` directory:

```
data/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-02-01.wav
│   └── ...
├── Actor_02/
│   └── ...
└── Actor_24/
    └── ...
```

### 5. Train the Model

```bash
python src/train_model.py
```

This will:
- Load and extract features from all RAVDESS audio files
- Train both SVM and Random Forest classifiers with hyperparameter tuning
- Evaluate and print accuracy, classification reports, and confusion matrices
- Save the best model to `models/`
- Save evaluation plots to `outputs/`

### 6. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser where you can:
- **Upload** an audio file (.wav, .mp3, .ogg, .flac)
- **Record** your voice using the microphone
- **Detect** the emotion with confidence scores

---

## 🔬 Features Extracted

| Feature             | Dimensions | Description                                    |
|---------------------|:----------:|------------------------------------------------|
| MFCC                | 40         | Mel-Frequency Cepstral Coefficients (timbre)   |
| Chroma              | 12         | Pitch class profiles (harmonic content)        |
| Spectral Contrast   | 7          | Peak-valley amplitude difference per sub-band  |
| **Total**           | **59**     | Combined feature vector                        |

---

## 🤖 Models

| Model          | Description                                  |
|----------------|----------------------------------------------|
| SVM            | Support Vector Machine with RBF/Linear kernel|
| Random Forest  | Ensemble of decision trees                   |

Both models are trained with **GridSearchCV** for hyperparameter tuning using 3-fold cross-validation.

---

## 📊 Evaluation Metrics

- **Accuracy** — Overall proportion of correct predictions
- **Confusion Matrix** — Per-class true vs. predicted counts
- **Classification Report** — Precision, recall, F1-score per class

Evaluation outputs are saved in the `outputs/` folder.

---

## 📝 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 24 professional actors (12 female, 12 male)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- We use 5 emotions: **happy, sad, angry, neutral, fearful**

**Citation:**  
> Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Librosa** — Audio feature extraction
- **Scikit-learn** — Machine learning models
- **NumPy / Pandas** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **Streamlit** — Interactive web application
- **Joblib** — Model serialization

---

## 📜 License

This project is for educational purposes. The RAVDESS dataset is licensed under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
