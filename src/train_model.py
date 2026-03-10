# =============================================================================
# Human Emotion Detection from Voice
# Model Training & Evaluation Module
# =============================================================================
# Trains SVM and Random Forest classifiers on extracted audio features,
# evaluates them, and saves the best model to disk.
# =============================================================================

import os
import sys
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Ensure import works when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_ravdess_dataset


# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str,
):
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: List of class label names.
        title: Plot title.
        save_path: File path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved → {save_path}")


def plot_model_comparison(results: dict, save_path: str):
    """
    Plot a bar chart comparing model accuracies.

    Args:
        results: Dict mapping model name → accuracy score.
        save_path: File path to save the figure.
    """
    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, scores, color=["#4A90D9", "#50C878"], edgecolor="black")
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison — Accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Comparison chart saved → {save_path}")


# ── Training functions ────────────────────────────────────────────────────────

def train_svm(X_train, y_train) -> SVC:
    """
    Train a Support Vector Machine with hyperparameter tuning.

    Uses GridSearchCV over a small parameter grid to find a good
    (C, gamma, kernel) combination.
    """
    print("\n🔧 Training SVM (with GridSearchCV)...")
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"],
    }
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with hyperparameter tuning.
    """
    print("\n🔧 Training Random Forest (with GridSearchCV)...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> float:
    """
    Evaluate a trained model and print/save metrics.

    Returns the test accuracy.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Test Accuracy: {acc:.4f}")
    print(f"{'='*60}")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # Save classification report to file
    report_path = os.path.join(OUTPUTS_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{model_name} — Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"  📄 Report saved → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(OUTPUTS_DIR, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, f"{model_name} — Confusion Matrix", cm_path)

    return acc


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    """
    End-to-end training pipeline:
      1. Load & extract features from RAVDESS dataset
      2. Split data (80/20)
      3. Scale features
      4. Train SVM & Random Forest
      5. Evaluate both models
      6. Save the best model + scaler
    """
    ensure_dirs()

    # ── 1. Load dataset ───────────────────────────────────────────────────
    print("=" * 60)
    print("  HUMAN EMOTION DETECTION FROM VOICE")
    print("  Training Pipeline")
    print("=" * 60)

    X, y, class_names = load_ravdess_dataset(DATA_DIR)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {class_names}")

    # ── 2. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]}  |  Test set: {X_test.shape[0]}")

    # ── 3. Feature scaling ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── 4. Train models ──────────────────────────────────────────────────
    svm_model = train_svm(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # ── 5. Evaluate models ───────────────────────────────────────────────
    results = {}
    results["SVM"] = evaluate_model(svm_model, X_test, y_test, class_names, "SVM")
    results["RandomForest"] = evaluate_model(
        rf_model, X_test, y_test, class_names, "RandomForest"
    )

    # Comparison chart
    plot_model_comparison(results, os.path.join(OUTPUTS_DIR, "model_comparison.png"))

    # ── 6. Save the best model ───────────────────────────────────────────
    best_name = max(results, key=results.get)
    best_model = svm_model if best_name == "SVM" else rf_model

    model_path = os.path.join(MODELS_DIR, "emotion_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save metadata so the Streamlit app knows which model was chosen
    metadata = {
        "best_model": best_name,
        "accuracy": results[best_name],
        "classes": class_names,
        "feature_dim": X.shape[1],
        "all_results": results,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Best model ({best_name}, acc={results[best_name]:.4f}) saved:")
    print(f"   Model  → {model_path}")
    print(f"   Scaler → {scaler_path}")
    print(f"   Meta   → {meta_path}")
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
