"""
src/train_models.py
Trains Multi-Label OneVsRestClassifier with:
1. RandomForestClassifier
2. Support Vector Classifier (SVC with probability=True and StandardScaler)

Evaluates on validation split using:
- Precision (micro/macro/weighted)
- Recall (micro/macro/weighted)
- F1-score (micro/macro/weighted)
- Hamming Loss
- Exact Match Accuracy

Saves:
- models/random_forest_multilabel.pkl
- models/svm_model_multilabel.pkl
- models/scaler.pkl
- models/label_names.pkl
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract_features import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data():
    train_path = DATA_DIR / "train_features.csv"
    valid_path = DATA_DIR / "valid_features.csv"
    test_path = DATA_DIR / "test_features.csv"

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    non_label_cols = set(FEATURE_COLS + ["filename", "split"])
    label_cols = [c for c in train_df.columns if c not in non_label_cols]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[label_cols].values

    X_valid = valid_df[FEATURE_COLS].values
    y_valid = valid_df[label_cols].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[label_cols].values

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), label_cols


def compute_metrics(y_true, y_pred):
    return {
        "Exact Match Accuracy": accuracy_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Precision (Micro)": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Recall (Micro)": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "F1 (Micro)": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (X_train, y_train), (X_valid, y_valid), _, label_cols = load_data()

    print("=" * 75)
    print("   Training Multi-Label Classifiers (OneVsRest RF & SVM)")
    print("=" * 75)
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Labels ({len(label_cols)}): {label_cols}")
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_valid)}\n")

    # 1. Random Forest (OneVsRest)
    print("--- 1. Training OneVsRest(RandomForestClassifier) ---")
    rf_base = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf_ovr = OneVsRestClassifier(rf_base)
    rf_ovr.fit(X_train, y_train)

    rf_valid_preds = rf_ovr.predict(X_valid)
    rf_metrics = compute_metrics(y_valid, rf_valid_preds)

    print("Validation Results for Random Forest:")
    for k, v in rf_metrics.items():
        if "Accuracy" in k:
            print(f"  • {k:22s}: {v * 100:.2f}%")
        else:
            print(f"  • {k:22s}: {v:.4f}")
    print()

    # 2. Support Vector Machine (OneVsRest with StandardScaler)
    print("--- 2. Training OneVsRest(SVC(probability=True)) with StandardScaler ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    svm_base = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
    svm_ovr = OneVsRestClassifier(svm_base)
    svm_ovr.fit(X_train_scaled, y_train)

    svm_valid_preds = svm_ovr.predict(X_valid_scaled)
    svm_metrics = compute_metrics(y_valid, svm_valid_preds)

    print("Validation Results for SVM:")
    for k, v in svm_metrics.items():
        if "Accuracy" in k:
            print(f"  • {k:22s}: {v * 100:.2f}%")
        else:
            print(f"  • {k:22s}: {v:.4f}")
    print()

    # Save Models and Artifacts
    joblib.dump(rf_ovr, MODELS_DIR / "random_forest_multilabel.pkl")
    joblib.dump(rf_ovr, MODELS_DIR / "random_forest.pkl")  # compatibility alias
    joblib.dump(svm_ovr, MODELS_DIR / "svm_model_multilabel.pkl")
    joblib.dump(svm_ovr, MODELS_DIR / "svm_model.pkl")      # compatibility alias
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(label_cols, MODELS_DIR / "label_names.pkl")

    with open(MODELS_DIR / "label_names.json", "w") as f:
        json.dump(label_cols, f, indent=2)

    print(f"All models and label names saved to {MODELS_DIR}/:")
    print("  • random_forest_multilabel.pkl")
    print("  • svm_model_multilabel.pkl")
    print("  • scaler.pkl")
    print("  • label_names.pkl")
    print("  • label_names.json")
    print("=" * 75)


if __name__ == "__main__":
    train_models()
