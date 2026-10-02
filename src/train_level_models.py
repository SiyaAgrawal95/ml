"""
src/train_level_models.py
Trains separate Random Forest and SVM classifiers for each of the 6 urine parameters:
- Glucose
- Protein
- pH
- Ketone
- Blood
- Leukocytes

Directly predicts chart result levels (e.g. Negative, 1/4 (+), Trace, 6.5, etc.).
Saves models to models/level_models/
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from generate_level_training_data import FEATURE_COLS, TARGET_PARAMETERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "level_models"
DATASET_PATH = DATA_DIR / "synthetic_level_training_data.csv"


def train_level_classifiers():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        import generate_level_training_data
        generate_level_training_data.generate_training_data()

    df = pd.read_csv(DATASET_PATH)

    print("=" * 80)
    print("   Training Direct Result Level Classifiers (Random Forest & SVM)")
    print("=" * 80)
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Target Parameters ({len(TARGET_PARAMETERS)}): {TARGET_PARAMETERS}\n")

    metadata = {}
    summary_results = []

    for param in TARGET_PARAMETERS:
        sub_df = df[df["parameter"] == param].copy()
        X = np.asarray(sub_df[FEATURE_COLS].values, dtype=np.float32)
        y = np.asarray(sub_df["target_level"].values, dtype=str)

        # Build mapping from level to value & unit
        level_map = {}
        for _, row in sub_df.drop_duplicates(subset=["target_level"]).iterrows():
            level_map[str(row["target_level"])] = {
                "value": str(row["value"]),
                "unit": str(row["unit"])
            }
        metadata[param] = level_map

        # Train / Test split (80/20 stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 1. Random Forest Classifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        rf_preds = rf_clf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds, average="weighted", zero_division=0)

        # 2. SVM with StandardScaler in a Pipeline
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, random_state=42))
        ])
        svm_pipe.fit(X_train, y_train)
        svm_preds = svm_pipe.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_preds)
        svm_f1 = f1_score(y_test, svm_preds, average="weighted", zero_division=0)

        summary_results.append({
            "Parameter": param,
            "Classes": len(level_map),
            "RF Test Acc": f"{rf_acc * 100:.1f}%",
            "RF F1": f"{rf_f1:.4f}",
            "SVM Test Acc": f"{svm_acc * 100:.1f}%",
            "SVM F1": f"{svm_f1:.4f}"
        })

        # Save model files
        joblib.dump(rf_clf, MODELS_DIR / f"rf_{param}.pkl")
        joblib.dump(svm_pipe, MODELS_DIR / f"svm_{param}.pkl")

    # Save metadata
    meta_path = MODELS_DIR / "level_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("--- Model Performance Summary on Evaluation Split ---")
    print(pd.DataFrame(summary_results).to_string(index=False))
    print(f"\nAll models and metadata saved to: {MODELS_DIR}/")
    print("=" * 80)
    return summary_results


if __name__ == "__main__":
    train_level_classifiers()
