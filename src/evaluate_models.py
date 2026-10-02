"""
src/evaluate_models.py
Evaluates trained Multi-Label OneVsRest Random Forest and SVM models on Validation and Test sets.
Outputs:
- Comparison Table (Exact Match Accuracy, Hamming Loss, Precision, Recall, F1)
- Per-label performance summary
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score

from extract_features import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_artifacts():
    rf_model = joblib.load(MODELS_DIR / "random_forest_multilabel.pkl")
    svm_model = joblib.load(MODELS_DIR / "svm_model_multilabel.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    label_names = joblib.load(MODELS_DIR / "label_names.pkl")
    return rf_model, svm_model, scaler, label_names


def evaluate_split(split_name: str, csv_path: Path, rf_model, svm_model, scaler, label_names):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y_true = df[label_names].values

    # RF Predictions
    rf_preds = rf_model.predict(X)

    # SVM Predictions
    X_scaled = scaler.transform(X)
    svm_preds = svm_model.predict(X_scaled)

    # Compute metrics
    metrics = []
    for model_name, preds in [("Random Forest", rf_preds), ("SVM", svm_preds)]:
        acc = accuracy_score(y_true, preds)
        h_loss = hamming_loss(y_true, preds)
        prec_micro = precision_score(y_true, preds, average="micro", zero_division=0)
        rec_micro = recall_score(y_true, preds, average="micro", zero_division=0)
        f1_micro = f1_score(y_true, preds, average="micro", zero_division=0)
        prec_macro = precision_score(y_true, preds, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, preds, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)

        metrics.append({
            "Model": model_name,
            "Exact Match Acc": f"{acc * 100:.2f}%",
            "Hamming Loss": f"{h_loss:.4f}",
            "Precision (Micro)": f"{prec_micro:.4f}",
            "Recall (Micro)": f"{rec_micro:.4f}",
            "F1 (Micro)": f"{f1_micro:.4f}",
            "F1 (Macro)": f"{f1_macro:.4f}"
        })

    summary_df = pd.DataFrame(metrics)

    print("\n" + "=" * 80)
    print(f"      MULTI-LABEL EVALUATION ON [{split_name.upper()} SET] ({len(y_true)} samples)")
    print("=" * 80)
    print("\n--- Model Comparison Table ---")
    print(summary_df.to_string(index=False))

    # Per-label breakdown for Random Forest
    print(f"\n--- Per-Parameter Accuracy on [{split_name.upper()}] ---")
    per_param = []
    for i, lbl in enumerate(label_names):
        rf_col_acc = accuracy_score(y_true[:, i], rf_preds[:, i])
        svm_col_acc = accuracy_score(y_true[:, i], svm_preds[:, i])
        per_param.append({
            "Parameter": lbl,
            "True Positive Count": int(np.sum(y_true[:, i])),
            "RF Accuracy": f"{rf_col_acc * 100:.1f}%",
            "SVM Accuracy": f"{svm_col_acc * 100:.1f}%"
        })
    print(pd.DataFrame(per_param).to_string(index=False))
    return summary_df


def main():
    rf_model, svm_model, scaler, label_names = load_artifacts()

    # 1. Validation split
    evaluate_split(
        "Validation",
        DATA_DIR / "valid_features.csv",
        rf_model, svm_model, scaler, label_names
    )

    # 2. Test split
    evaluate_split(
        "Test",
        DATA_DIR / "test_features.csv",
        rf_model, svm_model, scaler, label_names
    )


if __name__ == "__main__":
    main()
