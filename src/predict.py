"""
src/predict.py
Unified prediction pipeline:
Directly predicts urinalysis result levels and values using trained Random Forest & SVM classifiers.
(No reference chart lookup performed at runtime).
"""

import argparse
from pathlib import Path
import sys
import cv2
import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_features import FEATURE_COLS, extract_color_features, preprocess_image
from predict_levels import predict_strip_direct_ml

MODELS_DIR = PROJECT_ROOT / "models"


def load_presence_model(model_name: str = "rf"):
    """
    Loads multi-label presence detection model for additional validation.
    """
    model_key = model_name.lower().strip().replace(" ", "_")
    if model_key in ["rf", "random_forest", "randomforest"]:
        model_file = MODELS_DIR / "random_forest_multilabel.pkl"
        scaler = None
        display_name = "Random Forest"
    else:
        model_file = MODELS_DIR / "svm_model_multilabel.pkl"
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        display_name = "SVM"

    model = joblib.load(model_file)
    label_names = joblib.load(MODELS_DIR / "label_names.pkl")
    return model, scaler, label_names, display_name


def predict_image(image_input, model_name: str = "rf"):
    """
    Full prediction function:
    1. Direct Level Prediction via trained RF/SVM classifiers:
       -> Parameter | Predicted Level | Value/Unit | Confidence
    2. Feature extraction (RGB, HSV).
    3. Multi-label presence probabilities.
    """
    # 1. Direct ML Level Prediction (Source of truth ML models)
    level_res = predict_strip_direct_ml(image_input, model_type=model_name)
    level_table = level_res["table"]

    # 2. Extract whole-strip color features
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
    else:
        img_bgr = image_input

    img_rgb = preprocess_image(img_bgr)
    features = extract_color_features(img_rgb)

    # 3. Multi-label presence model
    feat_vector = np.array([[features[col] for col in FEATURE_COLS]], dtype=np.float32)
    model, scaler, label_names, display_name = load_presence_model(model_name)
    if scaler is not None:
        feat_vector = scaler.transform(feat_vector)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat_vector)[0]
    else:
        probs = [float(p) for p in model.predict(feat_vector)[0]]

    ml_presence_table = []
    for label, prob in zip(label_names, probs):
        p_float = float(prob)
        ml_presence_table.append({
            "Parameter": label,
            "Presence Probability": f"{p_float * 100:.1f}%",
            "Detection": "Detected" if p_float >= 0.5 else "Not Detected"
        })

    return {
        "level_table": level_table,
        "ml_presence_table": ml_presence_table,
        "features": features,
        "model_used": level_res["model_used"],
        "pad_results": level_res["pad_results"],
        "strip_roi": level_res["strip_roi"]
    }


def main():
    parser = argparse.ArgumentParser(description="Predict urine strip parameter levels using direct ML models")
    parser.add_argument("--image", type=str, required=True, help="Path to input strip image")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm", "Random Forest", "SVM"],
                        help="Model to use ('rf' or 'svm')")
    args = parser.parse_args()

    results = predict_image(args.image, model_name=args.model)
    df_levels = pd.DataFrame(results["level_table"])[["Parameter", "Predicted Level", "Value/Unit", "Confidence"]]

    print("=" * 75)
    print(f"DIRECT ML LEVEL PREDICTION ({results['model_used']}): {Path(args.image).name}")
    print("=" * 75)
    print(df_levels.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
