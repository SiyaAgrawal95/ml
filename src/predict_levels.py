"""
src/predict_levels.py
Direct Result Level Prediction using trained Random Forest and SVM models.
Pipeline:
Uploaded strip image -> Detect 6 reagent pads -> Extract RGB/HSV/LAB features ->
Direct classification via trained RF / SVM models -> Predicted Level, Value/Unit, Confidence.
(No reference chart lookup performed at runtime).
"""

import argparse
import json
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

from color_matcher import TARGET_PARAMETERS, sample_reagent_pads
from generate_level_training_data import FEATURE_COLS

MODELS_DIR = PROJECT_ROOT / "models" / "level_models"
METADATA_PATH = MODELS_DIR / "level_metadata.json"


def load_level_models(model_type: str = "rf"):
    """
    Loads the 6 trained level classifiers for either 'rf' or 'svm'.
    """
    prefix = "rf" if model_type.lower() in ["rf", "random_forest"] else "svm"
    models = {}

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}. Run train_level_models.py first.")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    for param in TARGET_PARAMETERS:
        model_file = MODELS_DIR / f"{prefix}_{param}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}. Run train_level_models.py first.")
        models[param] = joblib.load(model_file)

    display_name = "Random Forest" if prefix == "rf" else "SVM"
    return models, metadata, display_name


def predict_strip_direct_ml(image_input, model_type: str = "rf"):
    """
    Direct ML prediction pipeline:
    1. Detects and extracts the 6 reagent pads from the input image.
    2. Feeds each pad's RGB, HSV, and LAB features directly into its RF/SVM classifier.
    3. Outputs the table: Parameter | Predicted Level | Value/Unit | Confidence.
    """
    # 1. Load image
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise ValueError(f"Could not load image at {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise TypeError("image_input must be a file path or numpy ndarray")

    # 2. Extract reagent pads (using geometric & photometric pad localization)
    pads, strip_roi = sample_reagent_pads(img_bgr)

    # 3. Load ML models and metadata
    models, metadata, display_name = load_level_models(model_type)

    table_rows = []
    pad_results = {}

    for param in TARGET_PARAMETERS:
        pad_data = pads[param]
        is_pad_valid = pad_data.get("valid", True)
        rgb = pad_data["RGB"]
        hsv = pad_data["HSV"]
        lab = pad_data["LAB"]

        # Check: Never predict from an empty/black crop
        if not is_pad_valid or (rgb[0] + rgb[1] + rgb[2]) / 3.0 < 25.0:
            pred_level = "Unable to detect pad"
            val_unit_formatted = "Unable to detect pad"
            conf_str = "N/A"
            confidence_val = None
            is_pad_valid = False
        else:
            # 9-dimensional color feature vector
            feat_dict = {
                "R": rgb[0],
                "G": rgb[1],
                "B": rgb[2],
                "H": hsv[0],
                "S": hsv[1],
                "V": hsv[2],
                "L": lab[0],
                "A": lab[1],
                "B_lab": lab[2]
            }
            feat_vector = np.array([[feat_dict[c] for c in FEATURE_COLS]], dtype=np.float32)

            # Direct ML classification
            model = models[param]
            pred_level = model.predict(feat_vector)[0]

            # Confidence via predict_proba
            confidence_val = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feat_vector)[0]
                confidence_val = float(np.max(probs))
                conf_str = f"{confidence_val * 100:.1f}%"
            else:
                conf_str = "N/A"

            # Format Value/Unit
            val_info = metadata[param].get(str(pred_level), {"value": str(pred_level), "unit": ""})
            val_str = val_info["value"]
            unit_str = val_info["unit"]

            if unit_str == "pH" or val_str.lower() in ["negative", "trace", "small", "moderate", "large"]:
                val_unit_formatted = val_str
            elif unit_str:
                val_unit_formatted = f"{val_str} {unit_str}"
            else:
                val_unit_formatted = val_str

        table_rows.append({
            "Parameter": param,
            "Predicted Level": str(pred_level),
            "Value/Unit": val_unit_formatted,
            "Confidence": conf_str,
            "_conf_float": confidence_val,
            "_valid": is_pad_valid
        })

        pad_results[param] = {
            "bbox": pad_data["bbox"],
            "RGB": rgb,
            "HSV": hsv,
            "LAB": [round(c, 1) for c in lab],
            "predicted_level": str(pred_level),
            "value_unit": val_unit_formatted,
            "confidence": conf_str,
            "valid": is_pad_valid,
            "patch": pad_data.get("patch", None)
        }

    return {
        "table": table_rows,
        "model_used": display_name,
        "pad_results": pad_results,
        "strip_roi": strip_roi
    }


def main():
    parser = argparse.ArgumentParser(description="Direct ML Urine Strip Level Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input strip image")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm", "Random Forest", "SVM"],
                        help="Model to use for level classification ('rf' or 'svm')")
    args = parser.parse_args()

    results = predict_strip_direct_ml(args.image, model_type=args.model)
    df = pd.DataFrame(results["table"])[["Parameter", "Predicted Level", "Value/Unit", "Confidence"]]

    print("=" * 75)
    print(f"Direct ML Level Prediction ({results['model_used']}): {Path(args.image).name}")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
