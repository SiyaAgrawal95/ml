"""
src/extract_features.py
Extracts mean and standard deviation RGB and HSV features from each urine strip image.
Excludes non-chemical structural CV tags ('background', 'strip').
Saves:
- data/train_features.csv
- data/valid_features.csv
- data/test_features.csv
- data/color_features.csv (unified)
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"

TARGET_SIZE = (256, 256)

# 12 Extracted Color Features (RGB + HSV)
FEATURE_COLS = [
    "R_mean", "G_mean", "B_mean",
    "R_std", "G_std", "B_std",
    "H_mean", "S_mean", "V_mean",
    "H_std", "S_std", "V_std"
]

# Structural non-chemical tags to exclude
EXCLUDED_TAGS = {"background", "strip"}


def preprocess_image(img_bgr: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    """
    Minimal preprocessing:
    1. Resize to standardized target size (256x256).
    2. Convert OpenCV BGR to RGB.
    Preserves exact photometric color distribution for colorimetry.
    """
    if img_bgr is None:
        raise ValueError("Image input is None")
    resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return img_rgb


def extract_color_features(img_rgb: np.ndarray) -> dict:
    """
    Extracts mean and standard deviation of RGB and HSV channels:
    - RGB: R_mean, G_mean, B_mean, R_std, G_std, B_std
    - HSV: H_mean, S_mean, V_mean, H_std, S_std, V_std
    """
    # RGB channels
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    # HSV channels (OpenCV HSV: H in 0-179, S in 0-255, V in 0-255)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = img_hsv[:, :, 0].astype(np.float32)
    s = img_hsv[:, :, 1].astype(np.float32)
    v = img_hsv[:, :, 2].astype(np.float32)

    features = {
        "R_mean": float(np.mean(r)),
        "G_mean": float(np.mean(g)),
        "B_mean": float(np.mean(b)),
        "R_std": float(np.std(r)),
        "G_std": float(np.std(g)),
        "B_std": float(np.std(b)),
        "H_mean": float(np.mean(h)),
        "S_mean": float(np.mean(s)),
        "V_mean": float(np.mean(v)),
        "H_std": float(np.std(h)),
        "S_std": float(np.std(s)),
        "V_std": float(np.std(v)),
    }
    return features


def extract_split_features(split: str, dataset_dir: Path = DATASET_DIR):
    """
    Extracts color features for all images in a given split using its _classes.csv,
    excluding non-chemical structural tags (background, strip).
    """
    split_dir = dataset_dir / split
    classes_csv = split_dir / "_classes.csv"
    if not classes_csv.exists():
        raise FileNotFoundError(f"Classes CSV not found: {classes_csv}")

    df_meta = pd.read_csv(classes_csv)
    df_meta.columns = [c.strip() for c in df_meta.columns]

    filename_col = "filename" if "filename" in df_meta.columns else df_meta.columns[0]
    raw_label_cols = [c for c in df_meta.columns if c != filename_col]

    # Exclude structural tags, keep only genuine urine test parameters
    urine_param_cols = [c for c in raw_label_cols if c not in EXCLUDED_TAGS]

    records = []
    print(f"Extracting features for split [{split.upper()}] ({len(df_meta)} images)...")

    for _, row in df_meta.iterrows():
        fname = row[filename_col]
        img_path = split_dir / fname

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Warning: Could not read image at {img_path}")
            continue

        img_rgb = preprocess_image(img_bgr)
        feats = extract_color_features(img_rgb)

        record = {
            "filename": fname,
            "split": split,
            **feats,
            **{lbl: int(row[lbl]) for lbl in urine_param_cols}
        }
        records.append(record)

    df_feats = pd.DataFrame(records)
    print(f"  Extracted {len(df_feats)} records for [{split.upper()}]. Excluded structural tags: {EXCLUDED_TAGS}")
    return df_feats, urine_param_cols


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits = ["train", "valid", "test"]
    dfs = {}
    label_names = None

    print("=" * 75)
    print("   Starting Urine Parameter Color Feature Extraction")
    print("=" * 75)

    for split in splits:
        df_split, lbls = extract_split_features(split)
        dfs[split] = df_split
        label_names = lbls
        split_out_csv = DATA_DIR / f"{split}_features.csv"
        df_split.to_csv(split_out_csv, index=False)
        print(f"  Saved {split} features to: {split_out_csv}")

    unified_df = pd.concat([dfs["train"], dfs["valid"], dfs["test"]], ignore_index=True)
    unified_csv = DATA_DIR / "color_features.csv"
    unified_df.to_csv(unified_csv, index=False)

    print(f"\nUnified features saved to: {unified_csv} ({len(unified_df)} total records)")
    print(f"Color feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Actual Urine Test Parameters ({len(label_names)}): {label_names}")
    print("=" * 75)


if __name__ == "__main__":
    main()
