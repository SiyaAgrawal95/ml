"""
src/generate_level_training_data.py
Builds training dataset for direct result level/value classification.
Uses reference chart colors (data/reference_colors.csv) and generates
small synthetic color variations to simulate minor lighting/camera differences.
Saves:
- data/synthetic_level_training_data.csv
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REF_CSV_PATH = DATA_DIR / "reference_colors.csv"
OUTPUT_CSV_PATH = DATA_DIR / "synthetic_level_training_data.csv"

# Target 6 clinical parameters
TARGET_PARAMETERS = [
    "Glucose",
    "Protein",
    "pH",
    "Ketone",
    "Blood",
    "Leukocytes"
]

FEATURE_COLS = ["R", "G", "B", "H", "S", "V", "L", "A", "B_lab"]


def generate_training_data(samples_per_level: int = 150, noise_std: float = 4.0, random_seed: int = 42):
    """
    Generates synthetic color variations around each reference chart level:
    - Adds small Gaussian noise to R, G, B
    - Computes exact HSV and LAB values
    - Assigns target_level, value, and unit
    """
    np.random.seed(random_seed)

    if not REF_CSV_PATH.exists():
        import extract_reference_colors
        extract_reference_colors.extract_reference_colors()

    ref_df = pd.read_csv(REF_CSV_PATH)
    ref_df = ref_df[ref_df["parameter"].isin(TARGET_PARAMETERS)].copy()

    records = []
    print("=" * 75)
    print("   Generating Synthetic Reference-Chart-Derived Training Data")
    print("=" * 75)
    print(f"Base reference levels: {len(ref_df)} across 6 parameters")
    print(f"Generating {samples_per_level} variations per level (noise std={noise_std})...\n")

    for _, row in ref_df.iterrows():
        param = row["parameter"]
        level = str(row["level"])
        val = str(row["value"])
        unit = str(row["unit"])
        base_r = float(row["R"])
        base_g = float(row["G"])
        base_b = float(row["B"])

        # 1. Include exact base reference color
        pixel_bgr = np.uint8([[[round(base_b), round(base_g), round(base_r)]]])
        hsv_base = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0, 0]
        lab_base = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)[0, 0]

        records.append({
            "parameter": param,
            "target_level": level,
            "value": val,
            "unit": unit,
            "R": int(round(base_r)),
            "G": int(round(base_g)),
            "B": int(round(base_b)),
            "H": int(hsv_base[0]),
            "S": int(hsv_base[1]),
            "V": int(hsv_base[2]),
            "L": int(lab_base[0]),
            "A": int(lab_base[1]),
            "B_lab": int(lab_base[2]),
            "is_synthetic": 0
        })

        # 2. Generate small variations
        for _ in range(samples_per_level - 1):
            # Minor Gaussian color perturbation (simulating camera exposure/lighting)
            dr = np.random.normal(0, noise_std)
            dg = np.random.normal(0, noise_std)
            db = np.random.normal(0, noise_std)

            r_syn = int(np.clip(round(base_r + dr), 0, 255))
            g_syn = int(np.clip(round(base_g + dg), 0, 255))
            b_syn = int(np.clip(round(base_b + db), 0, 255))

            syn_bgr = np.uint8([[[b_syn, g_syn, r_syn]]])
            syn_hsv = cv2.cvtColor(syn_bgr, cv2.COLOR_BGR2HSV)[0, 0]
            syn_lab = cv2.cvtColor(syn_bgr, cv2.COLOR_BGR2LAB)[0, 0]

            records.append({
                "parameter": param,
                "target_level": level,
                "value": val,
                "unit": unit,
                "R": r_syn,
                "G": g_syn,
                "B": b_syn,
                "H": int(syn_hsv[0]),
                "S": int(syn_hsv[1]),
                "V": int(syn_hsv[2]),
                "L": int(syn_lab[0]),
                "A": int(syn_lab[1]),
                "B_lab": int(syn_lab[2]),
                "is_synthetic": 1
            })

    df_out = pd.DataFrame(records)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Generated {len(df_out)} training samples saved to: {OUTPUT_CSV_PATH}")
    for p in TARGET_PARAMETERS:
        sub = df_out[df_out["parameter"] == p]
        print(f"  • {p:12s}: {len(sub)} samples across {sub['target_level'].nunique()} levels")
    print("=" * 75)
    return df_out


if __name__ == "__main__":
    generate_training_data()
