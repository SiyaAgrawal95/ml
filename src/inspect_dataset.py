"""
src/inspect_dataset.py
Inspect the Roboflow Multi-Label Dataset.
Distinguishes between actual urine chemical reagent parameters and structural CV tags (background, strip).
"""

from pathlib import Path
import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# Structural labels to exclude from urinalysis
STRUCTURAL_TAGS = {"background", "strip"}


def inspect_dataset(dataset_dir: Path = DATASET_DIR):
    print("=" * 75)
    print("   Urine Test Strip Multi-Label Dataset Inspection")
    print("=" * 75)
    print(f"Dataset root: {dataset_dir}\n")

    splits = ["train", "valid", "test"]
    split_info = {}

    for split in splits:
        split_path = dataset_dir / split
        csv_file = split_path / "_classes.csv"

        print(f"--- Split: [{split.upper()}] ---")
        if not csv_file.exists():
            print(f"  Error: {csv_file} does not exist.")
            continue

        df = pd.read_csv(csv_file)
        df.columns = [c.strip() for c in df.columns]

        filename_col = "filename" if "filename" in df.columns else df.columns[0]
        raw_label_cols = [c for c in df.columns if c != filename_col]

        # Distinguish clinical urine parameters from structural labels
        urine_params = [c for c in raw_label_cols if c not in STRUCTURAL_TAGS]
        structural_labels = [c for c in raw_label_cols if c in STRUCTURAL_TAGS]

        total_images = len(df)
        print(f"  Total images annotated: {total_images}")
        print(f"  Structural CV tags (excluded) : {structural_labels}")
        print(f"  Actual urine parameters ({len(urine_params)}): {urine_params}")

        # Check sample image resolution
        if total_images > 0:
            sample_img_name = df.iloc[0][filename_col]
            sample_img_path = split_path / sample_img_name
            if sample_img_path.exists():
                img = cv2.imread(str(sample_img_path))
                if img is not None:
                    h, w, c = img.shape
                    print(f"  Sample image resolution ({sample_img_name}): {w}x{h} ({c} channels)")

        # Frequency breakdown for clinical parameters
        print("  Urine Parameter Positive Frequencies (Detected / 1):")
        pos_counts = df[urine_params].sum()
        for label, count in pos_counts.items():
            pct = (count / total_images) * 100 if total_images > 0 else 0
            print(f"    • {label:15s}: {int(count):4d} / {total_images} ({pct:5.1f}%)")
        print()

        split_info[split] = {
            "total_images": total_images,
            "urine_params": urine_params,
            "structural_labels": structural_labels,
            "pos_counts": pos_counts.to_dict()
        }

    print("=" * 75)
    print("Inspection complete.")
    print("=" * 75)
    return split_info


if __name__ == "__main__":
    inspect_dataset()
