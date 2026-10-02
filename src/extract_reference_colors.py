"""
src/extract_reference_colors.py
Extracts exact reference color squares from the urinalysis chart image.
Computes RGB, HSV, and LAB values.
Saves to data/reference_colors.csv
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = PROJECT_ROOT / "data" / "reference_chart.png"
OUTPUT_CSV = PROJECT_ROOT / "data" / "reference_colors.csv"

# Definitions of the 6 target parameters and their exact levels/values from the chart
# (x, y, w, h) bounding box coordinates of each color square in data/reference_chart.png
CHART_DEFINITIONS = [
    # 1. Glucose (GLU, 30 sec, mg/dL)
    {"parameter": "Glucose", "level": "Negative", "value": "0", "unit": "mg/dL", "bbox": (182, 785, 50, 30)},
    {"parameter": "Glucose", "level": "1/10 (tr.)", "value": "100", "unit": "mg/dL", "bbox": (314, 784, 50, 32)},
    {"parameter": "Glucose", "level": "1/4 (+)", "value": "250", "unit": "mg/dL", "bbox": (381, 784, 50, 32)},
    {"parameter": "Glucose", "level": "1/2 (++)", "value": "500", "unit": "mg/dL", "bbox": (449, 784, 50, 32)},
    {"parameter": "Glucose", "level": "1 (+++)", "value": "1000", "unit": "mg/dL", "bbox": (516, 784, 50, 32)},
    {"parameter": "Glucose", "level": "2 or more (++++)", "value": "2000 or more", "unit": "mg/dL", "bbox": (583, 784, 51, 32)},

    # 2. Protein (PRO, 60 sec, mg/dL)
    {"parameter": "Protein", "level": "Negative", "value": "0", "unit": "mg/dL", "bbox": (182, 313, 50, 31)},
    {"parameter": "Protein", "level": "Trace", "value": "Trace", "unit": "mg/dL", "bbox": (247, 313, 50, 31)},
    {"parameter": "Protein", "level": "Small (+)", "value": "30", "unit": "mg/dL", "bbox": (384, 313, 50, 31)},
    {"parameter": "Protein", "level": "Moderate (++)", "value": "100", "unit": "mg/dL", "bbox": (450, 313, 50, 31)},
    {"parameter": "Protein", "level": "Large (+++)", "value": "300", "unit": "mg/dL", "bbox": (518, 313, 50, 31)},
    {"parameter": "Protein", "level": "Large (++++)", "value": "2000 or more", "unit": "mg/dL", "bbox": (585, 313, 50, 31)},

    # 3. pH (60 sec)
    {"parameter": "pH", "level": "5.0", "value": "5.0", "unit": "pH", "bbox": (182, 379, 50, 31)},
    {"parameter": "pH", "level": "6.0", "value": "6.0", "unit": "pH", "bbox": (247, 379, 50, 31)},
    {"parameter": "pH", "level": "6.5", "value": "6.5", "unit": "pH", "bbox": (316, 379, 50, 31)},
    {"parameter": "pH", "level": "7.0", "value": "7.0", "unit": "pH", "bbox": (383, 379, 50, 31)},
    {"parameter": "pH", "level": "7.5", "value": "7.5", "unit": "pH", "bbox": (450, 379, 50, 31)},
    {"parameter": "pH", "level": "8.0", "value": "8.0", "unit": "pH", "bbox": (518, 379, 50, 31)},
    {"parameter": "pH", "level": "8.5", "value": "8.5", "unit": "pH", "bbox": (584, 379, 51, 31)},

    # 4. Ketone (KET, 40 sec, mg/dL)
    {"parameter": "Ketone", "level": "Negative", "value": "0", "unit": "mg/dL", "bbox": (182, 614, 50, 31)},
    {"parameter": "Ketone", "level": "Trace", "value": "5", "unit": "mg/dL", "bbox": (315, 614, 50, 31)},
    {"parameter": "Ketone", "level": "Small (+)", "value": "15", "unit": "mg/dL", "bbox": (383, 614, 50, 32)},
    {"parameter": "Ketone", "level": "Moderate (++)", "value": "40", "unit": "mg/dL", "bbox": (449, 614, 51, 32)},
    {"parameter": "Ketone", "level": "Large (+++)", "value": "80", "unit": "mg/dL", "bbox": (517, 614, 50, 32)},
    {"parameter": "Ketone", "level": "Large (++++)", "value": "160", "unit": "mg/dL", "bbox": (584, 614, 51, 32)},

    # 5. Blood (BLO, 60 sec)
    {"parameter": "Blood", "level": "Negative", "value": "Negative", "unit": "cells/uL", "bbox": (182, 471, 50, 30)},
    {"parameter": "Blood", "level": "Non-Hemolyzed Trace", "value": "Trace (Non-Hemolyzed)", "unit": "cells/uL", "bbox": (247, 470, 50, 32)},
    {"parameter": "Blood", "level": "Non-Hemolyzed Moderate", "value": "Moderate (Non-Hemolyzed)", "unit": "cells/uL", "bbox": (315, 471, 50, 30)},
    {"parameter": "Blood", "level": "Hemolyzed Trace", "value": "Trace (Hemolyzed)", "unit": "cells/uL", "bbox": (383, 470, 50, 32)},
    {"parameter": "Blood", "level": "Small (+)", "value": "Small", "unit": "cells/uL", "bbox": (450, 470, 50, 32)},
    {"parameter": "Blood", "level": "Moderate (++)", "value": "Moderate", "unit": "cells/uL", "bbox": (518, 470, 50, 32)},
    {"parameter": "Blood", "level": "Large (+++)", "value": "Large", "unit": "cells/uL", "bbox": (584, 470, 51, 32)},

    # 6. Leukocytes (LEU, 2 min, Leu/uL)
    {"parameter": "Leukocytes", "level": "Negative", "value": "Negative", "unit": "Leu/uL", "bbox": (183, 87, 49, 30)},
    {"parameter": "Leukocytes", "level": "Trace", "value": "Trace", "unit": "Leu/uL", "bbox": (385, 87, 49, 30)},
    {"parameter": "Leukocytes", "level": "Small (+)", "value": "15", "unit": "Leu/uL", "bbox": (451, 86, 50, 31)},
    {"parameter": "Leukocytes", "level": "Moderate (++)", "value": "70", "unit": "Leu/uL", "bbox": (519, 86, 50, 31)},
    {"parameter": "Leukocytes", "level": "Large (+++)", "value": "125", "unit": "Leu/uL", "bbox": (586, 86, 50, 31)},
]


def extract_reference_colors():
    if not CHART_PATH.exists():
        raise FileNotFoundError(f"Chart image not found at {CHART_PATH}")

    img_bgr = cv2.imread(str(CHART_PATH))
    records = []

    for item in CHART_DEFINITIONS:
        x, y, w, h = item["bbox"]
        # Sample center region (inner 40% of the box to avoid borders/text)
        cx, cy = x + w // 2, y + h // 2
        half_w = max(2, int(w * 0.2))
        half_h = max(2, int(h * 0.2))
        patch_bgr = img_bgr[cy - half_h : cy + half_h + 1, cx - half_w : cx + half_w + 1]

        # Mean BGR
        mean_bgr = np.mean(patch_bgr, axis=(0, 1))
        b, g, r = mean_bgr[0], mean_bgr[1], mean_bgr[2]

        # Convert 1x1 pixel to HSV and LAB
        pixel_bgr = np.uint8([[[round(b), round(g), round(r)]]])
        pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0, 0]
        pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)[0, 0]

        record = {
            "parameter": item["parameter"],
            "level": item["level"],
            "value": item["value"],
            "unit": item["unit"],
            "R": int(round(r)),
            "G": int(round(g)),
            "B": int(round(b)),
            "H": int(pixel_hsv[0]),
            "S": int(pixel_hsv[1]),
            "V": int(pixel_hsv[2]),
            "L": int(pixel_lab[0]),
            "A": int(pixel_lab[1]),
            "B_lab": int(pixel_lab[2]),
        }
        records.append(record)

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully generated {OUTPUT_CSV} with {len(df)} reference color levels across 6 parameters.")
    return df


if __name__ == "__main__":
    df = extract_reference_colors()
    print("\nReference Colors Preview:")
    print(df[["parameter", "level", "value", "unit", "R", "G", "B", "L", "A", "B_lab"]].to_string(index=False))
