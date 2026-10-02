"""
src/color_matcher.py
Colorimetric result level predictor using reference chart color matching (Delta-E in CIELAB).
Identifies 6 target reagent pads:
- Glucose
- Protein
- pH
- Ketone
- Blood
- Leukocytes
Matches each pad against data/reference_colors.csv.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REF_CSV_PATH = DATA_DIR / "reference_colors.csv"

# Target 6 parameters in standard strip sequence (from Leu to Glu)
TARGET_PARAMETERS = [
    "Glucose",
    "Protein",
    "pH",
    "Ketone",
    "Blood",
    "Leukocytes"
]

# Standard 10-pad index mapping (0 = Leukocytes, 9 = Glucose)
PAD_INDEX_MAP = {
    "Leukocytes": 0,
    "Protein": 3,
    "pH": 4,
    "Blood": 5,
    "Ketone": 7,
    "Glucose": 9
}


def load_reference_colors():
    if not REF_CSV_PATH.exists():
        # Try generating if missing
        import extract_reference_colors
        return extract_reference_colors.extract_reference_colors()
    return pd.read_csv(REF_CSV_PATH)


def locate_strip_roi(img_bgr: np.ndarray):
    """
    Locates the test strip bounding box using multi-strategy analysis:
    1. Outer black/letterbox removal.
    2. Collinear reagent pad clustering (horizontal and vertical).
    3. Elongated strip body contour analysis (Otsu & adaptive thresholding).
    """
    h_orig, w_orig = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Clean letterboxing: find non-black content
    row_m = gray.mean(axis=1)
    col_m = gray.mean(axis=0)
    valid_rows = np.where(row_m > 20)[0]
    valid_cols = np.where(col_m > 20)[0]

    if len(valid_rows) < 30 or len(valid_cols) < 30:
        return (0, 0, w_orig, h_orig), False

    y1, y2 = valid_rows[0] + 3, valid_rows[-1] - 3
    x1, x2 = valid_cols[0] + 3, valid_cols[-1] - 3

    content = img_bgr[y1:y2, x1:x2]
    c_gray = gray[y1:y2, x1:x2]
    ch, cw = content.shape[:2]

    candidates = []

    # 2. Strategy A: Collinear square pad detection
    blur = cv2.GaussianBlur(c_gray, (3, 3), 0)
    edges = cv2.Canny(blur, 35, 110)
    th_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    comb = cv2.dilate(cv2.bitwise_or(edges, th_adapt), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    cnts, _ = cv2.findContours(comb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 8 <= bw <= 65 and 8 <= bh <= 65 and 0.4 <= (bw / max(1, bh)) <= 2.5:
            if bx <= 2 or by <= 2 or (bx + bw) >= cw - 2 or (by + bh) >= ch - 2:
                continue
            roi = content[by:by+bh, bx:bx+bw]
            m_val = roi.mean()
            if m_val > 50:
                squares.append((bx, by, bw, bh, bx + bw // 2, by + bh // 2, m_val))

    h_groups, v_groups = [], []
    for sq in squares:
        m_h = [g for g in h_groups if abs(np.mean([s[5] for s in g]) - sq[5]) <= 16]
        if m_h:
            m_h[0].append(sq)
        else:
            h_groups.append([sq])

        m_v = [g for g in v_groups if abs(np.mean([s[4] for s in g]) - sq[4]) <= 16]
        if m_v:
            m_v[0].append(sq)
        else:
            v_groups.append([sq])

    def deduplicate(g, idx):
        g.sort(key=lambda s: s[idx])
        out = []
        for s in g:
            if not out or abs(s[idx] - out[-1][idx]) > 12:
                out.append(s)
        return out

    for g in [deduplicate(grp, 4) for grp in h_groups]:
        if len(g) >= 4:
            min_x = max(0, min(s[0] for s in g) - 15)
            max_x = min(cw, max(s[0] + s[2] for s in g) + 15)
            mean_cy = int(np.mean([s[5] for s in g]))
            mean_h = int(np.mean([s[3] for s in g]))
            sh = max(24, int(mean_h * 1.6))
            cand_roi = content[max(0, mean_cy - sh // 2):min(ch, mean_cy + sh // 2), min_x:max_x]
            m_br = cand_roi.mean() if cand_roi.size > 0 else 0
            score = len(g) * 20.0 * (m_br / 255.0)
            candidates.append((x1 + min_x, y1 + max(0, mean_cy - sh // 2), max_x - min_x, sh, score, True))

    for g in [deduplicate(grp, 5) for grp in v_groups]:
        if len(g) >= 4:
            min_y = max(0, min(s[1] for s in g) - 15)
            max_y = min(ch, max(s[1] + s[3] for s in g) + 15)
            mean_cx = int(np.mean([s[4] for s in g]))
            mean_w = int(np.mean([s[2] for s in g]))
            sw = max(24, int(mean_w * 1.6))
            cand_roi = content[min_y:max_y, max(0, mean_cx - sw // 2):min(cw, mean_cx + sw // 2)]
            m_br = cand_roi.mean() if cand_roi.size > 0 else 0
            score = len(g) * 20.0 * (m_br / 255.0)
            candidates.append((x1 + max(0, mean_cx - sw // 2), y1 + min_y, sw, max_y - min_y, score, False))

    # 3. Strategy B: Elongated Strip Body Contours
    th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    for mask in [th_otsu, 255 - th_otsu]:
        cnts_b, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_b:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw >= cw * 0.96 and bh >= ch * 0.96:
                continue
            major, minor = max(bw, bh), min(bw, bh)
            if major >= 80 and minor >= 12:
                aspect = major / max(1, minor)
                if aspect >= 2.2:
                    if bx <= 2 or by <= 2 or (bx + bw) >= cw - 2 or (by + bh) >= ch - 2:
                        if minor < 18:
                            continue
                    roi = content[by:by+bh, bx:bx+bw]
                    mean_val = roi.mean()
                    if mean_val > 50:
                        score = (major / 10.0) * np.sqrt(aspect) * (mean_val / 255.0)
                        if 15 <= minor <= 75:
                            score *= 2.0
                        candidates.append((x1 + bx, y1 + by, bw, bh, score, bw >= bh))

    if candidates:
        candidates.sort(key=lambda x: x[4], reverse=True)
        best = candidates[0]
        return (int(best[0]), int(best[1]), int(best[2]), int(best[3])), bool(best[5])

    return (int(x1), int(y1), int(cw), int(ch)), bool(cw >= ch)


def sample_reagent_pads(img_bgr: np.ndarray):
    """
    Segments the strip, extracts the 6 target pads, validates non-empty color values,
    and ensures correct polarity (Leu -> Glu).
    """
    h_img, w_img = img_bgr.shape[:2]
    strip_roi, is_horizontal = locate_strip_roi(img_bgr)
    sx, sy, sw, sh = strip_roi

    sampled_10 = []
    for i in range(10):
        frac = (i + 0.5) / 10.0
        if is_horizontal:
            cx_local = int(frac * sw)
            cy_local = sh // 2
            pw = max(8, int(sw * 0.05))
            ph = max(8, int(sh * 0.50))
        else:
            cx_local = sw // 2
            cy_local = int(frac * sh)
            pw = max(8, int(sw * 0.50))
            ph = max(8, int(sh * 0.05))

        cx_abs = sx + cx_local
        cy_abs = sy + cy_local
        x1 = max(0, cx_abs - pw // 2)
        x2 = min(w_img, cx_abs + pw // 2)
        y1 = max(0, cy_abs - ph // 2)
        y2 = min(h_img, cy_abs + ph // 2)

        patch = img_bgr[y1:y2, x1:x2]
        is_valid = True
        err_msg = ""

        if patch.size == 0 or (x2 <= x1) or (y2 <= y1):
            is_valid = False
            err_msg = "Crop out of bounds"
            patch = np.zeros((8, 8, 3), dtype=np.uint8)
            b, g, r = 0, 0, 0
            px_hsv = [0, 0, 0]
            px_lab = [0.0, 0.0, 0.0]
        else:
            mean_bgr = np.mean(patch, axis=(0, 1))
            b, g, r = mean_bgr[0], mean_bgr[1], mean_bgr[2]

            # Validation: Never predict from empty / black crop
            if (r + g + b) / 3.0 < 25.0:
                is_valid = False
                err_msg = "Crop too dark or empty (mean RGB < 25)"

            px_bgr = np.uint8([[[round(b), round(g), round(r)]]])
            px_hsv = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV)[0, 0]
            px_lab = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2LAB)[0, 0]

        sampled_10.append({
            "index": i,
            "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            "center": (int(cx_abs), int(cy_abs)),
            "RGB": [int(round(r)), int(round(g)), int(round(b))],
            "HSV": [int(px_hsv[0]), int(px_hsv[1]), int(px_hsv[2])],
            "LAB": [float(px_lab[0]), float(px_lab[1]), float(px_lab[2])],
            "patch": patch,
            "valid": is_valid,
            "error": err_msg
        })

    # Polarity check: Is pad 0 Leukocytes and pad 9 Glucose, or vice versa?
    ref_df = load_reference_colors()
    leu_rows = ref_df[(ref_df["parameter"] == "Leukocytes") & (ref_df["level"] == "Negative")]
    glu_rows = ref_df[(ref_df["parameter"] == "Glucose") & (ref_df["level"] == "Negative")]

    if len(leu_rows) > 0 and len(glu_rows) > 0:
        leu_ref = leu_rows.iloc[0]
        glu_ref = glu_rows.iloc[0]

        lab_0 = sampled_10[0]["LAB"]
        lab_9 = sampled_10[9]["LAB"]

        dist_a = (
            np.sqrt((lab_0[0] - float(leu_ref["L"]))**2 + (lab_0[1] - float(leu_ref["A"]))**2 + (lab_0[2] - float(leu_ref["B_lab"]))**2) +
            np.sqrt((lab_9[0] - float(glu_ref["L"]))**2 + (lab_9[1] - float(glu_ref["A"]))**2 + (lab_9[2] - float(glu_ref["B_lab"]))**2)
        )

        dist_b = (
            np.sqrt((lab_0[0] - float(glu_ref["L"]))**2 + (lab_0[1] - float(glu_ref["A"]))**2 + (lab_0[2] - float(glu_ref["B_lab"]))**2) +
            np.sqrt((lab_9[0] - float(leu_ref["L"]))**2 + (lab_9[1] - float(leu_ref["A"]))**2 + (lab_9[2] - float(leu_ref["B_lab"]))**2)
        )

        if dist_b < dist_a:
            sampled_10.reverse()

    # Extract target 6 parameters
    target_pads = {}
    for param_name in TARGET_PARAMETERS:
        idx = PAD_INDEX_MAP[param_name]
        target_pads[param_name] = sampled_10[idx]

    return target_pads, strip_roi


def match_color_to_reference(pad_lab, parameter: str, ref_df=None):
    """
    Compares pad's CIELAB color to all reference levels for this parameter.
    Selects closest level based on Euclidean LAB distance (Delta-E).
    """
    if ref_df is None:
        ref_df = load_reference_colors()

    param_refs = ref_df[ref_df["parameter"] == parameter]
    best_dist = float("inf")
    best_row = None

    pl, pa, pb = float(pad_lab[0]), float(pad_lab[1]), float(pad_lab[2])

    for _, row in param_refs.iterrows():
        rl, ra, rb = float(row["L"]), float(row["A"]), float(row["B_lab"])
        dist = np.sqrt((pl - rl)**2 + (pa - ra)**2 + (pb - rb)**2)
        if dist < best_dist:
            best_dist = dist
            best_row = row

    unit_str = str(best_row["unit"])
    val_str = str(best_row["value"])
    if unit_str == "pH" or val_str.lower() in ["negative", "trace", "small", "moderate", "large"]:
        val_unit = val_str
    else:
        val_unit = f"{val_str} {unit_str}"

    return {
        "Parameter": parameter,
        "Predicted Level": best_row["level"],
        "Value/Unit": val_unit,
        "Color Distance": round(best_dist, 2),
        "Ref_RGB": [int(best_row["R"]), int(best_row["G"]), int(best_row["B"])]
    }


def predict_strip_levels(image_input):
    """
    Analyzes an uploaded urine test strip image and predicts levels for the 6 target parameters.

    Returns:
    dict containing:
      - 'table': list of dicts [{'Parameter': ..., 'Predicted Level': ..., 'Value/Unit': ..., 'Color Distance': ...}]
      - 'pad_details': dictionary of pad metrics and coordinates
      - 'annotated_image': OpenCV BGR image with drawn boxes and labels
    """
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            raise ValueError(f"Could not load image at {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise TypeError("image_input must be path or numpy array")

    ref_df = load_reference_colors()
    pads, strip_roi = sample_reagent_pads(img_bgr)

    annotated = img_bgr.copy()
    # Draw strip ROI in cyan
    sx, sy, sw, sh = strip_roi
    cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)

    results_table = []
    pad_details = {}

    for param in TARGET_PARAMETERS:
        pad_info = pads[param]
        match = match_color_to_reference(pad_info["LAB"], param, ref_df)

        results_table.append({
            "Parameter": param,
            "Predicted Level": match["Predicted Level"],
            "Value/Unit": match["Value/Unit"],
            "Color Distance": match["Color Distance"]
        })

        pad_details[param] = {
            "bbox": pad_info["bbox"],
            "RGB": pad_info["RGB"],
            "HSV": pad_info["HSV"],
            "LAB": [round(c, 1) for c in pad_info["LAB"]],
            "Ref_RGB": match["Ref_RGB"],
            "match": match
        }

        # Draw pad bounding box in green and label
        bx, by, bw, bh = pad_info["bbox"]
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            param[:4],
            (bx, max(12, by - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return {
        "table": results_table,
        "pad_details": pad_details,
        "annotated_image": annotated,
        "strip_roi": strip_roi
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict urine strip parameter levels using reference chart")
    parser.add_argument("--image", type=str, required=True, help="Path to strip image")
    args = parser.parse_args()

    res = predict_strip_levels(args.image)
    df = pd.DataFrame(res["table"])

    print("=" * 70)
    print(f"Urine Strip Level Analysis: {Path(args.image).name}")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
