"""
app.py
Streamlit Web Application:
“Machine Learning-Based Urine Test Strip Analysis Using Colorimetry”

Redesigned Frontend:
1. Top Summary: Six clean result cards with Parameter, Value, Category Badge, and Confidence.
2. Result Table: Parameter | Value | Level | Confidence.
3. Analysis Graph: Normalized Level Score (0–100%) for ordinal parameters + separate pH scale gauge.
4. Simple Factual Result Summary: Plain-language bullet points without clinical or diagnostic claims.
5. Expandable Technical Details (collapsed by default): Pad crops, RGB/HSV/LAB, ML model details, synthetic training data notice.
6. Educational Prototype Disclaimer.
"""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from color_matcher import TARGET_PARAMETERS
from predict import predict_image

# Page Configuration
st.set_page_config(
    page_title="Urine Test Strip Analysis | Colorimetry & ML",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Healthcare-Grade Cards & Badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FFFDF5;
        border-left: 4px solid #D69E2E;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #744210;
    }
    .card-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 16px;
        transition: transform 0.15s ease-in-out;
    }
    .card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .card-param {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1A202C;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }
    .badge-negative { background-color: #EDF2F7; color: #4A5568; border: 1px solid #CBD5E0; }
    .badge-trace    { background-color: #FEFCBF; color: #744210; border: 1px solid #ECC94B; }
    .badge-small    { background-color: #FEEBC8; color: #7B341E; border: 1px solid #ED8936; }
    .badge-moderate { background-color: #FED7D7; color: #742A2A; border: 1px solid #E53E3E; }
    .badge-large    { background-color: #FED7E2; color: #702459; border: 1px solid #D53F8C; }
    .badge-ph       { background-color: #EBF8FF; color: #2B6CB0; border: 1px solid #63B3ED; }
    .badge-failed   { background-color: #FFF5F5; color: #C53030; border: 1px solid #FEB2B2; }
    .card-conf {
        font-size: 0.82rem;
        color: #718096;
    }
    .summary-box {
        background-color: #F7FAFC;
        border-left: 4px solid #3182CE;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .summary-item {
        font-size: 0.95rem;
        color: #2D3748;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown("<div class='main-title'>🧪 Urine Test Strip Analysis Using Colorimetry</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>College Mini-Project | Direct Result Level Prediction via Classical ML (Random Forest & SVM)</div>", unsafe_allow_html=True)

# Mandatory Educational Prototype Notice
st.markdown(
    """
    <div class='disclaimer-box'>
        <strong>⚠️ Educational Prototype Notice:</strong>
        This is an educational mini-project prototype demonstrating colorimetry and classical machine learning (Random Forest / SVM).
        It is not a certified medical device and should not be used for medical diagnosis, clinical decision-making, or self-treatment.
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar: Configuration & Controls
with st.sidebar:
    st.header("⚙️ Settings & Input")
    model_choice = st.radio(
        "Choose ML Classifier:",
        options=["Random Forest", "SVM"],
        index=0,
        help="Random Forest: Decision tree ensemble trained on reference color levels\nSVM: RBF kernel with StandardScaler"
    )

    st.markdown("---")
    st.subheader("Image Input")
    input_source = st.radio(
        "Select source:",
        options=["Upload My Strip Image", "Use Dataset Test Sample"],
        index=1
    )

    uploaded_file = None
    sample_path = None

    if input_source == "Upload My Strip Image":
        uploaded_file = st.file_uploader(
            "Upload strip photo (JPG/PNG):",
            type=["jpg", "jpeg", "png"]
        )
    else:
        test_dir = PROJECT_ROOT / "dataset" / "test"
        sample_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*/*.jpg")))
        if sample_files:
            sample_path = st.selectbox(
                "Pick a test image:",
                options=sample_files,
                format_func=lambda p: p.name[:35] + ("..." if len(p.name) > 35 else "")
            )
        else:
            st.warning("No sample images found in dataset/test/")

    st.markdown("---")
    st.markdown(
        "<small style='color: gray;'>"
        "<strong>Pipeline Overview:</strong><br>"
        "1. Image Ingestion<br>"
        "2. Strip & Pad Localization<br>"
        "3. RGB, HSV, CIELAB Extraction<br>"
        "4. Direct ML Classification (RF/SVM)<br>"
        "5. Normalized Visualizations"
        "</small>",
        unsafe_allow_html=True
    )


def categorize_level(param: str, level_str: str, val_str: str):
    """
    Maps predicted level string to standard ordinal category, badge CSS class, and numerical score.
    Ordinal Categories: Negative (0), Trace (1), Small (2), Moderate (3), Large (4).
    """
    l_lower = level_str.lower()
    if "unable" in l_lower:
        return "Unable to detect pad", "badge-failed", 0, False

    if param == "pH":
        try:
            val = float(val_str.split()[0])
        except Exception:
            try:
                val = float(level_str)
            except Exception:
                val = 7.0
        return f"pH {val}", "badge-ph", val, True

    if "negative" in l_lower or val_str.lower() == "0 mg/dl" or val_str.lower() == "negative":
        return "Negative", "badge-negative", 0, True
    elif "trace" in l_lower or "1/10" in l_lower or "tr." in l_lower:
        return "Trace", "badge-trace", 1, True
    elif "small" in l_lower or "1/4" in l_lower or "30" in l_lower or ("(+)" in l_lower and "(++)" not in l_lower):
        return "Small", "badge-small", 2, True
    elif "moderate" in l_lower or "1/2" in l_lower or "100" in l_lower or ("(++)" in l_lower and "(+++)" not in l_lower):
        return "Moderate", "badge-moderate", 3, True
    elif "large" in l_lower or "1 " in l_lower or "2 " in l_lower or "300" in l_lower or "(+++)" in l_lower or "(++++)" in l_lower or "2 or more" in l_lower:
        return "Large", "badge-large", 4, True

    return "Detected", "badge-small", 2, True


# Load Selected Image
img_bgr = None
if input_source == "Upload My Strip Image" and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif input_source == "Use Dataset Test Sample" and sample_path is not None:
    img_bgr = cv2.imread(str(sample_path))

# Main UI Area
if img_bgr is None:
    st.info("👈 Please select or upload a urine test strip image from the sidebar to begin.")
else:
    # Top Section: Image Preview & Trigger
    preview_col1, preview_col2 = st.columns([1, 2])

    with preview_col1:
        st.subheader("📸 Strip Preview")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Input Test Strip", use_container_width=True)

    with preview_col2:
        st.subheader("⚡ Ready for Analysis")
        st.write(
            f"**Selected ML Model:** `{model_choice}`<br>"
            f"**Resolution:** `{img_bgr.shape[1]} × {img_bgr.shape[0]} px`<br>"
            f"**Target Parameters (6):** `Glucose, Protein, pH, Ketone, Blood, Leukocytes`",
            unsafe_allow_html=True
        )
        analyze_clicked = st.button("🔬 Analyze Strip Now", type="primary", use_container_width=True)

    if analyze_clicked:
        with st.spinner(f"Segmenting reagent pads and classifying result levels using {model_choice}..."):
            results = predict_image(img_bgr, model_name=model_choice)

        st.markdown("---")

        # -------------------------------------------------------------
        # 1. TOP SUMMARY: Six Clean Result Cards
        # -------------------------------------------------------------
        st.subheader("1. 📊 Result Summary Cards")
        st.caption(f"Direct classification using **{results['model_used']}** without runtime reference chart lookup.")

        cards_row1 = st.columns(3)
        cards_row2 = st.columns(3)
        all_cols = cards_row1 + cards_row2

        category_data = {}

        for idx, row in enumerate(results["level_table"]):
            param = row["Parameter"]
            pred_level = row["Predicted Level"]
            val_unit = row["Value/Unit"]
            conf = row["Confidence"]
            col = all_cols[idx]

            category, badge_class, score, is_valid = categorize_level(param, pred_level, val_unit)
            category_data[param] = {
                "category": category,
                "badge_class": badge_class,
                "score": score,
                "val_unit": val_unit,
                "level": pred_level,
                "conf": conf,
                "valid": is_valid
            }

            with col:
                st.markdown(
                    f"""
                    <div class='card-container'>
                        <div class='card-param'>{param}</div>
                        <div class='card-value'>{val_unit}</div>
                        <div><span class='badge {badge_class}'>Level: {category}</span></div>
                        <div class='card-conf'>Confidence: <strong>{conf}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 2. RESULT TABLE: Parameter | Value | Level | Confidence
        # -------------------------------------------------------------
        st.subheader("2. 📋 Urinalysis Result Table")

        table_rows = []
        for row in results["level_table"]:
            param = row["Parameter"]
            info = category_data[param]
            table_rows.append({
                "Parameter": param,
                "Value": info["val_unit"],
                "Level": info["level"],
                "Confidence": info["conf"]
            })

        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 3. ANALYSIS GRAPH: Normalized Level Score (0–100%) & pH Scale
        # -------------------------------------------------------------
        st.subheader("3. 📈 Parameter Level Analysis Overview")
        st.caption(
            "Ordinal categories (Negative=0, Trace=1, Small=2, Moderate=3, Large=4) normalized to 0–100% "
            "for standardized comparison across differing physical units. pH is displayed separately on its reference scale."
        )

        g_col1, g_col2 = st.columns([3, 2])

        # Subplot 1: Ordinal 5 parameters
        ordinal_params = ["Glucose", "Protein", "Ketone", "Blood", "Leukocytes"]
        norm_scores = []
        bar_colors = []
        color_palette = {
            "Negative": "#A0AEC0",
            "Trace": "#D69E2E",
            "Small": "#DD6B20",
            "Moderate": "#E53E3E",
            "Large": "#9B2C2C"
        }

        for p in ordinal_params:
            d = category_data.get(p, {"score": 0, "category": "Negative"})
            score = d["score"]
            norm_pct = (score / 4.0) * 100.0
            norm_scores.append(norm_pct)
            cat = d["category"]
            bar_colors.append(color_palette.get(cat, "#3182CE"))

        with g_col1:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            y_pos = np.arange(len(ordinal_params))
            bars = ax.barh(y_pos, norm_scores, color=bar_colors, height=0.55, edgecolor="#2D3748", linewidth=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(ordinal_params, fontsize=10, fontweight="bold")
            ax.set_xlim(0, 105)
            ax.set_xlabel("Normalized Level Score (%)", fontsize=10)
            ax.set_title("Ordinal Parameters Normalized Scale (0–100%)", fontsize=11, fontweight="bold", pad=12)

            # Reference ticks along 0-100
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["Neg\n(0%)", "Trace\n(25%)", "Small\n(50%)", "Mod\n(75%)", "Large\n(100%)"], fontsize=8.5)
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Value labels
            for bar, p in zip(bars, ordinal_params):
                w = bar.get_width()
                cat = category_data[p]["category"]
                ax.text(w + 2, bar.get_y() + bar.get_height() / 2, f"{cat} ({int(w)}%)", va="center", fontsize=8.5, fontweight="600")

            plt.tight_layout()
            st.pyplot(fig)

        with g_col2:
            # pH Scale Gauge Chart
            ph_info = category_data.get("pH", {"score": 7.0, "val_unit": "7.0"})
            ph_val = float(ph_info["score"])

            fig_ph, ax_ph = plt.subplots(figsize=(4.8, 3.8))
            # Scale background: 5.0 to 8.5
            ph_scale_range = [5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
            ax_ph.barh([0], [8.5 - 5.0], left=5.0, color="#E2E8F0", height=0.35, edgecolor="#718096", linewidth=1.0)

            # Marker for predicted pH
            ax_ph.plot([ph_val], [0], marker="o", markersize=14, color="#3182CE", markeredgecolor="#1A365D", markeredgewidth=2)
            ax_ph.text(ph_val, 0.28, f"pH {ph_val}", ha="center", fontsize=11, fontweight="bold", color="#1A365D")

            ax_ph.set_xlim(4.8, 8.7)
            ax_ph.set_ylim(-0.6, 0.6)
            ax_ph.set_yticks([])
            ax_ph.set_xticks(ph_scale_range)
            ax_ph.set_xticklabels([str(v) for v in ph_scale_range], fontsize=8.5)
            ax_ph.set_xlabel("Urine pH Scale (Reference Chart)", fontsize=10)
            ax_ph.set_title("pH Reading on Reference Scale", fontsize=11, fontweight="bold", pad=12)

            ax_ph.spines["top"].set_visible(False)
            ax_ph.spines["right"].set_visible(False)
            ax_ph.spines["left"].set_visible(False)
            ax_ph.grid(axis="x", linestyle=":", alpha=0.6)

            st.pyplot(fig_ph)
            st.caption("ℹ️ Note: pH represents acidity/alkalinity; a higher pH position does not indicate abnormality.")

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 4. SIMPLE FACTUAL RESULT SUMMARY
        # -------------------------------------------------------------
        st.subheader("4. 📝 Factual Result Summary")
        st.markdown(
            "<div class='summary-box'>",
            unsafe_allow_html=True
        )

        for param in TARGET_PARAMETERS:
            info = category_data[param]
            cat = info["category"]
            val_u = info["val_unit"]

            if not info["valid"]:
                st.markdown(f"• **{param}**: Unable to detect reagent pad reliably from the input image.", unsafe_allow_html=True)
            elif param == "pH":
                st.markdown(f"• **pH**: Reference scale reading of **{val_u}** (physiological urine range typically spans 5.0 to 8.5).", unsafe_allow_html=True)
            elif cat == "Negative":
                st.markdown(f"• **{param}**: **Negative** on the reference scale.", unsafe_allow_html=True)
            else:
                st.markdown(f"• **{param}**: **{cat}** level detected on the reference scale ({val_u}).", unsafe_allow_html=True)

        st.markdown(
            "<small style='display:block;margin-top:10px;color:#718096;'>"
            "<em>Summary is strictly factual and non-diagnostic. No medical condition or clinical interpretation is implied.</em>"
            "</small>"
            "</div>",
            unsafe_allow_html=True
        )

        # -------------------------------------------------------------
        # 5. EXPANDABLE TECHNICAL DETAILS (collapsed by default)
        # -------------------------------------------------------------
        with st.expander("🛠️ Technical Details & Extracted Color Metrics (Expand)"):
            st.markdown("#### Reagent Pad Colorimeter Swatches")
            st.caption("Each crop is segmented via OpenCV and validated against empty/dark boundaries before classification.")

            pad_cols = st.columns(6)
            for idx, param in enumerate(TARGET_PARAMETERS):
                p_data = results["pad_results"][param]
                rgb = p_data["RGB"]
                hsv = p_data["HSV"]
                lab = p_data["LAB"]

                with pad_cols[idx]:
                    st.markdown(f"**{param}**")
                    st.markdown(
                        f"<div style='width:100%;height:32px;background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});"
                        f"border:1px solid #4A5568;border-radius:4px;margin-bottom:6px;'></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<small>"
                        f"<strong>RGB:</strong> {rgb}<br>"
                        f"<strong>HSV:</strong> {hsv}<br>"
                        f"<strong>LAB:</strong> [{lab[0]}, {lab[1]}, {lab[2]}]<br>"
                        f"<strong>Valid:</strong> {'✅ Yes' if p_data['valid'] else '❌ No'}"
                        f"</small>",
                        unsafe_allow_html=True
                    )

            st.markdown("---")
            st.markdown("#### Classifier Specifications")
            st.write(
                f"- **Model Architecture:** `{results['model_used']}`\n"
                f"- **Feature Dimension:** `9 colorimetric features (R, G, B, H, S, V, L, a*, b*)`\n"
                f"- **Pad Localization Method:** `Collinear square pad clustering with adaptive thresholding & letterbox clearing`\n"
                f"- **Target Classes:** `37 distinct manufacturer-level categories across 6 parameters`"
            )

            st.markdown("---")
            st.markdown("#### Synthetic Training Data Methodology")
            st.info(
                "ℹ️ **Reference Chart Synthetic Data Generation:**\n\n"
                "Because physical test strips in public datasets lack granular ground-truth concentration labels for all pad levels, "
                "the level classifiers were trained on synthetic color distributions derived directly from the manufacturer reference color chart. "
                "Small Gaussian perturbations (σ=3.0) in RGB, HSV, and CIELAB space were applied around the reference colors to simulate minor lighting "
                "and camera variations without altering class semantics. This enables real-world strip reads without requiring thousands of manually labeled clinical strips."
            )

            st.markdown("---")
            st.markdown("#### Overall Strip Colorimetry")
            feats = results["features"]
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("Mean RGB", f"[{feats['R_mean']:.0f}, {feats['G_mean']:.0f}, {feats['B_mean']:.0f}]")
            f_col2.metric("Mean HSV", f"[{feats['H_mean']:.0f}, {feats['S_mean']:.0f}, {feats['V_mean']:.0f}]")
            f_col3.metric("Strip ROI", f"{results['strip_roi']}")

        # -------------------------------------------------------------
        # 6. DISCLAIMER
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(
            "⚠️ **Disclaimer:** “This is an educational mini-project prototype demonstrating colorimetry and "
            "classical machine learning (Random Forest / SVM). It is not a certified medical device and should "
            "not be used for medical diagnosis, clinical decision-making, or self-treatment.”"
        )
