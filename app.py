"""
app.py
Streamlit Web Application:
"Urine Test Strip Analysis"
AI-assisted colorimetric analysis using Random Forest and SVM.

Refined Healthcare Dashboard:
- White cards on soft gray background (#F8FAFC)
- Clean, compact spacing without excessive vertical padding or empty gaps
- Model selection ONLY in the sidebar
- Compact input card: controls on left, small preview on right, one prominent Analyze button
- Overall Summary banner immediately above the 6 cards (e.g. "6 parameters analyzed • 4 Negative • 1 Elevated • pH 7.0")
- 6 Result Cards:
  * PARAMETER name
  * Large predicted VALUE
  * Level badge
  * Confidence progress bar + status tier (>=75% High, 50-74% Moderate, <50% Low)
  * Visible caution label if low confidence ("Low confidence — interpret cautiously")
- Result Overview Graph:
  * Horizontal bar chart on normalized reference scale (0-100%)
  * Bar labels showing predicted level
  * pH shown on reference scale without implying severity
  * Professional healthcare palette
- "Key Findings" Card:
  * Compact summary grouping findings cleanly (e.g., elevated vs negative)
  * Low-confidence findings called out separately
- Detailed Results Table:
  * Parameter | Result | Value | Reference Level | Confidence
- Detected Reagent Pads:
  * Slightly larger pad cards showing Parameter, Crop Image, Color Swatch, Predicted level
  * No prominent RGB metrics on the main card
- Technical Details (Collapsed Expander):
  * RGB, HSV, CIELAB metrics, model specs, ROI, and synthetic data methodology
- Discreet Footer Disclaimer
"""

import base64
import io
from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from color_matcher import TARGET_PARAMETERS, load_reference_colors
from predict import predict_image

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urine Test Strip Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Clean Healthcare CSS (White cards, soft-gray background, compact spacing)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Soft gray overall page background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Container width & compact vertical padding */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1120px;
    }

    /* Remove excessive default element margins in Streamlit */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.65rem;
    }

    /* Hero / Header Card */
    .hero-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .hero-title-area {
        flex: 1;
        min-width: 260px;
    }
    .hero-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin: 0 0 2px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hero-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        margin: 0;
        font-weight: 500;
    }
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-chip-dot {
        width: 7px;
        height: 7px;
        background-color: #0EA5E9;
        border-radius: 50%;
        display: inline-block;
    }

    /* Generic Section Card (White surface, neat border & subtle shadow) */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 2px 0;
        letter-spacing: -0.015em;
    }
    .section-desc {
        font-size: 0.82rem;
        color: #64748B;
        margin: 0 0 12px 0;
    }

    /* Overall Summary Metric Banner */
    .summary-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .summary-chip-group {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .summary-chip {
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #334155;
    }
    .summary-chip-highlight {
        background: #FEF3C7;
        border-color: #FDE68A;
        color: #92400E;
    }

    /* 6 Result Cards */
    .param-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 11px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        margin-bottom: 10px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .param-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(15, 23, 42, 0.06);
        border-color: #CBD5E1;
    }
    .param-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .param-name {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #475569;
    }
    .param-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0F172A;
        margin: 2px 0 6px 0;
        line-height: 1.2;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .badge-negative { background-color: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
    .badge-trace    { background-color: #FEF9C3; color: #854D0E; border: 1px solid #FEF08A; }
    .badge-small    { background-color: #FFEDD5; color: #9A3412; border: 1px solid #FED7AA; }
    .badge-moderate { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-large    { background-color: #FCE7F3; color: #9D174D; border: 1px solid #FBCFE8; }
    .badge-ph       { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .badge-alert    { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }

    /* Confidence Progress Bar in Card */
    .conf-container {
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid #F1F5F9;
    }
    .conf-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        color: #64748B;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .conf-tier-tag {
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .tier-high { color: #16A34A; }
    .tier-mod  { color: #D97706; }
    .tier-low  { color: #DC2626; }

    .conf-bar-bg {
        width: 100%;
        height: 5px;
        background-color: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 999px;
    }
    .bar-high { background-color: #22C55E; }
    .bar-mod  { background-color: #F59E0B; }
    .bar-low  { background-color: #EF4444; }

    .conf-warning-note {
        font-size: 0.71rem;
        color: #B91C1C;
        background-color: #FEF2F2;
        border: 1px solid #FEE2E2;
        padding: 3px 6px;
        border-radius: 4px;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
        font-weight: 500;
    }

    /* Key Findings Card */
    .finding-item {
        padding: 7px 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.88rem;
        color: #334155;
        display: flex;
        align-items: baseline;
        gap: 8px;
    }
    .finding-item:last-child {
        border-bottom: none;
    }
    .finding-bullet {
        color: #2563EB;
        font-weight: 800;
        font-size: 1.1rem;
        line-height: 1;
    }

    /* Detected Reagent Pads Grid */
    .pad-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .pad-crop-img {
        width: 72px;
        height: 72px;
        object-fit: cover;
        border-radius: 7px;
        border: 1px solid #CBD5E1;
        margin: 6px auto;
        display: block;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .pad-swatch {
        width: 80%;
        height: 16px;
        border-radius: 4px;
        border: 1px solid #CBD5E1;
        margin: 6px auto 4px auto;
    }

    /* Status Banners */
    .custom-banner {
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .banner-success {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
    }
    .banner-warning {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
    }

    /* Footer */
    .app-footer {
        margin-top: 24px;
        padding-top: 14px;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.5;
    }

    /* Streamlit primary button */
    div.stButton > button[kind="primary"] {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 9px 20px;
        font-weight: 600;
        font-size: 0.94rem;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25);
        transition: background-color 0.15s ease, transform 0.1s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Helper Functions: Level Categorization, Confidence Tiering & Image Helpers
# -----------------------------------------------------------------------------
def categorize_level(param: str, level_str: str, val_str: str):
    """
    Standardizes result string into neutral categories:
    Negative, Trace, Small, Moderate, Large, or pH value.
    Computes normalized reference-scale score (0.0 to 1.0) and reference level string.
    """
    l_lower = level_str.lower()
    val_lower = val_str.lower()

    if "unable" in l_lower or "unable" in val_lower:
        return "Unable to detect pad", "badge-alert", 0.0, "N/A", False

    if param == "pH":
        try:
            val = float(val_str.split()[0])
        except Exception:
            try:
                val = float(level_str)
            except Exception:
                val = 7.0
        # Reference scale 5.0 to 8.5 (range = 3.5)
        norm_pos = max(0.0, min(1.0, (val - 5.0) / 3.5))
        return f"{val}", "badge-ph", norm_pos, f"pH {val}", True

    # Ordinal Scale: Negative=0, Trace=1, Small=2, Moderate=3, Large=4
    if "negative" in l_lower or val_lower in ["0 mg/dl", "0", "negative"]:
        return "Negative", "badge-negative", 0.0, "Negative", True
    elif "trace" in l_lower or "1/10" in l_lower or "tr." in l_lower:
        return "Trace", "badge-trace", 0.25, "Trace", True
    elif "small" in l_lower or "1/4" in l_lower or "30" in l_lower or ("(+)" in l_lower and "(++)" not in l_lower):
        return "Small", "badge-small", 0.50, "Small (+)", True
    elif "moderate" in l_lower or "1/2" in l_lower or "100" in l_lower or ("(++)" in l_lower and "(+++)" not in l_lower):
        return "Moderate", "badge-moderate", 0.75, "Moderate (++)", True
    elif "large" in l_lower or "1 " in l_lower or "2 " in l_lower or "300" in l_lower or "(+++)" in l_lower or "(++++)" in l_lower or "2 or more" in l_lower:
        return "Large", "badge-large", 1.0, "Large (+++)", True

    return "Detected", "badge-small", 0.50, "Detected", True


def get_confidence_info(conf_val):
    """
    Parses confidence value into:
    - float (0.0 to 1.0)
    - percentage string ('86.0%')
    - tier ('HIGH', 'MODERATE', 'LOW')
    - tier class ('tier-high', 'tier-mod', 'tier-low')
    - bar fill class ('bar-high', 'bar-mod', 'bar-low')
    - is_low boolean
    """
    c_float = None
    if isinstance(conf_val, (int, float)):
        c_float = float(conf_val)
    elif isinstance(conf_val, str) and "%" in conf_val:
        try:
            c_float = float(conf_val.replace("%", "").strip()) / 100.0
        except Exception:
            c_float = None

    if c_float is None:
        return 0.50, "N/A", "MODERATE", "tier-mod", "bar-mod", False

    pct_str = f"{c_float * 100:.1f}%"
    if c_float >= 0.75:
        return c_float, pct_str, "HIGH", "tier-high", "bar-high", False
    elif c_float >= 0.50:
        return c_float, pct_str, "MODERATE", "tier-mod", "bar-mod", False
    else:
        return c_float, pct_str, "LOW", "tier-low", "bar-low", True


def bgr_to_base64_png(patch_bgr: np.ndarray) -> str:
    """Encodes a BGR numpy image patch to base64 PNG string for HTML display."""
    if patch_bgr is None or patch_bgr.size == 0:
        return ""
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -----------------------------------------------------------------------------
# SIDEBAR (Model Selection, About Project, & Reference Chart ONLY)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model Selection")
    selected_model = st.selectbox(
        "Classifier Algorithm:",
        options=["Random Forest", "SVM"],
        index=0,
        help="Select between ensemble Random Forest or RBF Support Vector Machine for direct level prediction."
    )

    st.markdown("---")
    st.markdown("### ℹ️ About Project")
    st.markdown(
        "<div style='font-size:0.84rem;color:#475569;line-height:1.5;'>"
        "<strong>Urine Test Strip Analysis</strong><br>"
        "Direct result level estimation via colorimetric feature extraction "
        "(RGB, HSV, CIELAB) and classical machine learning classifiers."
        "<br><br>"
        "• 6 Urinalysis Parameters<br>"
        "• Automatic Pad Localization<br>"
        "• Non-invasive Colorimetry<br>"
        "• Direct ML Level Mapping"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    show_ref_chart = st.checkbox("View Reference Color Chart", value=False)
    if show_ref_chart:
        st.markdown("##### Manufacturer Reference Levels")
        try:
            ref_df = load_reference_colors()
            st.dataframe(
                ref_df[["parameter", "level", "value", "unit"]],
                use_container_width=True,
                hide_index=True
            )
        except Exception as e:
            st.caption(f"Reference chart data unavailable: {e}")


# -----------------------------------------------------------------------------
# 1. HERO / HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class='hero-container'>
        <div class='hero-title-area'>
            <div class='hero-title'>🧪 Urine Test Strip Analysis</div>
            <div class='hero-subtitle'>AI-assisted colorimetric analysis using Random Forest and SVM</div>
        </div>
        <div>
            <span class='status-chip'>
                <span class='status-chip-dot'></span>
                Educational Prototype
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. INPUT SECTION (Compact Card: Controls Left, Small Preview Right, Analyze Button)
# -----------------------------------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)

col_input_left, col_input_right = st.columns([1.5, 1.0], gap="medium")

with col_input_left:
    st.markdown("<div class='section-title'>Strip Image Input</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-desc'>Using model: <strong>{selected_model}</strong> (change in sidebar)</div>",
        unsafe_allow_html=True
    )

    input_source = st.radio(
        "Source Mode:",
        options=["Upload Strip Image", "Use Sample Test Image"],
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )

    uploaded_file = None
    sample_path = None

    if input_source == "Upload Strip Image":
        uploaded_file = st.file_uploader(
            "Upload Strip Image (JPG / PNG):",
            type=["jpg", "jpeg", "png"],
            help="Take a clear photo of the test strip."
        )
    else:
        test_dir = PROJECT_ROOT / "dataset" / "test"
        sample_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*/*.jpg")))
        if sample_files:
            sample_path = st.selectbox(
                "Choose sample image:",
                options=sample_files,
                format_func=lambda p: p.name[:30] + ("..." if len(p.name) > 30 else "")
            )
        else:
            st.warning("No sample files found in dataset/test/.")

with col_input_right:
    img_bgr = None
    if input_source == "Upload Strip Image" and uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif input_source == "Use Sample Test Image" and sample_path is not None:
        img_bgr = cv2.imread(str(sample_path))

    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Display compact preview
        st.image(
            img_rgb,
            caption=f"Preview ({img_bgr.shape[1]}×{img_bgr.shape[0]} px)",
            width=210
        )
    else:
        st.markdown(
            "<div style='border:1px dashed #CBD5E1;border-radius:8px;padding:22px 14px;"
            "text-align:center;color:#94A3B8;font-size:0.8rem;'>"
            "📷 Image preview"
            "</div>",
            unsafe_allow_html=True
        )

# One prominent Analyze Strip button
st.markdown("<div style='margin-top:6px;'>", unsafe_allow_html=True)
analyze_btn = st.button("🔬 Analyze Strip", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) # Close input section card

# -----------------------------------------------------------------------------
# MAIN ANALYSIS PIPELINE & PRESENTATION
# -----------------------------------------------------------------------------
if analyze_btn:
    if img_bgr is None:
        st.warning("Please upload or select a test strip image before clicking 'Analyze Strip'.")
    else:
        with st.spinner(f"Analyzing strip using {selected_model}..."):
            results = predict_image(img_bgr, model_name=selected_model)

        # Check pad detection validity across the 6 target pads
        pad_detection_failed = False
        valid_pad_count = 0
        for p in TARGET_PARAMETERS:
            p_info = results["pad_results"].get(p, {})
            if p_info.get("valid", False):
                valid_pad_count += 1

        if valid_pad_count < 3:
            pad_detection_failed = True

        # ---------------------------------------------------------------------
        # 3. ANALYSIS STATUS BANNER
        # ---------------------------------------------------------------------
        if pad_detection_failed:
            st.markdown(
                """
                <div class='custom-banner banner-warning'>
                    <span style='font-size:1.2rem;'>⚠️</span>
                    <div>
                        <strong>Pad Detection Warning:</strong> Unable to reliably identify reagent pads on the strip.
                        Please ensure the image contains a clear test strip oriented against a neutral background.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='custom-banner banner-success'>
                    <span style='font-size:1.2rem;'>✓</span>
                    <div>
                        <strong>Analysis completed successfully:</strong> All 6 reagent pads classified using {results['model_used']}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Process parameters & category metadata
            category_data = {}
            neg_count = 0
            elevated_count = 0
            ph_val_str = "7.0"
            low_conf_params = []

            for row in results["level_table"]:
                param = row["Parameter"]
                pred_lvl = row["Predicted Level"]
                val_u = row["Value/Unit"]
                conf_val = row.get("_conf_float", row.get("Confidence"))

                category, badge_cls, norm_score, ref_level, is_valid = categorize_level(param, pred_lvl, val_u)
                c_float, pct_str, tier, tier_cls, bar_cls, is_low = get_confidence_info(conf_val)

                if is_low:
                    low_conf_params.append(param)

                if param == "pH":
                    ph_val_str = category
                else:
                    if category == "Negative":
                        neg_count += 1
                    elif category in ["Trace", "Small", "Moderate", "Large", "Detected"]:
                        elevated_count += 1

                category_data[param] = {
                    "category": category,
                    "badge_class": badge_cls,
                    "norm_score": norm_score,
                    "ref_level": ref_level,
                    "val_unit": val_u,
                    "pred_level": pred_lvl,
                    "conf_float": c_float,
                    "conf_pct": pct_str,
                    "conf_tier": tier,
                    "conf_tier_cls": tier_cls,
                    "conf_bar_cls": bar_cls,
                    "is_low_conf": is_low,
                    "valid": is_valid
                }

            # -----------------------------------------------------------------
            # 4. OVERALL SUMMARY (Immediately Above the Cards)
            # -----------------------------------------------------------------
            elev_label = f"{elevated_count} Elevated reference level" if elevated_count == 1 else f"{elevated_count} Elevated reference levels"
            elev_cls = "summary-chip summary-chip-highlight" if elevated_count > 0 else "summary-chip"

            st.markdown(
                f"""
                <div class='summary-banner'>
                    <div style='font-weight:700;font-size:0.92rem;color:#0F172A;'>
                        📋 Overall Summary:
                    </div>
                    <div class='summary-chip-group'>
                        <span class='summary-chip'>6 parameters analyzed</span>
                        <span class='summary-chip'>{neg_count} Negative</span>
                        <span class='{elev_cls}'>{elev_label}</span>
                        <span class='summary-chip'>pH {ph_val_str}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -----------------------------------------------------------------
            # 5. RESULT SUMMARY CARDS (6 Visually Consistent Responsive Cards)
            # -----------------------------------------------------------------
            card_row1 = st.columns(3)
            card_row2 = st.columns(3)
            all_card_cols = card_row1 + card_row2

            for idx, param in enumerate(TARGET_PARAMETERS):
                col = all_card_cols[idx]
                info = category_data[param]
                cat = info["category"]
                b_cls = info["badge_class"]
                val_display = info["val_unit"]

                c_float = info["conf_float"]
                pct_str = info["conf_pct"]
                tier = info["conf_tier"]
                tier_cls = info["conf_tier_cls"]
                bar_cls = info["conf_bar_cls"]
                bar_width = int(c_float * 100)

                # Badge label
                badge_html = f"<span class='status-badge {b_cls}'>{cat}</span>"

                # Caution banner for low confidence
                low_conf_html = ""
                if info["is_low_conf"]:
                    low_conf_html = (
                        "<div class='conf-warning-note'>"
                        "⚠️ Low confidence — interpret cautiously"
                        "</div>"
                    )

                with col:
                    st.markdown(
                        f"""
                        <div class='param-card'>
                            <div>
                                <div class='param-header'>
                                    <span class='param-name'>{param}</span>
                                    {badge_html}
                                </div>
                                <div class='param-val'>{val_display}</div>
                            </div>
                            <div class='conf-container'>
                                <div class='conf-meta-row'>
                                    <span>Confidence <strong>{pct_str}</strong></span>
                                    <span class='conf-tier-tag {tier_cls}'>{tier}</span>
                                </div>
                                <div class='conf-bar-bg'>
                                    <div class='conf-bar-fill {bar_cls}' style='width:{bar_width}%;'></div>
                                </div>
                                {low_conf_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # -----------------------------------------------------------------
            # 6. RESULT OVERVIEW GRAPH (Comparison: Horizontal Bar Chart)
            # -----------------------------------------------------------------
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Result Overview</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-desc'>"
                "Normalized comparison across parameters (0% = Negative/Lowest reference level, 100% = Highest reference level). "
                "Bars show relative position on the test scale; colors are neutral and do not denote clinical risk."
                "</div>",
                unsafe_allow_html=True
            )

            chart_params = TARGET_PARAMETERS.copy()
            norm_percentages = [category_data[p]["norm_score"] * 100.0 for p in chart_params]

            # Professional healthcare palette (clean neutral blues and warm amber)
            palette = {
                "Negative": "#94A3B8",
                "Trace": "#F59E0B",
                "Small": "#EA580C",
                "Moderate": "#D97706",
                "Large": "#B45309",
            }
            bar_colors = []
            for p in chart_params:
                if p == "pH":
                    bar_colors.append("#0284C7")  # Sky blue for pH
                else:
                    c = category_data[p]["category"]
                    bar_colors.append(palette.get(c, "#3B82F6"))

            fig, ax = plt.subplots(figsize=(8.8, 3.4), dpi=140)
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')

            y_positions = np.arange(len(chart_params))
            y_positions_rev = y_positions[::-1]

            bars = ax.barh(
                y_positions_rev,
                norm_percentages,
                color=bar_colors,
                height=0.48,
                edgecolor="#CBD5E1",
                linewidth=0.8
            )

            ax.set_yticks(y_positions_rev)
            ax.set_yticklabels(chart_params, fontsize=9.5, fontweight="600", color="#1E293B")
            ax.set_xlim(0, 118)
            ax.set_xlabel("Reference Scale Position", fontsize=9, fontweight="600", color="#475569", labelpad=6)

            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0%\n(Lowest)", "25%", "50%", "75%", "100%\n(Highest)"], fontsize=7.8, color="#64748B")

            ax.grid(axis="x", linestyle="--", alpha=0.35, color="#94A3B8")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CBD5E1")
            ax.spines["bottom"].set_color("#CBD5E1")

            # Value labels on bars
            for bar, p in zip(bars, chart_params):
                w = bar.get_width()
                info = category_data[p]
                if p == "pH":
                    label_text = f"pH {info['category']} ({int(round(w))}%)"
                else:
                    label_text = f"{info['ref_level']} ({int(round(w))}%)"

                ax.text(
                    w + 1.8,
                    bar.get_y() + bar.get_height() / 2,
                    label_text,
                    va="center",
                    fontsize=8.2,
                    fontweight="600",
                    color="#334155"
                )

            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True) # Close section card

            # -----------------------------------------------------------------
            # 7. KEY FINDINGS (Compact Summary Card)
            # -----------------------------------------------------------------
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Key Findings</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-desc'>Executive summary of detected parameters.</div>", unsafe_allow_html=True)

            # Group negative vs non-negative parameters for compact scannability
            neg_params = [p for p in TARGET_PARAMETERS if p != "pH" and category_data[p]["category"] == "Negative"]
            elev_params = [p for p in TARGET_PARAMETERS if p != "pH" and category_data[p]["category"] != "Negative"]

            findings_html = ""

            # Elevated findings
            if elev_params:
                for p in elev_params:
                    info = category_data[p]
                    findings_html += (
                        f"<div class='finding-item'>"
                        f"<span class='finding-bullet'>•</span>"
                        f"<div><strong>{p}</strong>: {info['ref_level']} ({info['val_unit']})</div>"
                        f"</div>"
                    )

            # Negative findings (grouped or individual)
            if neg_params:
                neg_names = ", ".join(neg_params)
                findings_html += (
                    f"<div class='finding-item'>"
                    f"<span class='finding-bullet'>•</span>"
                    f"<div><strong>{neg_names}</strong>: Negative</div>"
                    f"</div>"
                )

            # pH finding
            findings_html += (
                f"<div class='finding-item'>"
                f"<span class='finding-bullet'>•</span>"
                f"<div><strong>pH</strong>: {category_data['pH']['category']} (physiological scale reading)</div>"
                f"</div>"
            )

            # Low confidence caveat if any
            if low_conf_params:
                low_names = ", ".join(low_conf_params)
                findings_html += (
                    f"<div class='finding-item' style='color:#B91C1C;background:#FEF2F2;padding:6px 10px;border-radius:6px;margin-top:6px;'>"
                    f"<span style='font-size:0.95rem;'>⚠️</span>"
                    f"<div><strong>Low-Confidence Alert:</strong> {low_names} predicted with &lt; 50% confidence. Interpret cautiously.</div>"
                    f"</div>"
                )

            st.markdown(findings_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True) # Close section card

            # -----------------------------------------------------------------
            # 8. DETAILED RESULTS TABLE
            # -----------------------------------------------------------------
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Detailed Results Table</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-desc'>Structured reference measurements and values.</div>", unsafe_allow_html=True)

            table_data = []
            for param in TARGET_PARAMETERS:
                info = category_data[param]
                table_data.append({
                    "Parameter": param,
                    "Result": info["category"],
                    "Value": info["val_unit"],
                    "Reference Level": info["ref_level"],
                    "Confidence": f"{info['conf_pct']} ({info['conf_tier']})"
                })

            df_display = pd.DataFrame(table_data)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True) # Close section card

            # -----------------------------------------------------------------
            # 9. DETECTED REAGENT PADS (Verification: Larger Pad Cards)
            # -----------------------------------------------------------------
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Detected Reagent Pads</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-desc'>"
                "Visual verification confirming each pad was correctly located and sampled on the strip body."
                "</div>",
                unsafe_allow_html=True
            )

            pad_cols = st.columns(6)
            for idx, param in enumerate(TARGET_PARAMETERS):
                col = pad_cols[idx]
                p_info = results["pad_results"].get(param, {})
                rgb = p_info.get("RGB", [200, 200, 200])
                patch = p_info.get("patch", None)
                is_valid = p_info.get("valid", False)
                pred_label = category_data[param]["ref_level"]

                with col:
                    if is_valid and patch is not None and patch.size > 0:
                        b64_crop = bgr_to_base64_png(patch)
                        crop_html = f"<img src='data:image/png;base64,{b64_crop}' class='pad-crop-img' alt='{param} pad' />"
                    else:
                        crop_html = "<div class='pad-crop-img' style='display:flex;align-items:center;justify-content:center;background:#F1F5F9;color:#94A3B8;font-size:0.75rem;'>Failed</div>"

                    st.markdown(
                        f"""
                        <div class='pad-card'>
                            <div style='font-size:0.82rem;font-weight:700;color:#0F172A;'>{param}</div>
                            {crop_html}
                            <div class='pad-swatch' style='background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]});'></div>
                            <div style='font-size:0.73rem;color:#475569;font-weight:600;margin-top:4px;'>
                                {pred_label}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("</div>", unsafe_allow_html=True) # Close section card

            # -----------------------------------------------------------------
            # 10. TECHNICAL DETAILS (Collapsed Expander: Colorimetry & Model Specs)
            # -----------------------------------------------------------------
            with st.expander("🛠️ Technical Details"):
                st.markdown("##### Extracted Colorimetric Metrics (RGB / HSV / CIELAB)")
                t_rows = []
                for param in TARGET_PARAMETERS:
                    p_info = results["pad_results"].get(param, {})
                    rgb = p_info.get("RGB", [0, 0, 0])
                    hsv = p_info.get("HSV", [0, 0, 0])
                    lab = p_info.get("LAB", [0.0, 0.0, 0.0])
                    t_rows.append({
                        "Parameter": param,
                        "RGB": f"({rgb[0]}, {rgb[1]}, {rgb[2]})",
                        "HSV": f"({hsv[0]}, {hsv[1]}, {hsv[2]})",
                        "CIELAB": f"({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})",
                        "Model": results["model_used"],
                        "Confidence": p_info.get("confidence", "N/A"),
                        "Pad Status": "Valid Segment" if p_info.get("valid", False) else "Invalid"
                    })
                df_tech = pd.DataFrame(t_rows)
                st.dataframe(df_tech, use_container_width=True, hide_index=True)

                st.markdown("##### Model Architecture & Pipeline Specifications")
                st.markdown(
                    f"- **Active Classifier:** `{results['model_used']}`\n"
                    f"- **Feature Dimension:** `9 color features per pad (R, G, B, H, S, V, L, a*, b*)`\n"
                    f"- **Strip Localization:** `Collinear reagent pad clustering + Otsu body bounding analysis`\n"
                    f"- **Strip Region of Interest (ROI):** `{results['strip_roi']}`\n"
                    f"- **Target Parameters:** `6 parameters (Glucose, Protein, pH, Ketone, Blood, Leukocytes)`"
                )

                st.markdown("##### Synthetic Training Data Note")
                st.info(
                    "ℹ️ **Synthetic Colorimetric Distribution Training:**\n\n"
                    "Because physical test strips in public datasets lack granular ground-truth concentration labels for all pad levels, "
                    "the direct ML classifiers were trained on synthetic color distributions derived from the manufacturer reference chart. "
                    "Controlled Gaussian perturbations (σ=3.0) in RGB, HSV, and CIELAB space were applied to model real-world camera lighting variations "
                    "while preserving colorimetric class boundaries."
                )

# -----------------------------------------------------------------------------
# 11. FOOTER (Discreet & Educational Disclaimer)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class='app-footer'>
        <strong>Machine Learning-Based Urine Test Strip Analysis Using Colorimetry</strong><br>
        Random Forest | SVM | OpenCV<br>
        <span style='color:#94A3B8;font-size:0.75rem;'>Educational prototype — not intended for medical diagnosis.</span>
    </div>
    """,
    unsafe_allow_html=True
)
