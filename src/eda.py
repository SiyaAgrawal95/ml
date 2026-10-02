"""
src/eda.py
Exploratory Data Analysis (EDA) for Urine Strip Multi-Label Colorimetry Project.
Generates:
1. Multi-label frequency breakdown.
2. RGB & HSV feature statistics (mean, std, min, max).
3. RGB & HSV feature visualizations.
Saves plots to data/eda_plots/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from extract_features import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "color_features.csv"
PLOTS_DIR = DATA_DIR / "eda_plots"


def run_eda(csv_path: Path = CSV_PATH, plots_dir: Path = PLOTS_DIR):
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Features file not found at {csv_path}. Run extract_features.py first.")

    df = pd.read_csv(csv_path)

    print("=" * 75)
    print("   Exploratory Data Analysis (Multi-Label Colorimetry)")
    print("=" * 75)

    split_col = "split" if "split" in df.columns else "Split"
    print(f"\nTotal samples: {len(df)}")
    print(f"Splits present: {df[split_col].value_counts().to_dict()}")

    # Identify label columns
    non_label_cols = set(FEATURE_COLS + ["filename", "split", "Split", "Image", "Relative_Path"])
    label_cols = [c for c in df.columns if c not in non_label_cols]

    print("\n--- Multi-Label Detection Frequencies (Overall) ---")
    pos_counts = df[label_cols].sum()
    freq_df = pd.DataFrame({
        "Parameter": label_cols,
        "Detected (1)": pos_counts.values,
        "Percentage": [f"{(v / len(df)) * 100:.1f}%" for v in pos_counts.values]
    })
    print(freq_df.to_string(index=False))

    # Feature Statistics
    print("\n--- RGB & HSV Feature Statistics ---")
    stats = df[FEATURE_COLS].describe().T[["mean", "std", "min", "50%", "max"]]
    stats.columns = ["Mean", "Std", "Min", "Median", "Max"]
    print(stats.to_string())

    stats.to_csv(plots_dir / "feature_statistics.csv")

    # Visualizations
    # Plot A: Label Frequencies
    fig, ax = plt.subplots(figsize=(10, 5))
    pos_counts.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title("Urine Test Strip Multi-Label Positive Counts", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Detected Samples")
    ax.set_ylabel("Parameter")
    plt.tight_layout()
    fig.savefig(plots_dir / "label_distribution.png", dpi=150)
    plt.close(fig)

    # Plot B: RGB Mean Distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = [("R_mean", "red", "Red Mean"), ("G_mean", "green", "Green Mean"), ("B_mean", "blue", "Blue Mean")]
    for ax, (col, c, title) in zip(axes, colors):
        ax.hist(df[col], bins=30, color=c, alpha=0.7, edgecolor="black")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Intensity (0-255)")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("RGB Color Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(plots_dir / "rgb_distribution.png", dpi=150)
    plt.close(fig)

    # Plot C: HSV Mean Distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    hsv_specs = [
        ("H_mean", "#d95f02", "Hue Mean (0-179)"),
        ("S_mean", "#7570b3", "Saturation Mean (0-255)"),
        ("V_mean", "#e7298a", "Value / Brightness Mean (0-255)")
    ]
    for ax, (col, c, title) in zip(axes, hsv_specs):
        ax.hist(df[col], bins=30, color=c, alpha=0.7, edgecolor="black")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("HSV Color Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(plots_dir / "hsv_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved successfully to: {plots_dir}/")
    print("=" * 75)


if __name__ == "__main__":
    run_eda()
