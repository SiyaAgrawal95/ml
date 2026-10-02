# Machine Learning-Based Urine Test Strip Analysis Using Colorimetry

A complete, transparent, and interpretable college mini-project for **direct prediction of urinalysis result levels and values** using **digital colorimetry** and classical machine learning (**Random Forest & Support Vector Machine**).

> **Important Educational Prototype Notice:**  
> The value classifiers are trained on synthetic color variations generated from manufacturer/reference-chart colors and are intended only as an educational prototype, not for clinical use.

---

## 📌 Project Architecture

Traditional urine test strip analysis assesses color changes across distinct chemical reagent pads. Instead of using complex deep learning architectures (CNNs, YOLO, TensorFlow, PyTorch), this project uses **transparent digital colorimetry** and **classical machine learning**:

1. **One-Time Reference Chart Ingestion:**  
   The official Urinalysis Reagent Strip Reference Chart (`data/reference_chart.png`) was sampled across all levels of the 6 target parameters.
2. **Synthetic Reference-Chart-Derived Training Dataset (`data/synthetic_level_training_data.csv`):**  
   Small Gaussian variations ($\sigma = 4.0$) were generated around each reference color to simulate real-world camera and illumination variations without changing the clinical level meaning.
3. **Direct Level Classifiers (`models/level_models/`):**  
   Dedicated Random Forest and SVM models were trained for each parameter to directly predict exact clinical result levels (e.g. *Negative*, *Trace*, *250 mg/dL*, *6.5*, etc.).
4. **Runtime Prediction Pipeline:**  
   When a strip image is uploaded, the system localizes the strip, crops the 6 reagent pads, extracts their $[R, G, B, H, S, V, L, A, B_{lab}]$ color features, and passes each pad to its trained classifier. **No runtime reference-chart lookup is performed.**
5. **Interactive Streamlit Web UI:**  
   Displays the uploaded strip preview, extracted pad colors, and the primary output table:
   ```text
   Parameter | Predicted Level | Value/Unit | Confidence
   ```

---

## 📂 Project Structure

```text
MLmini/
├── dataset/                               # Roboflow Multi-Label Dataset
│   ├── train/                             # 2,394 training images & _classes.csv
│   ├── valid/                             # 98 validation images & _classes.csv
│   └── test/                              # 49 test images & _classes.csv
├── data/
│   ├── reference_chart.png                # Official Urinalysis Reference Chart
│   ├── reference_colors.csv               # 37 reference color levels
│   ├── synthetic_level_training_data.csv  # 5,550 reference-derived training samples
│   ├── train_features.csv                 # Strip-level train features
│   ├── valid_features.csv                 # Strip-level valid features
│   ├── test_features.csv                  # Strip-level test features
│   └── eda_plots/                         # Exploratory data analysis charts
├── models/
│   ├── level_models/                      # Direct ML result-level classifiers
│   │   ├── rf_Glucose.pkl, svm_Glucose.pkl
│   │   ├── rf_Protein.pkl, svm_Protein.pkl
│   │   ├── rf_pH.pkl, svm_pH.pkl
│   │   ├── rf_Ketone.pkl, svm_Ketone.pkl
│   │   ├── rf_Blood.pkl, svm_Blood.pkl
│   │   ├── rf_Leukocytes.pkl, svm_Leukocytes.pkl
│   │   └── level_metadata.json            # Level to value/unit mapping
│   ├── random_forest_multilabel.pkl       # Whole-strip presence classifier (RF)
│   ├── svm_model_multilabel.pkl           # Whole-strip presence classifier (SVM)
│   └── scaler.pkl                         # StandardScaler for presence model
├── src/
│   ├── extract_reference_colors.py        # Ingests chart into reference_colors.csv
│   ├── generate_level_training_data.py    # Generates synthetic training dataset
│   ├── train_level_models.py              # Trains 6 parameter RF & SVM classifiers
│   ├── predict_levels.py                  # Direct ML level prediction pipeline
│   ├── inspect_dataset.py                 # Multi-label dataset inspector
│   ├── extract_features.py                # Whole-strip feature extraction
│   ├── train_models.py                    # Multi-label presence model training
│   ├── evaluate_models.py                 # Multi-label evaluation & metrics
│   └── predict.py                         # Unified CLI & frontend prediction interface
├── app.py                                 # Streamlit Web UI
├── requirements.txt                       # Dependencies
└── README.md                              # Complete documentation
```

---

## 🎯 Target Parameters & Chart Levels

| Parameter | Chart Reference Levels | Value / Unit |
| :--- | :--- | :--- |
| **Glucose** | Negative, 1/10 (tr.), 1/4 (+), 1/2 (++), 1 (+++), 2 or more (++++) | 0, 100, 250, 500, 1000, 2000+ mg/dL |
| **Protein** | Negative, Trace, Small (+), Moderate (++), Large (+++), Large (++++) | 0, Trace, 30, 100, 300, 2000+ mg/dL |
| **pH** | 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5 | 5.0 to 8.5 pH |
| **Ketone** | Negative, Trace, Small (+), Moderate (++), Large (+++), Large (++++) | 0, 5, 15, 40, 80, 160 mg/dL |
| **Blood** | Negative, Non-Hemolyzed Trace, Non-Hemolyzed Moderate, Hemolyzed Trace, Small (+), Moderate (++), Large (+++) | Negative to Large cells/uL |
| **Leukocytes** | Negative, Trace, Small (+), Moderate (++), Large (+++) | Negative, Trace, 15, 70, 125 Leu/uL |

---

## 🔬 Direct Level Classifier Performance

Evaluation on held-out test split (20% stratified test split):

| Parameter | Number of Classes | Random Forest Test Acc | RF F1-Score | SVM Test Acc (with Scaler) | SVM F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Glucose** | 6 | **100.0%** | **1.0000** | **100.0%** | **1.0000** |
| **Protein** | 6 | **98.9%** | **0.9889** | **99.4%** | **0.9944** |
| **pH** | 7 | **100.0%** | **1.0000** | **100.0%** | **1.0000** |
| **Ketone** | 6 | **99.4%** | **0.9944** | **99.4%** | **0.9944** |
| **Blood** | 7 | **99.0%** | **0.9905** | **99.5%** | **0.9952** |
| **Leukocytes** | 5 | **100.0%** | **1.0000** | **99.3%** | **0.9933** |

---

## 🚀 How to Run

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest Reference Chart & Build Training Data
```bash
python src/extract_reference_colors.py
python src/generate_level_training_data.py
```

### 3. Train Direct Level Classifiers
```bash
python src/train_level_models.py
```

### 4. Run Prediction on Any Image via CLI
```bash
python src/predict.py --image "dataset/test/23_jpg.rf.bfb3a582db9234141a2009ba9fda482c.jpg" --model rf
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🖥️ Streamlit Web Interface

The user interface has been redesigned for clinical clarity and user comprehension:
1. **Top Summary Cards:** Six clean result cards displaying Parameter, Predicted Value, Level Badge (Negative, Trace, Small, Moderate, Large), and Confidence.
2. **Urinalysis Result Table:** Formatted summary of `Parameter | Value | Level | Confidence`.
3. **Parameter Level Analysis Graph:**
   - Normalized Level Scores (0–100%) for ordinal parameters (Negative = 0%, Trace = 25%, Small = 50%, Moderate = 75%, Large = 100%).
   - Dedicated pH reference scale gauge (5.0 to 8.5) highlighting the exact pH position without implying abnormality.
4. **Factual Result Summary:** Plain-language, non-diagnostic bullet points reporting observations on the reference scale.
5. **Expandable Technical Details:** Collapsed section containing individual pad crops, RGB, HSV, CIELAB metrics, classifier parameters, and synthetic training data notices.
6. **Prominent Educational Disclaimer:** Explicitly notifies users that this is an educational prototype and not a medical device.

---

## 🛡️ Robust Pad Localization & Bug Fixes
- **Letterbox Removal:** Automatically detects and strips outer black borders/letterboxing (`mean < 20`) to prevent false edge contours.
- **Collinear Pad Cluster Detection:** Groups square reagent pad contours (aspect ratio 0.4–2.5) along the strip axis (horizontal and vertical) with brightness verification.
- **Empty Crop Prevention:** Every pad crop is validated for minimum average brightness (`mean(RGB) >= 25`) and standard deviation; if localization fails, the system outputs `"Unable to detect pad"` instead of predicting from black pixels.

