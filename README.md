# OmniML: Enterprise Visual Machine Learning Platform

An enterprise-grade **Machine Learning Image Classification Platform** and interactive **Analytics Dashboard** covering the full ML lifecycle in a unified single-frame interface.

---

## 1. Pipeline Architecture

| Stage | Name | Description | Key Modules & Methods |
|---|---|---|---|
| **Stage 01** | **Data Ingestion** | Ingestion of benchmark datasets (MNIST Digits, Synthetic Shapes) & custom input acquisition | `sklearn.datasets.load_digits`, OpenCV, Data Partitioning |
| **Stage 02** | **Preprocessing** | Transformation filter chain, binarization & noise reduction | Grayscale, Gaussian/Median Blur, Otsu/Adaptive Thresholding, Morphology, Normalization |
| **Stage 03** | **Feature Extraction** | Visual descriptor computation & eigenspace projection | Histogram of Oriented Gradients (HOG), Intensity Histograms, Canny Edges, LBP Texture, 3D PCA |
| **Stage 04** | **Model Training** | Multi-classifier training engine with hyperparameter configuration | Support Vector Machine (SVM), Random Forest, KNN, MLP Neural Network, Logistic Regression, Decision Tree |
| **Stage 05** | **Evaluation & Inference** | Diagnostic telemetry & real-time live testing bench | Accuracy, Macro Precision, Recall, F1-score, Confusion Matrix Heatmap, Per-Class Bar Charts, Softmax Distribution |
| **Stage 06** | **Code Studio & Guide** | In-browser script editor, execution engine & compilation guide | Live server execution, stdout/stderr capture, bytecode compilation |

---

## 2. Compilation & Execution Manual

### Prerequisites & Environment Setup
Verify Python (>= 3.9) is installed and install production dependencies:
```bash
pip install -r requirements.txt
```

---

### Bytecode Compilation
In Python, source code files (`.py`) can be compiled into optimized **Bytecode (`.pyc`)**:

```bash
# Compile backend ML pipeline engine
python -m py_compile ml_pipeline.py

# Compile dashboard application
python -m py_compile app.py
```

---

### Executing CLI Automated Pipeline
To execute the automated 5-stage pipeline from your terminal:
```bash
python ml_pipeline.py
```

**Expected Execution Output:**
```
============================================================
=== EXECUTING 5-STAGE MACHINE LEARNING PIPELINE ===
============================================================

[Stage 1] Loading Dataset: 'digits'...
 -> Loaded 1797 images across 10 classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

[Stage 2 & 3] Preprocessing Images & Extracting Features: 'HOG'...
 -> Feature Matrix Shape: (1797, 1296), Label Vector Shape: (1797,)
 -> Train Samples: 1347, Test Samples: 450

[Stage 4] Training Model: 'Support Vector Machine (SVM)'...
 -> Model trained successfully in 3.55 seconds!

[Stage 5] Performance Evaluation:
 -> Accuracy Score   : 99.56%
 -> Macro Precision  : 99.56%
 -> Macro Recall     : 99.56%
 -> Macro F1-Score   : 99.56%
 -> Macro ROC-AUC    : 100.00%

Confusion Matrix:
[[45  0  0  0  0  0  0  0  0  0]
 [ 0 46  0  0  0  0  0  0  0  0]
 ...
 [ 0  0  0  0  0  0  0  1  0 44]]

============================================================
[SUCCESS] PIPELINE EXECUTION COMPLETE
============================================================
```

---

### Launching the Web Dashboard
Launch the web application dashboard in your browser:
```bash
streamlit run app.py
```
The interface will automatically open at:
```
http://localhost:8501
```

---

## 3. Repository File Structure

```
Machine learning/
├── app.py              # Single-Frame Executive Web Dashboard (All 6 Stages)
├── ml_pipeline.py      # Standalone 5-Stage ML Pipeline Engine (Python OOP Script)
├── requirements.txt    # Production Dependencies
├── README.md           # Documentation & Compilation Manual
└── digit_mlp_model.pkl # Exported Trained Model Checkpoint
```
