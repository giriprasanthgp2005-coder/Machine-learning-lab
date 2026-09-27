"""
================================================================================
OMNIML | ENTERPRISE VISUAL MACHINE LEARNING PLATFORM
================================================================================
Continuous Sequential 6-Step Visual Machine Learning Pipeline:
STEP 01: Data Ingestion & Dataset Management
STEP 02: Image Preprocessing & Spatial Transformations
STEP 03: Feature Engineering & Dimensionality Reduction
STEP 04: Classifier Training & Hyperparameter Tuning
STEP 05: Performance Evaluation & Real-Time Inference Bench
STEP 06: Interactive Code Studio & CLI Execution Manual
================================================================================
"""

import os
import sys
import io
import time
import base64
import contextlib
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    st_canvas = None
    CANVAS_AVAILABLE = False

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog, local_binary_pattern
import joblib

# ----------------- BASE DIRECTORY RESOLUTION -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="OmniML | Enterprise Machine Learning Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- HIGH-END ENTERPRISE STYLING & DESIGN SYSTEM -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Typography & Root Settings */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #E2E8F0;
        background-color: #070A12 !important;
        scroll-behavior: smooth;
    }

    /* Ambient Background Mesh */
    .stApp {
        background: radial-gradient(circle at 15% 10%, rgba(37, 99, 235, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 85% 25%, rgba(139, 92, 246, 0.07) 0%, transparent 45%),
                    radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%),
                    #070A12 !important;
    }

    code, pre, .font-mono {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }

    /* Smooth Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #090E1A;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Executive Hero Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.03);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 35%, #10B981 70%, #F59E0B 100%);
    }

    .hero-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
        flex-wrap: wrap;
        gap: 0.75rem;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 60%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 1.2rem;
        line-height: 1.55;
    }

    /* Telemetry HUD & Status Pills */
    .telemetry-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .telemetry-pill {
        background: rgba(30, 41, 59, 0.7);
        color: #94A3B8;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.35rem 0.85rem;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        transition: all 0.2s ease;
    }
    .telemetry-pill:hover {
        border-color: rgba(255, 255, 255, 0.18);
        background: rgba(30, 41, 59, 0.95);
    }

    .telemetry-pill.active {
        background: rgba(59, 130, 246, 0.14);
        color: #60A5FA;
        border-color: rgba(59, 130, 246, 0.35);
    }

    .telemetry-pill.success {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.35);
    }

    .telemetry-pill.purple {
        background: rgba(139, 92, 246, 0.14);
        color: #A78BFA;
        border-color: rgba(139, 92, 246, 0.35);
    }

    .telemetry-pill.amber {
        background: rgba(245, 158, 11, 0.14);
        color: #FBBF24;
        border-color: rgba(245, 158, 11, 0.35);
    }

    /* Live Pulse Indicator */
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-glow 2s infinite;
        display: inline-block;
    }
    @keyframes pulse-glow {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Sticky Pipeline Navigation Track */
    .pipeline-track {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.85rem 1.4rem;
        margin-bottom: 2.25rem;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
        overflow-x: auto;
        gap: 0.5rem;
    }

    .track-step-link {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.03em;
        text-decoration: none;
        white-space: nowrap;
        transition: color 0.15s ease;
    }
    .track-step-link:hover {
        color: #F8FAFC;
    }
    .track-step-badge {
        background: rgba(30, 41, 59, 0.9);
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .track-divider {
        color: #334155;
        font-size: 0.85rem;
        user-select: none;
    }

    /* Sequential Step Section Card */
    .step-section {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(11, 17, 32, 0.92) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem 2.25rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .step-section:hover {
        border-color: rgba(255, 255, 255, 0.14);
        box-shadow: 0 12px 36px -4px rgba(0, 0, 0, 0.65);
    }

    /* Top Accent Line per step */
    .step-accent-01::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .step-accent-02::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #10B981, #34D399); }
    .step-accent-03::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #8B5CF6, #C084FC); }
    .step-accent-04::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .step-accent-05::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #F43F5E, #FB7185); }
    .step-accent-06::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #06B6D4, #38BDF8); }

    .step-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        padding-bottom: 1.1rem;
        margin-bottom: 1.6rem;
        flex-wrap: wrap;
        gap: 0.75rem;
    }

    .step-header-left {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .step-number-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 800;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        letter-spacing: 0.08em;
    }

    .badge-01 { background: linear-gradient(135deg, #2563EB, #1D4ED8); color: #FFF; box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35); }
    .badge-02 { background: linear-gradient(135deg, #059669, #047857); color: #FFF; box-shadow: 0 2px 10px rgba(5, 150, 105, 0.35); }
    .badge-03 { background: linear-gradient(135deg, #7C3AED, #6D28D9); color: #FFF; box-shadow: 0 2px 10px rgba(124, 58, 237, 0.35); }
    .badge-04 { background: linear-gradient(135deg, #D97706, #B45309); color: #FFF; box-shadow: 0 2px 10px rgba(217, 119, 6, 0.35); }
    .badge-05 { background: linear-gradient(135deg, #E11D48, #BE123C); color: #FFF; box-shadow: 0 2px 10px rgba(225, 29, 72, 0.35); }
    .badge-06 { background: linear-gradient(135deg, #0891B2, #0E7490); color: #FFF; box-shadow: 0 2px 10px rgba(8, 145, 178, 0.35); }

    .step-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .step-header-desc {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Metric KPI Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.3rem 1.1rem;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
    }
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-3px);
        box-shadow: 0 10px 24px -4px rgba(37, 99, 235, 0.25);
    }
    .metric-val {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-lbl {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-top: 0.45rem;
        font-weight: 600;
    }

    /* Inner Cards & Containers */
    .inner-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.1rem;
    }

    /* Visual Pipeline Transformation Frame */
    .visual-frame {
        background: #090E1A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.85rem 0.6rem;
        text-align: center;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .visual-frame:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .visual-frame-caption {
        font-size: 0.82rem;
        font-weight: 600;
        color: #CBD5E1;
        margin-top: 0.5rem;
    }
    .visual-frame-meta {
        font-size: 0.72rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.15rem;
    }

    /* Cyber Console / Terminal Window */
    .terminal-container {
        background: #050810;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        margin-top: 0.75rem;
    }
    .terminal-topbar {
        background: #0F172A;
        padding: 0.45rem 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .term-btn {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .term-btn.red { background: #EF4444; }
    .term-btn.yellow { background: #F59E0B; }
    .term-btn.green { background: #10B981; }
    .term-title {
        color: #64748B;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 0.5rem;
        font-weight: 500;
    }
    .terminal-window {
        background: #050810;
        padding: 1.25rem;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        line-height: 1.65;
        white-space: pre-wrap;
        max-height: 420px;
        overflow-y: auto;
    }

    /* Sidebar Refinements */
    [data-testid="stSidebar"] {
        background-color: #070B14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Primary & Secondary Buttons */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: 1px solid #3B82F6 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.35rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    }

    /* Section Subheadings */
    .subhead-label {
        font-size: 0.98rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .subhead-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# DATASET GENERATION & PREPROCESSING ENGINES
# =====================================================================

@st.cache_data
def get_digits_dataset():
    """Loads standard 8x8 digits benchmark dataset and scales to uint8."""
    digits = load_digits()
    images_uint8 = ((digits.images / 16.0) * 255.0).astype(np.uint8)
    return images_uint8, digits.target, [str(i) for i in range(10)]

@st.cache_data
def get_synthetic_shapes(samples_per_class=80):
    """Generates synthetic multi-class geometric shape dataset."""
    np.random.seed(42)
    images = []
    labels = []
    class_names = ["Circle", "Square", "Triangle", "Cross"]
    img_size = (28, 28)

    for class_idx in range(4):
        for _ in range(samples_per_class):
            img = np.zeros(img_size, dtype=np.uint8)
            center_x = np.random.randint(9, 19)
            center_y = np.random.randint(9, 19)
            size = np.random.randint(6, 9)

            if class_idx == 0:  # Circle
                cv2.circle(img, (center_x, center_y), size, 255, -1)
            elif class_idx == 1:  # Square
                cv2.rectangle(img, (center_x - size, center_y - size), (center_x + size, center_y + size), 255, -1)
            elif class_idx == 2:  # Triangle
                pts = np.array([(center_x, center_y - size), (center_x - size, center_y + size), (center_x + size, center_y + size)], np.int32)
                cv2.fillPoly(img, [pts], 255)
            elif class_idx == 3:  # Cross
                th = np.random.randint(2, 4)
                cv2.line(img, (center_x - size, center_y), (center_x + size, center_y), 255, th)
                cv2.line(img, (center_x, center_y - size), (center_x, center_y + size), 255, th)

            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((img_size[0] // 2, img_size[1] // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, img_size)
            noise = np.random.normal(0, 8, img_size).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            images.append(img)
            labels.append(class_idx)

    return np.array(images), np.array(labels), class_names


def preprocess_single_image(image, denoise_method="gaussian", kernel_size=3, thresh_method="otsu", morph_op="none"):
    """Full preprocessing transformation filter pipeline with strict uint8 enforcement."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = (image * 255.0).astype(np.uint8)
        elif image.max() <= 16.0:
            img_uint8 = ((image / 16.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image.copy()

    # 1. Grayscale Conversion
    if len(img_uint8.shape) == 3:
        if img_uint8.shape[2] == 4:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8.copy()

    # Auto Invert if background is light (standardize to dark background / bright strokes)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # 2. Resizing to standard 28x28 matrix
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # 3. Noise Filtering / Denoising
    k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    if denoise_method == "gaussian":
        denoised = cv2.GaussianBlur(resized, (k, k), 0)
    elif denoise_method == "median":
        denoised = cv2.medianBlur(resized, k)
    elif denoise_method == "bilateral":
        denoised = cv2.bilateralFilter(resized, d=k * 2, sigmaColor=75, sigmaSpace=75)
    else:
        denoised = resized.copy()

    if denoised.dtype != np.uint8:
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    # 4. Thresholding / Binarization
    if thresh_method == "otsu":
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_method == "adaptive":
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif thresh_method == "binary":
        _, thresh = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
    else:
        thresh = denoised.copy()

    if thresh.dtype != np.uint8:
        thresh = np.clip(thresh, 0, 255).astype(np.uint8)

    # 5. Morphological Structuring
    if morph_op == "dilate":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.dilate(thresh, kernel, iterations=1)
    elif morph_op == "erode":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.erode(thresh, kernel, iterations=1)
    elif morph_op == "open":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    elif morph_op == "close":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        morphed = thresh.copy()

    # 6. Normalization [0.0, 1.0]
    normalized = morphed.astype(np.float32) / 255.0

    return {
        "original": image,
        "gray": gray,
        "resized": resized,
        "denoised": denoised,
        "thresh": thresh,
        "morphed": morphed,
        "normalized": normalized
    }


def extract_features_from_image(image_28, feature_type="HOG", orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """Computes multidimensional feature representations and spatial visual descriptors."""
    img_uint8 = (image_28 * 255).astype(np.uint8) if image_28.max() <= 1.0 else image_28.astype(np.uint8)
    if img_uint8.shape != (28, 28):
        img_uint8 = cv2.resize(img_uint8, (28, 28))

    if feature_type == "HOG":
        fd, hog_vis = hog(
            img_uint8,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            feature_vector=True
        )
        return fd, hog_vis

    elif feature_type == "Raw Pixels":
        norm = img_uint8.astype(np.float32) / 255.0
        return norm.flatten(), norm

    elif feature_type == "Intensity Histogram":
        hist = cv2.calcHist([img_uint8], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist, None

    elif feature_type == "Edges (Canny)":
        edges = cv2.Canny(img_uint8, 50, 150)
        norm_edges = edges.astype(np.float32) / 255.0
        return norm_edges.flatten(), edges

    elif feature_type == "LBP Texture":
        lbp = local_binary_pattern(img_uint8, P=8, R=1, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist, lbp

    elif feature_type == "Hybrid (HOG + LBP + Edges)":
        fd, hog_vis = hog(img_uint8, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
        lbp = local_binary_pattern(img_uint8, P=8, R=1, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        edges = cv2.Canny(img_uint8, 50, 150)
        edge_stats = np.array([np.mean(edges) / 255.0, np.std(edges) / 255.0])
        combined = np.hstack([fd, hist_lbp, edge_stats])
        return combined, hog_vis

    return img_uint8.flatten() / 255.0, None


# ----------------- SESSION STATE INITIALIZATION -----------------
if "dataset_choice" not in st.session_state:
    st.session_state["dataset_choice"] = "Handwritten Digits (MNIST Subset)"
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
if "active_model_name" not in st.session_state:
    st.session_state["active_model_name"] = "Not Trained"
if "model_metrics" not in st.session_state:
    st.session_state["model_metrics"] = None
if "X_test" not in st.session_state:
    st.session_state["X_test"] = None
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None
if "class_names" not in st.session_state:
    st.session_state["class_names"] = [str(i) for i in range(10)]
if "active_feature_type" not in st.session_state:
    st.session_state["active_feature_type"] = "HOG"


# =====================================================================
# SIDEBAR: SYSTEM PARAMETER CONTROLS
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 0.5rem; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">SYSTEM PARAMETERS</div>
        <div style="font-size: 0.75rem; color: #64748B; font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem;">CONTROL & CONFIG ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    dataset_option = st.selectbox(
        "Active Ingestion Source",
        ["Handwritten Digits (MNIST Subset)", "Synthetic Geometric Shapes", "Custom Upload / Canvas Drawing"],
        index=0,
        key="sb_dataset_select"
    )
    st.session_state["dataset_choice"] = dataset_option

    st.markdown("---")
    st.markdown("##### Preprocessing Configuration")
    sb_denoise = st.selectbox("Noise Attenuation", ["gaussian", "median", "bilateral", "none"], index=0, key="sb_denoise_select")
    sb_kernel = st.slider("Filter Kernel Size", 1, 9, 3, step=2, key="sb_kernel_slider")
    sb_thresh = st.selectbox("Binarization Mode", ["otsu", "adaptive", "binary", "none"], index=0, key="sb_thresh_select")
    sb_morph = st.selectbox("Morphological Filter", ["none", "dilate", "erode", "open", "close"], index=0, key="sb_morph_select")

    st.markdown("---")
    st.markdown("##### Feature Representation")
    sb_feature_type = st.selectbox(
        "Feature Extractor",
        ["HOG", "Raw Pixels", "Intensity Histogram", "Edges (Canny)", "LBP Texture", "Hybrid (HOG + LBP + Edges)"],
        index=0,
        key="sb_feat_select"
    )
    st.session_state["active_feature_type"] = sb_feature_type

    if "HOG" in sb_feature_type:
        sb_orientations = st.slider("Gradient Orientations", 6, 12, 9, key="sb_hog_ori")
        ppc_option = st.selectbox("Cell Granularity", ["4x4", "2x2", "8x8"], index=0, key="sb_hog_ppc")
        ppc_dict = {"2x2": (2, 2), "4x4": (4, 4), "8x8": (8, 8)}
        sb_ppc = ppc_dict[ppc_option]
    else:
        sb_orientations = 9
        sb_ppc = (4, 4)

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(15,23,42,0.6); padding: 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); font-size: 0.75rem; color: #64748B;">
        <div><strong>Runtime:</strong> Python 3.14 (x64)</div>
        <div><strong>Framework:</strong> Scikit-Learn / OpenCV</div>
        <div><strong>Status:</strong> Engine Operational</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------- LOAD GLOBAL ACTIVE DATASET -----------------
if st.session_state["dataset_choice"] == "Handwritten Digits (MNIST Subset)":
    raw_images, raw_labels, class_names = get_digits_dataset()
elif st.session_state["dataset_choice"] == "Synthetic Geometric Shapes":
    raw_images, raw_labels, class_names = get_synthetic_shapes()
else:
    raw_images, raw_labels, class_names = get_digits_dataset()

st.session_state["class_names"] = class_names
sample_id_default = 0
target_raw_img = raw_images[sample_id_default]
prep_results_default = preprocess_single_image(
    target_raw_img,
    denoise_method=sb_denoise,
    kernel_size=sb_kernel,
    thresh_method=sb_thresh,
    morph_op=sb_morph
)


# =====================================================================
# EXECUTIVE HERO HEADER & REAL-TIME TELEMETRY BAR
# =====================================================================
active_model_disp = st.session_state.get('active_model_name', 'Not Trained')
acc_disp = "N/A"
if st.session_state["model_metrics"] is not None:
    acc_disp = f"{st.session_state['model_metrics']['accuracy']*100:.2f}%"

st.markdown(f"""
<div class="hero-header">
    <div class="hero-title-row">
        <h1 class="hero-title">OMNIML | ENTERPRISE MACHINE LEARNING PLATFORM</h1>
        <span class="telemetry-pill success"><span class="pulse-dot"></span> SYSTEM ONLINE</span>
    </div>
    <div class="hero-subtitle">
        High-Performance Visual AI Pipeline: Top-to-Bottom Sequential Execution Architecture with Full Metric Telemetry
    </div>
    <div class="telemetry-row">
        <span class="telemetry-pill active">SOURCE: {st.session_state['dataset_choice']}</span>
        <span class="telemetry-pill">SAMPLES: {len(raw_images):,}</span>
        <span class="telemetry-pill">CLASSES: {len(class_names)}</span>
        <span class="telemetry-pill purple">DESCRIPTOR: {sb_feature_type}</span>
        <span class="telemetry-pill amber">MODEL: {active_model_disp}</span>
        <span class="telemetry-pill success">ACCURACY: {acc_disp}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sequential Workflow Progress Bar / Sticky Track
st.markdown("""
<div class="pipeline-track">
    <a href="#step-01-data-ingestion-dataset-management" class="track-step-link"><span class="track-step-badge">01</span> INGESTION</a>
    <span class="track-divider">→</span>
    <a href="#step-02-image-preprocessing-spatial-transformations" class="track-step-link"><span class="track-step-badge">02</span> PREPROCESSING</a>
    <span class="track-divider">→</span>
    <a href="#step-03-feature-engineering-dimensionality-projection" class="track-step-link"><span class="track-step-badge">03</span> FEATURES</a>
    <span class="track-divider">→</span>
    <a href="#step-04-machine-learning-classifier-training" class="track-step-link"><span class="track-step-badge">04</span> TRAINING</a>
    <span class="track-divider">→</span>
    <a href="#step-05-performance-evaluation-real-time-inference" class="track-step-link"><span class="track-step-badge">05</span> EVALUATION</a>
    <span class="track-divider">→</span>
    <a href="#step-06-code-studio-cli-compilation-manual" class="track-step-link"><span class="track-step-badge">06</span> CODE STUDIO</a>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# STEP 01: DATA INGESTION & DATASET MANAGEMENT
# =====================================================================
st.markdown("""
<div class="step-section step-accent-01">
    <div class="step-header-bar">
        <div class="step-header-left">
            <span class="step-number-badge badge-01">STEP 01</span>
            <h2 class="step-header-title">Data Ingestion & Dataset Management</h2>
        </div>
        <span class="step-header-desc">Ingestion & Partitioning Module</span>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.15, 1.35], gap="large")

with col1:
    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Dataset Acquisition & Partitioning Configurator</div>""", unsafe_allow_html=True)
    st.info(f"Active Ingestion Source: **{st.session_state['dataset_choice']}**")

    if st.session_state["dataset_choice"] == "Handwritten Digits (MNIST Subset)":
        st.success(f"[SUCCESS] Ingested benchmark digits: {len(raw_images):,} samples across 10 classes (0-9).")

    elif st.session_state["dataset_choice"] == "Synthetic Geometric Shapes":
        st.success(f"[SUCCESS] Ingested synthetic shapes: {len(raw_images):,} samples across 4 classes ({', '.join(class_names)}).")

    else:
        st.markdown("##### Custom Input Ingestion")
        sketch_tab, upload_tab = st.tabs(["Canvas Sketch", "File Upload"])

        with sketch_tab:
            if CANVAS_AVAILABLE and st_canvas is not None:
                canvas_draw = st_canvas(
                    fill_color="#000000",
                    stroke_width=18,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=240,
                    width=240,
                    drawing_mode="freedraw",
                    update_streamlit=True,
                    return_image_data=True,
                    key="step1_canvas"
                )
                try:
                    if canvas_draw is not None and getattr(canvas_draw, "image_data", None) is not None:
                        if not np.all(canvas_draw.image_data[:, :, :3] == 0):
                            st.caption("[INFO] Pattern strokes detected. Ready for Preprocessing.")
                except Exception:
                    pass
            else:
                st.info("[INFO] Canvas component optional. Use the File Upload tab or select a standard dataset.")

        with upload_tab:
            uploaded_file = st.file_uploader("Select PNG or JPG scan", type=["png", "jpg", "jpeg"], key="step1_upload")
            if uploaded_file is not None:
                bytes_arr = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                custom_img = cv2.imdecode(bytes_arr, cv2.IMREAD_UNCHANGED)
                st.image(custom_img, caption="Ingested Document Scan", width=180)

    # Dataset Split Configurator
    st.markdown("""<div class="subhead-label" style="margin-top: 1.25rem;"><span class="subhead-dot"></span> Partitioning & Stratification</div>""", unsafe_allow_html=True)
    test_ratio = st.slider("Holdout Validation Ratio (%)", min_value=10, max_value=40, value=25, step=5, key="split_slider")
    rand_seed = st.number_input("Random Initialization Seed", value=42, step=1, key="seed_input")

    total_samples = len(raw_images)
    test_samples = int(total_samples * (test_ratio / 100.0))
    train_samples = total_samples - test_samples

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""<div class="metric-card"><div class="metric-val">{total_samples}</div><div class="metric-lbl">Total Samples</div></div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #60A5FA;">{train_samples}</div><div class="metric-lbl">Train ({(100-test_ratio)}%)</div></div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #34D399;">{test_samples}</div><div class="metric-lbl">Holdout ({test_ratio}%)</div></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Class Representation Balance & Ingestion Analytics</div>""", unsafe_allow_html=True)
    df_dist = pd.DataFrame({
        "Class": [st.session_state["class_names"][i] for i in raw_labels],
    })
    counts = df_dist["Class"].value_counts().reset_index()
    counts.columns = ["Class Label", "Sample Count"]

    fig_dist = px.bar(
        counts,
        x="Class Label",
        y="Sample Count",
        color="Sample Count",
        color_continuous_scale=[[0, "#1E3A8A"], [0.5, "#2563EB"], [1, "#38BDF8"]],
        title="Class Balance Distribution"
    )
    fig_dist.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace")
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Sample Verification Gallery</div>""", unsafe_allow_html=True)
    gallery_cols = st.columns(5)
    for i in range(min(10, len(raw_images))):
        col_idx = i % 5
        with gallery_cols[col_idx]:
            sample_img = raw_images[i]
            sample_disp = sample_img if sample_img.dtype == np.uint8 else (sample_img * 255).astype(np.uint8)
            st.image(sample_disp, caption=f"Class {st.session_state['class_names'][raw_labels[i]]}", width=65)

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 02: PREPROCESSING & SPATIAL TRANSFORMATIONS
# =====================================================================
st.markdown("""
<div class="step-section step-accent-02">
    <div class="step-header-bar">
        <div class="step-header-left">
            <span class="step-number-badge badge-02">STEP 02</span>
            <h2 class="step-header-title">Image Preprocessing & Spatial Transformations</h2>
        </div>
        <span class="step-header-desc">Transformation & Denoising Engine</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("Execute full multi-stage spatial transformations including matrix normalization, noise attenuation, morphological filtering, and spatial binarization.")

sample_id = st.slider("Select Sample Index for Dynamic Inspection", 0, len(raw_images) - 1, 0, key="sample_id_slider")
target_raw_img = raw_images[sample_id]

prep_results = preprocess_single_image(
    target_raw_img,
    denoise_method=sb_denoise,
    kernel_size=sb_kernel,
    thresh_method=sb_thresh,
    morph_op=sb_morph
)

st.markdown("""<div class="subhead-label" style="margin-top: 1rem;"><span class="subhead-dot"></span> 6-Stage Spatial Transformation Pipeline</div>""", unsafe_allow_html=True)
step_c1, step_c2, step_c3, step_c4, step_c5, step_c6 = st.columns(6)

with step_c1:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(prep_results["original"], width=105, clamp=True)
    st.markdown(f"""<div class="visual-frame-caption">1. Raw Input</div><div class="visual-frame-meta">{prep_results['original'].shape}</div></div>""", unsafe_allow_html=True)

with step_c2:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(prep_results["gray"], width=105, clamp=True)
    st.markdown(f"""<div class="visual-frame-caption">2. Grayscale</div><div class="visual-frame-meta">Mean: {np.mean(prep_results['gray']):.1f}</div></div>""", unsafe_allow_html=True)

with step_c3:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(prep_results["denoised"], width=105, clamp=True)
    st.markdown(f"""<div class="visual-frame-caption">3. Denoised</div><div class="visual-frame-meta">{sb_denoise} ({sb_kernel}x{sb_kernel})</div></div>""", unsafe_allow_html=True)

with step_c4:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(prep_results["thresh"], width=105, clamp=True)
    st.markdown(f"""<div class="visual-frame-caption">4. Threshold</div><div class="visual-frame-meta">{sb_thresh}</div></div>""", unsafe_allow_html=True)

with step_c5:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(prep_results["morphed"], width=105, clamp=True)
    st.markdown(f"""<div class="visual-frame-caption">5. Morphology</div><div class="visual-frame-meta">{sb_morph}</div></div>""", unsafe_allow_html=True)

with step_c6:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(prep_results["normalized"], width=105, clamp=True)
    st.markdown(f"""<div class="visual-frame-caption">6. Standardized</div><div class="visual-frame-meta">[0.0, 1.0] (28x28)</div></div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 1.5rem 0;'>", unsafe_allow_html=True)
st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Stochastic Data Augmentation Bench</div>""", unsafe_allow_html=True)

aug_c1, aug_c2, aug_c3, aug_c4 = st.columns(4)

rot_angle = 18
h, w = prep_results["morphed"].shape
M_rot = cv2.getRotationMatrix2D((w//2, h//2), rot_angle, 1.0)
rot_img = cv2.warpAffine(prep_results["morphed"], M_rot, (w, h))

flip_img = cv2.flip(prep_results["morphed"], 1)

M_zoom = cv2.getRotationMatrix2D((w//2, h//2), 0, 1.25)
zoom_img = cv2.warpAffine(prep_results["morphed"], M_zoom, (w, h))

noise_matrix = np.random.normal(0, 20, (w, h)).astype(np.int16)
noisy_img = np.clip(prep_results["morphed"].astype(np.int16) + noise_matrix, 0, 255).astype(np.uint8)

with aug_c1:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(rot_img, width=115)
    st.markdown("""<div class="visual-frame-caption">Affine Rotation (+18 deg)</div></div>""", unsafe_allow_html=True)
with aug_c2:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(flip_img, width=115)
    st.markdown("""<div class="visual-frame-caption">Horizontal Reflection</div></div>""", unsafe_allow_html=True)
with aug_c3:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(zoom_img, width=115)
    st.markdown("""<div class="visual-frame-caption">Geometric Zoom (1.25x)</div></div>""", unsafe_allow_html=True)
with aug_c4:
    st.markdown("""<div class="visual-frame">""", unsafe_allow_html=True)
    st.image(noisy_img, width=115)
    st.markdown("""<div class="visual-frame-caption">Gaussian Jitter Injection</div></div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 03: FEATURE EXTRACTION & DIMENSIONALITY REDUCTION
# =====================================================================
st.markdown("""
<div class="step-section step-accent-03">
    <div class="step-header-bar">
        <div class="step-header-left">
            <span class="step-number-badge badge-03">STEP 03</span>
            <h2 class="step-header-title">Feature Engineering & Dimensionality Projection</h2>
        </div>
        <span class="step-header-desc">Feature Representation Suite</span>
    </div>
""", unsafe_allow_html=True)

feat_left, feat_right = st.columns([1.15, 1.35], gap="large")

with feat_left:
    st.markdown(f"""<div class="subhead-label"><span class="subhead-dot"></span> Feature Space Vector (Target Class: <code>{st.session_state['class_names'][raw_labels[sample_id]]}</code>)</div>""", unsafe_allow_html=True)

    feature_vector, feat_vis = extract_features_from_image(
        prep_results["morphed"],
        feature_type=sb_feature_type,
        orientations=sb_orientations,
        pixels_per_cell=sb_ppc
    )

    st.markdown(f"""
    <div class="inner-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
            <span style="color: #94A3B8;">Active Representation:</span>
            <span style="color: #38BDF8; font-weight: 700;">{sb_feature_type}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
            <span style="color: #94A3B8;">Vector Dimensionality:</span>
            <span style="color: #34D399; font-family: 'JetBrains Mono', monospace; font-weight: 700;">{len(feature_vector)} features</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">Min / Max / Mean:</span>
            <span style="color: #CBD5E1; font-family: 'JetBrains Mono', monospace;">{np.min(feature_vector):.3f} / {np.max(feature_vector):.3f} / {np.mean(feature_vector):.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    vis_c1, vis_c2 = st.columns(2)
    with vis_c1:
        st.caption("Standardized Pattern (28x28)")
        st.image(prep_results["morphed"], width=140)
    with vis_c2:
        if feat_vis is not None:
            st.caption(f"{sb_feature_type} Gradient Map")
            st.image(feat_vis, width=140, clamp=True)
        else:
            st.caption("Feature Magnitude Spectrum")
            fig_hist = px.bar(x=list(range(len(feature_vector))), y=feature_vector)
            fig_hist.update_layout(
                template="plotly_dark",
                height=140,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

with feat_right:
    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> 3D Principal Component Analysis (Eigenspace Projection)</div>""", unsafe_allow_html=True)
    st.caption("Visualizing high-dimensional feature vectors mapped into 3D eigenspace to evaluate class manifold separability.")

    with st.spinner("Computing 3D PCA projection..."):
        subset_size = min(300, len(raw_images))
        sub_feats = []
        for i in range(subset_size):
            f_vec, _ = extract_features_from_image(
                raw_images[i],
                feature_type=sb_feature_type,
                orientations=sb_orientations,
                pixels_per_cell=sb_ppc
            )
            sub_feats.append(f_vec)

        pca_3d = PCA(n_components=3, random_state=42)
        pca_coords = pca_3d.fit_transform(np.array(sub_feats))
        exp_var = pca_3d.explained_variance_ratio_

        df_pca = pd.DataFrame({
            "PC1": pca_coords[:, 0],
            "PC2": pca_coords[:, 1],
            "PC3": pca_coords[:, 2],
            "Class": [st.session_state["class_names"][raw_labels[i]] for i in range(subset_size)]
        })

        fig_pca = px.scatter_3d(
            df_pca,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Class",
            title=f"3D PCA Manifold ({sb_feature_type} - {subset_size} samples | Expl. Var: {np.sum(exp_var)*100:.1f}%)",
            opacity=0.88,
            height=340
        )
        fig_pca.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1),
            font=dict(family="JetBrains Mono, monospace")
        )
        st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 04: MODEL TRAINING & HYPERPARAMETER TUNING
# =====================================================================
st.markdown("""
<div class="step-section step-accent-04">
    <div class="step-header-bar">
        <div class="step-header-left">
            <span class="step-number-badge badge-04">STEP 04</span>
            <h2 class="step-header-title">Machine Learning Classifier Training</h2>
        </div>
        <span class="step-header-desc">Model Fitting Engine</span>
    </div>
""", unsafe_allow_html=True)

col_algo, col_train_results = st.columns([1.15, 1.35], gap="large")

with col_algo:
    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Classifier Architecture & Hyperparameters</div>""", unsafe_allow_html=True)

    selected_algo = st.selectbox(
        "Classification Algorithm",
        [
            "Support Vector Machine (SVM)",
            "Random Forest Classifier",
            "K-Nearest Neighbors (KNN)",
            "Multi-Layer Perceptron (MLP Neural Network)",
            "Logistic Regression",
            "Decision Tree"
        ],
        index=0,
        key="algo_select"
    )

    model_params = {}

    if selected_algo == "Support Vector Machine (SVM)":
        c_val = st.slider("Regularization Parameter (C)", 0.1, 20.0, 5.0, step=0.5, key="svm_c")
        kernel_choice = st.selectbox("Kernel Function", ["rbf", "linear", "poly", "sigmoid"], index=0, key="svm_kernel")
        gamma_choice = st.selectbox("Kernel Coefficient (gamma)", ["scale", "auto"], index=0, key="svm_gamma")
        model_params = {"C": c_val, "kernel": kernel_choice, "gamma": gamma_choice, "probability": True, "random_state": 42}

    elif selected_algo == "Random Forest Classifier":
        n_trees = st.slider("Ensemble Estimators (Trees)", 10, 300, 100, step=10, key="rf_trees")
        m_depth = st.slider("Max Tree Depth", 2, 30, 15, key="rf_depth")
        criterion = st.selectbox("Splitting Criterion", ["gini", "entropy", "log_loss"], index=0, key="rf_crit")
        model_params = {"n_estimators": n_trees, "max_depth": m_depth, "criterion": criterion, "random_state": 42}

    elif selected_algo == "K-Nearest Neighbors (KNN)":
        k_val = st.slider("Number of Neighbors (k)", 1, 15, 5, key="knn_k")
        weights_val = st.selectbox("Distance Weighting Scheme", ["uniform", "distance"], index=1, key="knn_weights")
        metric_val = st.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan"], index=0, key="knn_metric")
        model_params = {"n_neighbors": k_val, "weights": weights_val, "metric": metric_val}

    elif selected_algo == "Multi-Layer Perceptron (MLP Neural Network)":
        layer1 = st.slider("Hidden Layer 1 Neurons", 32, 256, 128, step=32, key="mlp_l1")
        layer2 = st.slider("Hidden Layer 2 Neurons", 0, 128, 64, step=16, key="mlp_l2")
        act_func = st.selectbox("Nonlinear Activation", ["relu", "tanh", "logistic"], index=0, key="mlp_act")
        max_iter = st.slider("Maximum Epochs", 50, 400, 150, step=25, key="mlp_iter")
        hidden_layers = (layer1, layer2) if layer2 > 0 else (layer1,)
        model_params = {"hidden_layer_sizes": hidden_layers, "activation": act_func, "max_iter": max_iter, "random_state": 42}

    elif selected_algo == "Logistic Regression":
        c_val = st.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, key="lr_c")
        max_iter = st.slider("Maximum Solver Iterations", 50, 500, 200, key="lr_iter")
        model_params = {"C": c_val, "max_iter": max_iter, "random_state": 42}

    elif selected_algo == "Decision Tree":
        max_depth = st.slider("Maximum Tree Depth", 2, 25, 10, key="dt_depth")
        criterion = st.selectbox("Purity Metric", ["gini", "entropy"], index=0, key="dt_crit")
        model_params = {"max_depth": max_depth, "criterion": criterion, "random_state": 42}

    train_btn = st.button("Fit & Train Classifier", type="primary", use_container_width=True, key="train_action_btn")

with col_train_results:
    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Training Telemetry & Real-Time Metrics</div>""", unsafe_allow_html=True)

    if train_btn:
        with st.spinner(f"Extracting features ({sb_feature_type}) and fitting {selected_algo}..."):
            feats = []
            for img in raw_images:
                f_vec, _ = extract_features_from_image(img, feature_type=sb_feature_type, orientations=sb_orientations, pixels_per_cell=sb_ppc)
                feats.append(f_vec)

            X_all = np.array(feats)
            y_all = raw_labels

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all,
                test_size=(test_ratio / 100.0),
                random_state=rand_seed,
                stratify=y_all
            )

            start_clock = time.time()
            if selected_algo == "Support Vector Machine (SVM)":
                clf = SVC(**model_params)
            elif selected_algo == "Random Forest Classifier":
                clf = RandomForestClassifier(**model_params)
            elif selected_algo == "K-Nearest Neighbors (KNN)":
                clf = KNeighborsClassifier(**model_params)
            elif selected_algo == "Multi-Layer Perceptron (MLP Neural Network)":
                clf = MLPClassifier(**model_params)
            elif selected_algo == "Logistic Regression":
                clf = LogisticRegression(**model_params)
            elif selected_algo == "Decision Tree":
                clf = DecisionTreeClassifier(**model_params)

            clf.fit(X_tr, y_tr)
            elapsed_time = time.time() - start_clock

            y_pred = clf.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_te, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_te, y_pred)
            rep = classification_report(y_te, y_pred, target_names=st.session_state["class_names"], output_dict=True, zero_division=0)

            prob_matrix = None
            if hasattr(clf, "predict_proba"):
                try:
                    prob_matrix = clf.predict_proba(X_te)
                except Exception:
                    pass

            st.session_state["trained_model"] = clf
            st.session_state["active_model_name"] = selected_algo
            st.session_state["X_test"] = X_te
            st.session_state["y_test"] = y_te
            st.session_state["model_metrics"] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "elapsed_time": elapsed_time,
                "confusion_matrix": cm,
                "report": rep,
                "y_pred": y_pred,
                "prob_matrix": prob_matrix
            }

            model_file_path = os.path.join(BASE_DIR, "digit_mlp_model.pkl")
            joblib.dump(clf, model_file_path)

        st.success(f"[SUCCESS] {selected_algo} trained in {elapsed_time:.4f} seconds with {acc*100:.2f}% holdout accuracy.")

    if st.session_state["model_metrics"] is not None:
        m = st.session_state["model_metrics"]
        st.markdown(f"**Model:** `{st.session_state.get('active_model_name', 'Trained Model')}` | **Duration:** `{m['elapsed_time']:.3f}s`")

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-val">{m['accuracy']*100:.1f}%</div><div class="metric-lbl">Accuracy</div></div>""", unsafe_allow_html=True)
        with kpi_c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #60A5FA;">{m['precision']*100:.1f}%</div><div class="metric-lbl">Precision</div></div>""", unsafe_allow_html=True)
        with kpi_c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #A78BFA;">{m['recall']*100:.1f}%</div><div class="metric-lbl">Recall</div></div>""", unsafe_allow_html=True)
        with kpi_c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color: #34D399;">{m['f1']*100:.1f}%</div><div class="metric-lbl">F1-Score</div></div>""", unsafe_allow_html=True)

        model_file_path = os.path.join(BASE_DIR, "digit_mlp_model.pkl")
        if os.path.exists(model_file_path):
            with open(model_file_path, "rb") as f_model:
                model_bytes = f_model.read()
            st.download_button(
                label="Export Serialized Model Checkpoint (.pkl)",
                data=model_bytes,
                file_name="trained_image_ml_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
    else:
        st.info("[STATUS] Click 'Fit & Train Classifier' to train model.")

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 05: PERFORMANCE EVALUATION & LIVE INFERENCE
# =====================================================================
st.markdown("""
<div class="step-section step-accent-05">
    <div class="step-header-bar">
        <div class="step-header-left">
            <span class="step-number-badge badge-05">STEP 05</span>
            <h2 class="step-header-title">Performance Evaluation & Real-Time Inference</h2>
        </div>
        <span class="step-header-desc">Telemetry & Testing Bench</span>
    </div>
""", unsafe_allow_html=True)

if st.session_state["model_metrics"] is None:
    st.warning("No trained model in memory. Please execute Step 04 (Classifier Training) above.")
else:
    m = st.session_state["model_metrics"]
    class_names = st.session_state["class_names"]

    eval_col1, eval_col2 = st.columns([1.2, 1.2], gap="large")

    with eval_col1:
        st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Confusion Matrix</div>""", unsafe_allow_html=True)
        cm = m["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=class_names,
            y=class_names,
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
            color_continuous_scale=[[0, "#0F172A"], [0.5, "#1E3A8A"], [1, "#38BDF8"]]
        )
        fig_cm.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with eval_col2:
        st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Per-Class Performance Breakdown</div>""", unsafe_allow_html=True)
        rep = m["report"]
        per_class_data = []
        for c in class_names:
            if c in rep:
                per_class_data.append({
                    "Class": c,
                    "Precision": rep[c]["precision"],
                    "Recall": rep[c]["recall"],
                    "F1-Score": rep[c]["f1-score"]
                })
        df_rep = pd.DataFrame(per_class_data)
        fig_rep = px.bar(
            df_rep,
            x="Class",
            y=["Precision", "Recall", "F1-Score"],
            barmode="group",
            color_discrete_sequence=["#38BDF8", "#A78BFA", "#34D399"]
        )
        fig_rep.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            font=dict(family="JetBrains Mono, monospace")
        )
        st.plotly_chart(fig_rep, use_container_width=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Real-Time Single Image Inference Bench</div>""", unsafe_allow_html=True)

    test_left, test_right = st.columns([1.15, 1.35], gap="large")

    with test_left:
        st.markdown("###### Test Input Acquisition")
        test_mode = st.radio("Acquisition Mode", ["Canvas Sketch", "File Upload"], horizontal=True, key="step5_input_mode")

        eval_target_img = None

        if test_mode == "Canvas Sketch":
            if CANVAS_AVAILABLE and st_canvas is not None:
                canvas_test = st_canvas(
                    fill_color="#000000",
                    stroke_width=22,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=240,
                    width=240,
                    drawing_mode="freedraw",
                    update_streamlit=True,
                    return_image_data=True,
                    key="step5_canvas"
                )
                try:
                    if canvas_test is not None and getattr(canvas_test, "image_data", None) is not None:
                        if not np.all(canvas_test.image_data[:, :, :3] == 0):
                            eval_target_img = canvas_test.image_data.astype("uint8")
                except Exception:
                    pass
            else:
                st.info("Interactive Canvas optional. Switch to 'File Upload' to evaluate custom test scans.")
        else:
            up_test = st.file_uploader("Select Test Scan Image", type=["png", "jpg", "jpeg"], key="step5_upload")
            if up_test is not None:
                bytes_arr = np.asarray(bytearray(up_test.read()), dtype=np.uint8)
                eval_target_img = cv2.imdecode(bytes_arr, cv2.IMREAD_UNCHANGED)
                st.image(eval_target_img, caption="Loaded Inference Test Image", width=140)

        run_test_btn = st.button("Execute Inference", type="primary", use_container_width=True, key="run_inf_btn")

    with test_right:
        st.markdown("###### Classification Telemetry & Posterior Distribution")

        if run_test_btn and eval_target_img is not None:
            test_prep = preprocess_single_image(eval_target_img, denoise_method=sb_denoise, kernel_size=sb_kernel, thresh_method=sb_thresh, morph_op=sb_morph)
            test_feat, test_vis = extract_features_from_image(test_prep["morphed"], feature_type=sb_feature_type, orientations=sb_orientations, pixels_per_cell=sb_ppc)

            clf = st.session_state["trained_model"]
            pred_label = clf.predict([test_feat])[0]
            pred_class_name = st.session_state["class_names"][pred_label]

            prob_dist = None
            conf = 95.0
            if hasattr(clf, "predict_proba"):
                try:
                    prob_dist = clf.predict_proba([test_feat])[0]
                    conf = float(prob_dist[pred_label] * 100)
                except Exception:
                    pass

            k_col1, k_col2 = st.columns(2)
            with k_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{pred_class_name}</div>
                    <div class="metric-lbl">Predicted Target</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: {'#34D399' if conf > 75 else '#FBBF24'};">{conf:.1f}%</div>
                    <div class="metric-lbl">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)

            if prob_dist is not None:
                st.markdown("###### Probability Spectrum (All Classes)")
                df_prob = pd.DataFrame({
                    "Class": st.session_state["class_names"],
                    "Probability (%)": [p * 100 for p in prob_dist]
                })
                fig_prob = px.bar(
                    df_prob,
                    x="Class",
                    y="Probability (%)",
                    color="Probability (%)",
                    color_continuous_scale=[[0, "#1E3A8A"], [0.5, "#2563EB"], [1, "#38BDF8"]]
                )
                fig_prob.update_layout(
                    template="plotly_dark",
                    height=200,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace")
                )
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("[STATUS] Input a pattern or upload an image on the left, then click 'Execute Inference'.")

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 06: CODE STUDIO & CLI EXECUTION MANUAL
# =====================================================================
st.markdown("""
<div class="step-section step-accent-06">
    <div class="step-header-bar">
        <div class="step-header-left">
            <span class="step-number-badge badge-06">STEP 06</span>
            <h2 class="step-header-title">Code Studio & CLI Compilation Manual</h2>
        </div>
        <span class="step-header-desc">Sandbox & Execution Documentation</span>
    </div>
""", unsafe_allow_html=True)

code_subtab1, code_subtab2 = st.tabs(["Interactive Python Sandbox", "Compilation & Execution Manual"])

with code_subtab1:
    st.markdown("Edit or compose Python machine learning routines below and click **'Execute Script'** for instant server execution.")

    code_template_choice = st.selectbox(
        "Select Pipeline Template",
        [
            "1. Complete 5-Stage Machine Learning Pipeline (Ingestion -> Preprocessing -> Features -> Training -> Evaluation)",
            "2. OpenCV Image Preprocessing & Spatial Filters",
            "3. HOG & LBP Feature Extraction Comparison",
            "4. Multi-Algorithm Benchmark Suite (SVM vs Random Forest vs KNN)",
            "5. Confusion Matrix & Detailed Metrics Report Generator"
        ],
        key="step6_template_select"
    )

    if "1. Complete 5-Stage" in code_template_choice:
        sample_code = """# Complete 5-Stage Machine Learning Pipeline
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

# Stage 1: Load Input Dataset
print("[STAGE 1] Loading Digits Dataset...")
digits = load_digits()
X_images, y = digits.images, digits.target
print(f" -> Loaded {len(X_images)} image samples across {len(np.unique(y))} classes.")

# Stage 2 & 3: Preprocessing & HOG Feature Extraction
print("[STAGE 2 & 3] Preprocessing & Extracting HOG Descriptors...")
features = [hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)) for img in X_images]
X = np.array(features)

# Holdout Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Stage 4: Model Training
print("[STAGE 4] Training Support Vector Machine (SVM)...")
model = SVC(C=5.0, kernel='rbf', gamma='scale')
model.fit(X_train, y_train)

# Stage 5: Performance Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[STAGE 5] Evaluation Complete.")
print(f" -> Holdout Accuracy: {acc * 100:.2f}%")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
"""
    elif "2. OpenCV Image Preprocessing" in code_template_choice:
        sample_code = """# Image Preprocessing Benchmark
import numpy as np
import cv2
from sklearn.datasets import load_digits

digits = load_digits()
sample = ((digits.images[0] / 16.0) * 255.0).astype(np.uint8)

print("Original Image Shape:", sample.shape)
print("Original Pixel Range: [", np.min(sample), ",", np.max(sample), "]")

# Resizing to standard matrix
resized = cv2.resize(sample, (28, 28), interpolation=cv2.INTER_AREA)

# Normalization
normalized = resized.astype(np.float32) / 255.0
print("Normalized Pixel Range: [", np.min(normalized), ",", np.max(normalized), "]")

# Gaussian Blur
blurred = cv2.GaussianBlur(resized, (3, 3), 0)
print("Gaussian Blur Filter Applied Successfully.")
"""
    elif "3. HOG & LBP Feature" in code_template_choice:
        sample_code = """# HOG vs LBP Feature Extraction Comparison
import numpy as np
from sklearn.datasets import load_digits
from skimage.feature import hog, local_binary_pattern

digits = load_digits()
img = ((digits.images[0] / 16.0) * 255.0).astype(np.uint8)

# HOG Extraction
hog_fd, hog_vis = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
print("HOG Feature Vector Dimensionality:", len(hog_fd))

# LBP Extraction
lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
print("LBP Feature Output Shape:", lbp.shape)
"""
    elif "4. Multi-Algorithm" in code_template_choice:
        sample_code = """# Multi-Classifier Benchmark Suite
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "SVM (RBF Kernel)": SVC(C=5.0),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5)
}

print("=== CLASSIFICATION BENCHMARK ===")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f" -> {name:<25}: Accuracy = {acc * 100:.2f}%")
"""
    else:
        sample_code = """# Confusion Matrix & Classification Report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

model = SVC(C=5.0).fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
"""

    user_code = st.text_area("Python Script Editor", value=sample_code, height=290, key="step6_code_editor")
    run_code_btn = st.button("Execute Script", type="primary", use_container_width=True, key="step6_exec_btn")

    if run_code_btn:
        stdout_capture = io.StringIO()
        start_exec = time.time()
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
                exec_globals = {
                    "np": np,
                    "cv2": cv2,
                    "pd": pd,
                    "plt": plt,
                    "load_digits": load_digits,
                    "train_test_split": train_test_split,
                    "SVC": SVC,
                    "RandomForestClassifier": RandomForestClassifier,
                    "KNeighborsClassifier": KNeighborsClassifier,
                    "MLPClassifier": MLPClassifier,
                    "LogisticRegression": LogisticRegression,
                    "DecisionTreeClassifier": DecisionTreeClassifier,
                    "accuracy_score": accuracy_score,
                    "classification_report": classification_report,
                    "confusion_matrix": confusion_matrix,
                    "hog": hog,
                    "local_binary_pattern": local_binary_pattern
                }
                exec(user_code, exec_globals)
            duration = time.time() - start_exec
            output_str = stdout_capture.getvalue()
            if not output_str.strip():
                output_str = "[Process completed successfully with exit code 0]"

            st.markdown(f"""
            <div class="terminal-container">
                <div class="terminal-topbar">
                    <span class="term-btn red"></span>
                    <span class="term-btn yellow"></span>
                    <span class="term-btn green"></span>
                    <span class="term-title">bash - python exec stdout [duration: {duration:.4f}s]</span>
                </div>
                <div class="terminal-window">{output_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.success("[SUCCESS] Script executed without runtime errors.")
        except Exception as e:
            output_str = stdout_capture.getvalue() + f"\nTraceback Exception: {e}"
            st.markdown(f"""
            <div class="terminal-container">
                <div class="terminal-topbar">
                    <span class="term-btn red"></span>
                    <span class="term-btn yellow"></span>
                    <span class="term-btn green"></span>
                    <span class="term-title">bash - python error</span>
                </div>
                <div class="terminal-window" style="color: #F87171;">{output_str}</div>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"[ERROR] Execution failed: {e}")

with code_subtab2:
    st.markdown("""
    ### Compilation & Execution Manual

    This platform is engineered as a modular, 5-stage Machine Learning system featuring both a Command Line Interface (CLI) engine and an enterprise web dashboard.

    ---

    #### 1. Repository Structure
    ```
    Machine learning/
    ├── app.py              # Continuous Workflow Dashboard (Streamlit)
    ├── ml_pipeline.py      # Standalone 5-Stage Pipeline Engine (Python OOP Script)
    ├── requirements.txt    # Production Dependencies
    └── digit_mlp_model.pkl # Serialized Model Binary Checkpoint
    ```

    ---

    #### 2. Dependency Installation
    Open your shell or terminal and execute:
    ```bash
    pip install -r requirements.txt
    ```

    ---

    #### 3. Bytecode Compilation Verification
    To compile the source code into optimized bytecode (`.pyc`) and check for syntax anomalies:
    ```bash
    # Compile backend pipeline engine
    python -m py_compile ml_pipeline.py

    # Compile dashboard application
    python -m py_compile app.py
    ```

    ---

    #### 4. Executing CLI Automated Pipeline
    To execute the full 5-stage pipeline from your terminal:
    ```bash
    python ml_pipeline.py
    ```
    **Pipeline Execution Stages:**
    * `Stage 01` Ingests benchmark dataset (1,797 samples)
    * `Stage 02 & 03` Executes spatial transformation and HOG feature computation
    * `Stage 04` Fits Support Vector Machine (SVM) classifier
    * `Stage 05` Generates accuracy (99.56%), macro metrics, and confusion matrix

    ---

    #### 5. Launching the Web Dashboard
    To start the enterprise web platform in your browser:
    ```bash
    streamlit run app.py
    ```
    The platform will open at `http://localhost:8501`.
    """)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""<div class="subhead-label"><span class="subhead-dot"></span> Source Code Export</div>""", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        ml_pipe_path = os.path.join(BASE_DIR, "ml_pipeline.py")
        if os.path.exists(ml_pipe_path):
            with open(ml_pipe_path, "r", encoding="utf-8") as f_pipe:
                pipe_code_data = f_pipe.read()
        else:
            pipe_code_data = "# ml_pipeline.py script"
        st.download_button(
            label="Download Pipeline Script (ml_pipeline.py)",
            data=pipe_code_data,
            file_name="ml_pipeline.py",
            mime="text/x-python",
            use_container_width=True,
            key="dl_pipe_btn"
        )

    with d_col2:
        app_code_path = os.path.join(BASE_DIR, "app.py")
        if os.path.exists(app_code_path):
            with open(app_code_path, "r", encoding="utf-8") as f_app:
                app_code_data = f_app.read()
        else:
            app_code_data = "# app.py script"
        st.download_button(
            label="Download Dashboard Script (app.py)",
            data=app_code_data,
            file_name="app.py",
            mime="text/x-python",
            use_container_width=True,
            key="dl_app_btn"
        )

st.markdown("</div>", unsafe_allow_html=True)
