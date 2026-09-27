"""
End-to-End Machine Learning Pipeline for Image Classification
============================================================
Stages covered:
1. Load Input Image & Datasets (MNIST / Custom / Synthetic)
2. Image Preprocessing (Grayscale, Denoising, Thresholding, Morphological, Normalization)
3. Feature Extraction (HOG, Color/Intensity Histograms, Edges, Texture Descriptors, PCA)
4. Model Training (SVM, Random Forest, KNN, MLP Neural Network, Logistic Regression, XGBoost)
5. Performance Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC)

Can be executed directly: `python ml_pipeline.py`
"""

import os
import sys
import time
import numpy as np
import cv2
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog, local_binary_pattern


# =====================================================================
# STAGE 1: LOAD INPUT IMAGE & DATASET
# =====================================================================
class ImageDatasetLoader:
    """Handles loading and synthetic generation of benchmark image datasets."""

    @staticmethod
    def load_digits_dataset():
        """Loads the standard 8x8 handwritten digits dataset (1,797 samples, 10 classes)."""
        digits = load_digits()
        X_images = ((digits.images / 16.0) * 255.0).astype(np.uint8)
        y = digits.target
        class_names = [str(i) for i in range(10)]
        return X_images, y, class_names

    @staticmethod
    def generate_synthetic_shapes(num_samples_per_class=100, img_size=(28, 28)):
        """
        Generates synthetic geometric shapes dataset:
        0: Circle, 1: Rectangle/Square, 2: Triangle, 3: Cross
        """
        np.random.seed(42)
        images = []
        labels = []
        class_names = ["Circle", "Square", "Triangle", "Cross"]

        for class_idx in range(4):
            for _ in range(num_samples_per_class):
                img = np.zeros(img_size, dtype=np.uint8)
                center_x = np.random.randint(8, 20)
                center_y = np.random.randint(8, 20)
                size = np.random.randint(5, 9)

                if class_idx == 0:  # Circle
                    cv2.circle(img, (center_x, center_y), size, 255, -1)
                elif class_idx == 1:  # Square
                    cv2.rectangle(img, (center_x - size, center_y - size), (center_x + size, center_y + size), 255, -1)
                elif class_idx == 2:  # Triangle
                    pt1 = (center_x, center_y - size)
                    pt2 = (center_x - size, center_y + size)
                    pt3 = (center_x + size, center_y + size)
                    pts = np.array([pt1, pt2, pt3], np.int32)
                    cv2.fillPoly(img, [pts], 255)
                elif class_idx == 3:  # Cross
                    thickness = np.random.randint(2, 4)
                    cv2.line(img, (center_x - size, center_y), (center_x + size, center_y), 255, thickness)
                    cv2.line(img, (center_x, center_y - size), (center_x, center_y + size), 255, thickness)

                # Add slight Gaussian noise & random rotation
                angle = np.random.uniform(-25, 25)
                M = cv2.getRotationMatrix2D((img_size[0] // 2, img_size[1] // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, img_size)
                noise = np.random.normal(0, 10, img_size).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                images.append(img)
                labels.append(class_idx)

        return np.array(images), np.array(labels), class_names


# =====================================================================
# STAGE 2: IMAGE PREPROCESSING
# =====================================================================
class ImagePreprocessor:
    """Provides OpenCV and NumPy image preprocessing operations."""

    @staticmethod
    def to_grayscale(image):
        """Converts RGB/RGBA image to single-channel Grayscale."""
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.copy()

    @staticmethod
    def denoise(image, method="gaussian", kernel_size=3):
        """Removes noise using Gaussian, Median, or Bilateral filters."""
        img = image.astype(np.uint8) if image.dtype != np.uint8 else image
        k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        if method == "gaussian":
            return cv2.GaussianBlur(img, (k, k), 0)
        elif method == "median":
            return cv2.medianBlur(img, k)
        elif method == "bilateral":
            return cv2.bilateralFilter(img, d=k * 2, sigmaColor=75, sigmaSpace=75)
        return img

    @staticmethod
    def threshold(image, method="otsu", threshold_val=127):
        """Binarizes the image using Otsu, Adaptive, or Binary thresholding."""
        img = image.astype(np.uint8) if image.dtype != np.uint8 else image
        if method == "otsu":
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        elif method == "adaptive":
            return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif method == "binary":
            _, thresh = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
            return thresh
        return img

    @staticmethod
    def morphology(image, op="none", kernel_size=3):
        """Applies morphological operations: dilation, erosion, opening, closing."""
        if op == "none":
            return image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if op == "dilate":
            return cv2.dilate(image, kernel, iterations=1)
        elif op == "erode":
            return cv2.erode(image, kernel, iterations=1)
        elif op == "open":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image

    @staticmethod
    def normalize_and_resize(image, target_size=(28, 28)):
        """Resizes and normalizes intensity to [0.0, 1.0]."""
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized, resized


# =====================================================================
# STAGE 3: FEATURE EXTRACTION
# =====================================================================
class FeatureExtractor:
    """Extracts distinctive visual descriptors from images."""

    @staticmethod
    def extract_hog_features(image, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False):
        """Extracts Histogram of Oriented Gradients (HOG) descriptor and optional visualization."""
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            img_uint8 = image

        # Ensure image is at least 16x16 for block calculation
        if img_uint8.shape[0] < 16 or img_uint8.shape[1] < 16:
            img_uint8 = cv2.resize(img_uint8, (28, 28))

        if visualize:
            fd, hog_image = hog(
                img_uint8,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=True,
                feature_vector=True
            )
            return fd, hog_image
        else:
            fd = hog(
                img_uint8,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=False,
                feature_vector=True
            )
            return fd, None

    @staticmethod
    def extract_intensity_histogram(image, bins=32):
        """Calculates normalized pixel intensity histogram."""
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            img_uint8 = image
        hist = cv2.calcHist([img_uint8], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    @staticmethod
    def extract_edges(image):
        """Extracts Canny edge density and Sobel gradient vectors."""
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            img_uint8 = image
        canny = cv2.Canny(img_uint8, 50, 150)
        sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edge_feat = np.array([
            np.mean(canny) / 255.0,
            np.std(canny) / 255.0,
            np.mean(np.abs(sobelx)),
            np.mean(np.abs(sobely)),
            np.std(sobelx),
            np.std(sobely)
        ])
        return edge_feat, canny

    @staticmethod
    def extract_texture_lbp(image, num_points=8, radius=1):
        """Extracts Local Binary Patterns (LBP) texture histogram."""
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            img_uint8 = image
        lbp = local_binary_pattern(img_uint8, num_points, radius, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist, lbp

    @classmethod
    def extract_composite_features(cls, image, feature_type="HOG", visualize=False):
        """Extracts chosen feature representation."""
        if feature_type == "HOG":
            feat, vis = cls.extract_hog_features(image, visualize=visualize)
            return feat, vis
        elif feature_type == "Raw Pixels":
            norm, resized = ImagePreprocessor.normalize_and_resize(image, (28, 28))
            return norm.flatten(), resized
        elif feature_type == "Intensity Histogram":
            feat = cls.extract_intensity_histogram(image, bins=32)
            return feat, None
        elif feature_type == "LBP Texture":
            feat, lbp_img = cls.extract_texture_lbp(image)
            return feat, lbp_img
        elif feature_type == "Hybrid (HOG + Histogram + Edges)":
            hog_f, hog_vis = cls.extract_hog_features(image, visualize=visualize)
            hist_f = cls.extract_intensity_histogram(image, bins=16)
            edge_f, _ = cls.extract_edges(image)
            combined = np.hstack([hog_f, hist_f, edge_f])
            return combined, hog_vis
        else:
            norm, resized = ImagePreprocessor.normalize_and_resize(image, (28, 28))
            return norm.flatten(), resized


# =====================================================================
# STAGE 4: MODEL TRAINING & ALGORITHMS
# =====================================================================
class ModelTrainer:
    """Manages training, cross-validation, and hyperparameter configuration."""

    SUPPORTED_MODELS = {
        "Support Vector Machine (SVM)": SVC,
        "Random Forest Classifier": RandomForestClassifier,
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier,
        "Multi-Layer Perceptron (MLP)": MLPClassifier,
        "Logistic Regression": LogisticRegression,
        "Decision Tree": DecisionTreeClassifier
    }

    @staticmethod
    def get_model(model_name="Support Vector Machine (SVM)", **hyperparams):
        """Instantiates classifier with given hyperparameters."""
        if model_name == "Support Vector Machine (SVM)":
            c_val = hyperparams.get("C", 1.0)
            kernel = hyperparams.get("kernel", "rbf")
            gamma = hyperparams.get("gamma", "scale")
            return SVC(C=c_val, kernel=kernel, gamma=gamma, probability=True, random_state=42)

        elif model_name == "Random Forest Classifier":
            n_estimators = hyperparams.get("n_estimators", 100)
            max_depth = hyperparams.get("max_depth", None)
            criterion = hyperparams.get("criterion", "gini")
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=42)

        elif model_name == "K-Nearest Neighbors (KNN)":
            n_neighbors = hyperparams.get("n_neighbors", 5)
            weights = hyperparams.get("weights", "distance")
            metric = hyperparams.get("metric", "minkowski")
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

        elif model_name == "Multi-Layer Perceptron (MLP)":
            hidden_layers = hyperparams.get("hidden_layers", (100, 50))
            activation = hyperparams.get("activation", "relu")
            alpha = hyperparams.get("alpha", 0.0001)
            max_iter = hyperparams.get("max_iter", 200)
            return MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, alpha=alpha, max_iter=max_iter, random_state=42)

        elif model_name == "Logistic Regression":
            c_val = hyperparams.get("C", 1.0)
            max_iter = hyperparams.get("max_iter", 200)
            return LogisticRegression(C=c_val, max_iter=max_iter, random_state=42)

        elif model_name == "Decision Tree":
            max_depth = hyperparams.get("max_depth", None)
            criterion = hyperparams.get("criterion", "gini")
            return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

        return SVC(probability=True, random_state=42)

    @classmethod
    def train_pipeline(cls, X_train, y_train, model_name="Support Vector Machine (SVM)", **hyperparams):
        """Trains classifier and measures execution time."""
        model = cls.get_model(model_name, **hyperparams)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        return model, training_time


# =====================================================================
# STAGE 5: PERFORMANCE EVALUATION
# =====================================================================
class PerformanceEvaluator:
    """Calculates comprehensive classification metrics and diagnostics."""

    @staticmethod
    def evaluate(model, X_test, y_test, class_names=None):
        """Computes accuracy, precision, recall, f1, confusion matrix, and report."""
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

        # Probabilities if available
        prob_dist = None
        roc_auc = None
        if hasattr(model, "predict_proba"):
            try:
                prob_dist = model.predict_proba(X_test)
                if len(np.unique(y_test)) > 1:
                    roc_auc = roc_auc_score(y_test, prob_dist, multi_class="ovr", average="macro")
            except Exception:
                pass

        return {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "roc_auc": roc_auc,
            "y_pred": y_pred
        }


# =====================================================================
# STANDALONE PIPELINE EXECUTION (CLI RUNNER)
# =====================================================================
def run_full_pipeline(dataset_choice="digits", feature_choice="HOG", model_choice="Support Vector Machine (SVM)"):
    print("=" * 60)
    print("=== EXECUTING 5-STAGE MACHINE LEARNING PIPELINE ===")
    print("=" * 60)

    # Stage 1: Load Dataset
    print(f"\n[Stage 1] Loading Dataset: '{dataset_choice}'...")
    if dataset_choice == "digits":
        images, labels, class_names = ImageDatasetLoader.load_digits_dataset()
    else:
        images, labels, class_names = ImageDatasetLoader.generate_synthetic_shapes()
    print(f" -> Loaded {len(images)} images across {len(class_names)} classes: {class_names}")

    # Stage 2 & 3: Preprocessing & Feature Extraction
    print(f"\n[Stage 2 & 3] Preprocessing Images & Extracting Features: '{feature_choice}'...")
    features = []
    for img in images:
        gray = ImagePreprocessor.to_grayscale(img)
        resized_gray = cv2.resize(gray, (28, 28))
        feat, _ = FeatureExtractor.extract_composite_features(resized_gray, feature_type=feature_choice)
        features.append(feat)

    X = np.array(features)
    y = labels
    print(f" -> Feature Matrix Shape: {X.shape}, Label Vector Shape: {y.shape}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f" -> Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")

    # Stage 4: Model Training
    print(f"\n[Stage 4] Training Model: '{model_choice}'...")
    model, train_time = ModelTrainer.train_pipeline(X_train, y_train, model_name=model_choice)
    print(f" -> Model trained successfully in {train_time:.4f} seconds!")

    # Stage 5: Performance Evaluation
    print(f"\n[Stage 5] Performance Evaluation:")
    metrics = PerformanceEvaluator.evaluate(model, X_test, y_test, class_names=class_names)
    print(f" -> Accuracy Score   : {metrics['accuracy'] * 100:.2f}%")
    print(f" -> Macro Precision  : {metrics['precision_macro'] * 100:.2f}%")
    print(f" -> Macro Recall     : {metrics['recall_macro'] * 100:.2f}%")
    print(f" -> Macro F1-Score   : {metrics['f1_macro'] * 100:.2f}%")
    if metrics['roc_auc'] is not None:
        print(f" -> Macro ROC-AUC    : {metrics['roc_auc'] * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\n" + "=" * 60)
    print("[SUCCESS] PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    return model, metrics


if __name__ == "__main__":
    run_full_pipeline()
