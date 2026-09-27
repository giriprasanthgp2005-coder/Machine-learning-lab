# Machine Learning Laboratory

A curated collection of practical machine learning experiments, algorithms, diagnostic implementations, and the **OmniML Enterprise Visual ML Platform** developed as part of the Machine Learning Laboratory course.

---

## 📚 Experiments Index

| Experiment | Title / Topic | Algorithm / Method | Key Files / Location |
|:---|:---|:---|:---|
| **Exp 1** | Decision Tree Classification | Decision Tree Classifier on Iris Dataset | [`exp1.ipynb`](exp1.ipynb) |
| **Exp 2** | Spam Mail Detection | Support Vector Machine (SVM) Classifier | [`exp2.ipynb`](exp2.ipynb) |
| **Exp 3** | Facial Recognition System | Artificial Neural Network (ANN) | [`exp3.ipynb`](exp3.ipynb) |
| **Exp 5** | K-Nearest Neighbours | K-NN Classification on Iris Dataset | [`exp5.ipynb`](exp5.ipynb) |
| **Exp 6** | Probabilistic Classification | Naïve Bayes Classifier | [`exp6.ipynb`](exp6.ipynb) |
| **Exp 7** | Regression & Clustering | Logistic Regression, Clustering & Ensembles | [`exp7.ipynb`](exp7.ipynb) |
| **Exp 8** | Bayesian Medical Diagnosis | Bayesian Network & Variable Elimination | [`exp8.ipynb`](exp8.ipynb), [`heart.csv`](heart.csv) |
| **Exp 9** | Online Fraud Detection | Anomaly & Fraud Classification | [`exp9.ipynb`](exp9.ipynb) |
| **Exp 10** | **OmniML: Visual ML Platform** | **Full 5-Stage ML Pipeline + Interactive Web Dashboard** | [📁 `10 exp/`](10%20exp/) |

---

## 🌟 Featured Project: Experiment 10 (OmniML)

The complete end-to-end Machine Learning project is located in the dedicated **[`10 exp/`](10%20exp/)** directory.

### Project Highlights
- **End-to-End Pipeline**: Standalone Python OOP pipeline (`ml_pipeline.py`) covering Data Ingestion, Image Preprocessing, HOG Feature Extraction, Multi-Classifier Training (SVM, RF, KNN, MLP), and Comprehensive Diagnostics.
- **Executive Web Dashboard**: Real-time interactive Streamlit application (`app.py`) featuring an integrated visual interface across all 6 stages.
- **Formal Project Report**: Publication-grade technical report (`OmniML_Project_Report.docx`) with generated figures and evaluation metrics.

### Quick Start for Experiment 10
```bash
# Navigate to the Experiment 10 directory
cd "10 exp"

# Install dependencies
pip install -r requirements.txt

# Run the automated CLI pipeline
python ml_pipeline.py

# Launch the interactive web dashboard
streamlit run app.py
```

For full architecture details, compilation instructions, and diagnostic logs, see the [Experiment 10 README](10%20exp/README.md).
