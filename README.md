# HealthGuard AI: Clinical Diabetes Risk Intelligence Platform
> **Phase 4: Final Project Implementation (Data Sciences Mini Project)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.0](https://img.shields.io/badge/flask-3.1.3-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange.svg)](https://xgboost.readthedocs.io/)
[![ROC-AUC 97.6%](https://img.shields.io/badge/ROC--AUC-97.6%25-brightgreen.svg)]()
[![Recall 88.1%](https://img.shields.io/badge/Clinical%20Recall-88.1%25-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, full-stack predictive healthcare SaaS application engineered to detect early cardiometabolic diabetes risk using non-invasive clinical biomarkers. Trained on a cohort of **100,000 patient records**, the platform pairs fine-tuned gradient-boosted decision trees (**XGBoost**) with an accessible, light-theme clinical design system inspired by modern healthcare-fintech aesthetics.

---

## Key Features & Highlights

- **Clinical-Grade ML Pipeline (`train_model.py`)**:
  - Preprocesses 100,000 patient records (deduplication, BMI outlier clipping, category normalization).
  - Evaluates 5 candidate classifiers under 5-Fold Stratified Cross-Validation: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
  - Implements **Clinical Decision Threshold Calibration** (`threshold = 0.58`) prioritizing diagnostic **Recall (88.1%)** and **ROC-AUC (97.6%)** to minimize dangerous false negatives.
  - Generates exploratory data analysis (EDA) plots and diagnostic evaluation curves.
- **Explainable AI (XAI)**:
  - Deconstructs individual predictions into top 3 contributing biomarkers with percentage impact ratings, severity color coding, and clinical physiological explanations.
- **Interactive Light-Theme UI/UX**:
  - **Landing Page**: 2.2-second intro splash screen with SVG heartbeat stroke animation, staggered hero headline reveal, count-up KPI metrics, and frosted-glass auth card.
  - **Dashboard**: Interactive prediction form with synced range sliders and segmented toggles, animated SVG donut ring chart, horizontal factor contribution bars, dynamic personalized lifestyle tips, and a Chart.js gradient area risk trend tracker.
  - **Longitudinal History & CSV Export**: Searchable historical patient logs with risk level filters and one-click tabular CSV download.
  - **Model Benchmark Suite**: Interactive comparison table, ROC curves, confusion matrix, threshold tuning curves, and feature importance rankings.
- **Enterprise Security & Data Persistence**:
  - SQLite database backing user credentials with Werkzeug password hashing.
  - Role-based session management using `Flask-Login`.
  - Comprehensive server-side validation against clinical range boundaries (HTTP 422 handlers).

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend API** | Python 3.12, Flask 3.1, Flask-Login, Werkzeug |
| **Machine Learning** | Scikit-Learn, XGBoost, Pandas, NumPy, Joblib |
| **Data Viz & EDA** | Matplotlib, Seaborn, Chart.js 4.4 |
| **Frontend UI/UX** | Semantic HTML5, Vanilla CSS3 (Custom Design Tokens), Lucide Icons |
| **Database** | SQLite 3 |
| **Testing** | Python `unittest` (9/9 End-to-End Test Suite) |

---

## Project Structure

```text
aswin_ibm/
├── app.py                      # Production Flask application & REST endpoints
├── train_model.py              # ML pipeline, 5-fold CV, calibration, and plot exports
├── test_app.py                 # Comprehensive automated test suite
├── Diabetes-prediction.csv     # 100,000-record clinical dataset
├── requirements.txt            # Frozen production dependencies
├── README.md                   # GitHub project overview & execution guide
├── REPORT.md                   # Academic project report, limitations & roadmap
├── .gitignore                  # Git repository exclusion rules
├── database.db                 # SQLite database (Users & Prediction logs)
├── models/
│   ├── diabetes_model.joblib   # Serialized Scikit-learn + XGBoost Pipeline
│   └── metrics.json            # Model benchmark table & threshold calibrations
├── static/
│   ├── css/
│   │   └── style.css           # Design system tokens, glassmorphism, animations
│   ├── js/
│   │   ├── main.js             # Splash screen, hero stagger, parallax, toasts
│   │   └── dashboard.js        # Donut animation, factor bars, Chart.js trend
│   ├── eda/                    # Exploratory Data Analysis visual assets
│   │   ├── class_distribution.png
│   │   ├── feature_histograms.png
│   │   ├── correlation_heatmap.png
│   │   └── boxplots_by_target.png
│   └── models/                 # Model evaluation and diagnostic plots
│       ├── roc_curves.png
│       ├── confusion_matrix.png
│       ├── threshold_tuning.png
│       └── feature_importance.png
└── templates/
    ├── base.html               # Shared layout shell with Lucide & ambient orbs
    ├── index.html              # Split landing page & frosted-glass auth card
    ├── dashboard.html          # Clinical assessment dashboard
    ├── results.html            # Model comparison matrix & validation charts
    ├── history.html            # Patient historical logs & CSV export
    ├── about.html              # Methodology, clinical rationale, disclaimer
    └── error.html              # Accessible 404 / 500 error display
```

---

## Quickstart & Installation

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/diabetes-risk-ai.git
cd diabetes-risk-ai

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train ML Models & Generate Diagnostic Charts

```bash
python train_model.py
```
*Executes EDA generation, 5-fold cross-validation on 5 candidate models, threshold calibration, and serializes `diabetes_model.joblib` and `metrics.json`.*

### 4. Run Automated Test Suite

```bash
python test_app.py
```
*Runs 9 end-to-end tests validating authentication, model inference across risk tiers, server-side validation error codes, and CSV exports.*

### 5. Launch Application Server

```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Default Clinician Demo Account

The system is pre-seeded with a sample clinician profile containing longitudinal assessment data:
- **Email:** `demo@healthguard.org`
- **Password:** `Password123!`
*(An "Auto-Fill" button is also provided directly on the landing page auth card).*

---

## Model Benchmark Summary

| Model Architecture | 5-Fold CV ROC-AUC | 5-Fold CV Recall | Test ROC-AUC | Test Recall (Sensitivity) | Test Precision | Test F1-Score |
|---|---|---|---|---|---|---|
| Logistic Regression | 96.25% | 88.24% | 95.96% | 87.62% | 42.58% | 57.31% |
| Decision Tree | 97.35% | 91.88% | 97.09% | 90.45% | 43.30% | 58.56% |
| Random Forest | 97.56% | 90.82% | 97.35% | 89.74% | 46.28% | 61.06% |
| Gradient Boosting | 97.74% | 66.78% | 97.50% | 67.98% | 99.91% | 80.91% |
| **XGBoost (Calibrated)** | **97.87%** | **92.87%** | **97.64%** | **88.15%** | **52.79%** | **66.03%** |

> **Clinical Decision Rationale:** In diabetes screening, a false negative (failing to identify an undiagnosed diabetic patient) leads to irreversible microvascular damage. By setting the decision threshold to `0.58`, our model captures **88.15% of true positive cases** while maintaining an overall **ROC-AUC of 97.64%**.

---

## UI / UX Design System Tokens

```css
--bg: #F4F7FC;             /* Light clinic background */
--surface: #FFFFFF;        /* Pure white cards */
--surfaceglass: rgba(255, 255, 255, 0.72) + blur(18px);
--primary: #2563EB;        /* Diagnostic Blue */
--primarydark: #0B3B8F;    /* Deep Navy */
--accent: #6366F1;         /* Indigo Gradient Accent */
--text: #0F172A;           /* High contrast Slate */
--textmuted: #64748B;      /* Secondary Body */
--border: #E2E8F0;         /* Clean Card Borders */
--success: #10B981;        /* Low Risk (< 30%) */
--warning: #F59E0B;        /* Moderate Risk (30% - 60%) */
--danger: #EF4444;         /* High Risk (> 60%) */
```

---

## Medical Disclaimer

HealthGuard AI is developed exclusively as an **academic, educational, and clinical decision-support tool**. It does **not** provide formal medical diagnoses or therapeutic prescriptions. Definitive clinical evaluation requires confirmatory venous blood sampling (fasting plasma glucose or oral glucose tolerance testing) performed by a licensed medical practitioner.
