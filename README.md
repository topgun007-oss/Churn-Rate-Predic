# Customer Churn Prediction

**Production-Grade ML Pipeline using TensorFlow & Scikit-learn**

End-to-end pipeline that predicts customer churn for a telecom company. Compares 10+ algorithms, applies feature engineering (22 features), SMOTE resampling, stacking/voting ensembles, SHAP explainability, calibration analysis, cost-sensitive business impact, and statistical model comparison.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
6. [Feature Engineering](#feature-engineering)
7. [Models Implemented](#models-implemented)
8. [SMOTE & Resampling](#smote--resampling)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Stacking & Voting Ensembles](#stacking--voting-ensembles)
11. [SHAP Explainability](#shap-explainability)
12. [Calibration & Statistical Rigor](#calibration--statistical-rigor)
13. [Cost-Sensitive Business Impact](#cost-sensitive-business-impact)
14. [Results & Performance](#results--performance)
15. [Key Findings](#key-findings)
16. [Saved Artifacts](#saved-artifacts)
17. [Usage & Inference](#usage--inference)

---

## Project Overview

Customer churn prediction is one of the highest-ROI ML applications. This project demonstrates the full DS2-level ML lifecycle:

**Objective:** Build a production-grade pipeline that:
- Predicts customer churn (binary classification) with high recall
- Validates features with 3 independent selection methods
- Compares 10+ algorithms including ensemble meta-learners
- Provides SHAP explanations for individual predictions
- Quantifies business ROI with cost-sensitive analysis
- Verifies model differences with statistical tests

**Tech Stack:**
- **TensorFlow / Keras** — Deep neural network with class weights
- **Scikit-learn** — Classical ML, preprocessing, pipelines, calibration
- **XGBoost / LightGBM** — Gradient boosted trees
- **SHAP** — Model explainability (TreeExplainer)
- **imbalanced-learn** — SMOTE, ADASYN, Borderline-SMOTE
- **SciPy** — Statistical model comparison

---

## Dataset

**Source:** [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (IBM Sample Dataset)

| Property | Value |
|----------|-------|
| Samples | 7,043 |
| Features | 21 (20 predictors + 1 target) |
| Target | `Churn` (Yes/No) |
| Churn Rate | 26.5% (imbalanced) |

### Feature Categories

| Category | Features |
|----------|----------|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## Project Structure

```
Customer Churn/
|-- Customer_Churn_Prediction.ipynb   # Main notebook (full pipeline)
|-- WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
|-- README.md                         # This guide
|-- INTERVIEW_GUIDE.md                # 30 FAQ interview guide (DS2 level)
|-- requirements.txt                  # Pinned dependencies
|
|-- (Generated after running notebook)
|-- best_churn_model.pkl              # Best sklearn model
|-- scaler.pkl                        # StandardScaler
|-- churn_pipeline.pkl                # End-to-end sklearn Pipeline
|-- churn_nn_model.keras              # TensorFlow neural network
|-- model_comparison_results.csv      # All model metrics
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+ (Python 3.12 recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**macOS users:** XGBoost requires OpenMP runtime:
```bash
brew install libomp
```

### Run the Notebook

```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```

---

## Pipeline Walkthrough

| Section | Content |
|---------|---------|
| 1. Imports & Config | All libraries + centralized CONFIG dict |
| 2. Data Loading | CSV loading, missing values, duplicates |
| 3. EDA | Churn distribution, contract type, tenure, charges, categorical features |
| 4. Preprocessing & Feature Engineering | Data cleaning, 22 engineered features, encoding, scaling |
| **4.5 Feature Selection** | Mutual Information + LightGBM + RFE consensus validation |
| 5. Model Training | 10 algorithms with unified evaluation |
| **5.5 Cross-Validation** | Mean +/- std for AUC, F1, Recall across 5 folds |
| 6. SMOTE & Resampling | SMOTE, ADASYN, Borderline-SMOTE comparison |
| 7. Neural Network | TensorFlow NN with class weights |
| 8. Hyperparameter Tuning | GridSearchCV on top 3 models |
| 8.5 Ensembles + Threshold | Voting, Stacking, threshold optimization |
| **8.6 SHAP Explainability** | Global importance, beeswarm, dependence, waterfall |
| **8.7 Learning Curves** | Data sufficiency analysis |
| **8.8 Calibration** | Reliability diagrams + Brier scores |
| **8.9 Statistical Comparison** | Paired t-test + bootstrap 95% CI |
| **8.10 Business Impact** | Expected value framework, Net ROI optimization |
| **8.11 sklearn Pipeline** | Production-ready pipeline construction |
| 9. Final Comparison | Results table, bar charts, confusion matrix, feature importance |
| 10. Recall Improvement | Quantified recall gains |
| 11. Save Artifacts | Model, scaler, pipeline, NN, results |

---

## Feature Engineering

22 new features (7 core + 15 advanced):

### Core Features (7)

| Feature | Description |
|---------|-------------|
| `tenure_group` | Binned tenure (0-12, 12-24, 24-48, 48-60, 60-72) |
| `AvgMonthlySpend` | TotalCharges / tenure |
| `ChargeDiff` | MonthlyCharges - AvgMonthlySpend |
| `NumServices` | Count of subscribed services |
| `HasStreaming` | Any streaming service flag |
| `HasProtection` | Any security/support flag |
| `TenureChargeInteraction` | tenure * MonthlyCharges |

### Advanced Features (15)

| Feature | Description |
|---------|-------------|
| `LogTotalCharges` / `LogMonthlyCharges` | Log transforms |
| `TenureSq` | Polynomial tenure term |
| `ChargePerService` | Value density per service |
| `IsNewCustomer` / `IsLoyalCustomer` | Lifecycle flags |
| `HighCharges` | Above 75th percentile |
| `HighRiskCombo` | No protection + month-to-month |
| `SeniorAlone` | Senior without partner |
| `LoyaltyValue` | TotalCharges / (tenure + 1) |
| `ContractMonths` / `ChargeContractInteraction` | Contract interactions |
| `ServiceTenureInteraction` | Service breadth over time |
| `NoInternet` / `ElecCheckMonthly` | Segment flags |

---

## Models Implemented

### Classical ML (10)

| Algorithm | Type |
|-----------|------|
| Logistic Regression | Linear |
| Decision Tree | Tree |
| Random Forest | Bagging |
| Gradient Boosting | Boosting |
| XGBoost | Boosting |
| LightGBM | Boosting |
| AdaBoost | Boosting |
| Bagging Classifier | Bagging |
| SVM (RBF) | Kernel |
| K-Nearest Neighbors | Instance-based |

### Deep Learning (1)

| Layer | Config |
|-------|--------|
| Dense 1 | 128 units, ReLU, BatchNorm, Dropout(0.3) |
| Dense 2 | 64 units, ReLU, BatchNorm, Dropout(0.3) |
| Dense 3 | 32 units, ReLU, Dropout(0.2) |
| Output | 1 unit, Sigmoid |

---

## SMOTE & Resampling

Three oversampling strategies applied **only to training data**:

| Method | Approach |
|--------|----------|
| SMOTE | Interpolation between minority neighbors |
| ADASYN | Focuses on harder-to-learn boundary samples |
| Borderline-SMOTE | Targets borderline minority instances |

Average recall improved from **0.514 to 0.635** (Borderline-SMOTE).

---

## SHAP Explainability

- **Global importance**: Mean |SHAP| bar chart
- **Beeswarm plot**: Directional impact of feature values
- **Dependence plots**: Interaction effects between top features
- **Waterfall plot**: Individual customer prediction explanation

---

## Calibration & Statistical Rigor

- **Calibration curves**: Reliability diagrams comparing predicted vs actual probabilities
- **Brier scores**: Quantitative calibration quality metric
- **Paired t-test**: Statistical significance of model differences on CV folds
- **Bootstrap CI**: 95% confidence interval for best model AUC (1,000 iterations)

---

## Cost-Sensitive Business Impact

Expected Value Framework:
- Revenue Saved = TP * retention_success * LTV
- Retention Cost = (TP + FP) * offer_cost
- Net ROI = Revenue Saved - Retention Cost

Finds the **business-optimal threshold** that maximizes Net ROI, not accuracy.

---

## Results & Performance

- **20+ model configurations** evaluated
- **Best AUC-ROC:** ~0.847
- **Best Accuracy:** ~80-81% (dataset ceiling: 80-82%)
- **Best Recall:** ~83% (Borderline-SMOTE enhanced)
- **Recall Improvement:** +55%+ from baseline
- **Dataset ceiling:** ~80-82% (well-documented benchmark)

---

## Key Findings

### Business Insights

1. Contract type is the #1 churn predictor (month-to-month: 42% churn)
2. New customers (0-12 months) churn most
3. Fiber optic users churn more (premium service paradox)
4. Protection services reduce churn (switching cost effect)
5. Electronic check payment correlates with highest churn

### Technical Insights

1. Boosting ensembles dominate on structured tabular data
2. SHAP confirms domain-engineered features (Contract, tenure, charges) as top predictors
3. Logistic Regression has the best calibration; tree models may need Platt scaling
4. Statistical tests show top 3 models are often not significantly different
5. Learning curves indicate feature-limited (not data-limited) performance
6. Business-optimal threshold differs from accuracy-optimal threshold

---

## Saved Artifacts

| File | Description | Usage |
|------|-------------|-------|
| `best_churn_model.pkl` | Best tuned sklearn model | `joblib.load('best_churn_model.pkl')` |
| `scaler.pkl` | StandardScaler | `joblib.load('scaler.pkl')` |
| `churn_pipeline.pkl` | End-to-end Pipeline (scaler + model) | `joblib.load('churn_pipeline.pkl')` |
| `churn_nn_model.keras` | TensorFlow NN | `keras.models.load_model('churn_nn_model.keras')` |
| `model_comparison_results.csv` | All metrics | `pd.read_csv('model_comparison_results.csv')` |

---

## Usage & Inference

### Using the Pipeline (Recommended)

```python
import joblib

pipeline = joblib.load('churn_pipeline.pkl')
# Apply same feature engineering as training to new_data
probabilities = pipeline.predict_proba(new_data_processed)[:, 1]
predictions = pipeline.predict(new_data_processed)
```

### Using the Neural Network

```python
from tensorflow import keras
import joblib

nn_model = keras.models.load_model('churn_nn_model.keras')
scaler = joblib.load('scaler.pkl')
new_data_scaled = scaler.transform(new_data_processed)
churn_probability = nn_model.predict(new_data_scaled).flatten()
```

---

*Built with TensorFlow, Scikit-learn, XGBoost, LightGBM, SHAP, and imbalanced-learn*
