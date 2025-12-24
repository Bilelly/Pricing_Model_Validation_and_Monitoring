# 🚗 Drift Monitoring System for Insurance Pricing Models

> **By Bilel SAYOUD – Data Scientist & Actuarial Engineer**  
> *An end-to-end ML pipeline for insurance pricing with robust drift detection capabilities*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-black)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Project Overview

In insurance pricing, models rapidly become obsolete due to changing market conditions and policyholder behaviors. This project implements a production-ready MLOps pipeline addressing two critical challenges:

1. **Pricing Accuracy**: Building a high-performance prediction engine for commercial premiums.
2. **Model Stability**: Implementing multi-method drift monitoring to detect data distribution shifts and prevent financial losses from under/over-pricing.

This solution replicates industry best practices used in regulated banking and insurance environments where model governance, explainability, and continuous monitoring are mandatory.

## 📁 Project Structure


````
DRIFT-MONITORING-INSURANCE/
├── data/
│ ├── X_train.csv # Historical training data
│ ├── X_test.csv # Testing dataset
│ └── X_drift.csv # Production data (for drift detection)
├── models/
│ ├── best_xgb_regressor_model.pkl # Final trained model with pipeline
│ └── drift_discriminator.pkl # Binary classifier for drift detection
├── notebooks/
│ ├── Prediction/
│ │ ├── 01_data_exploration_EDA.ipynb
│ │ ├── 02_data_preprocessing.ipynb
│ │ ├── 03_data_modeling.ipynb
│ │ └── 04_data_optimizer.ipynb
│ └── Drift/
│ └── 00_data_drift_detection.ipynb
├── src/
│ ├── DriftComputePSI.py # Custom PSI implementation
│ ├── EvaluateModelDrift.py # Drift detection metrics
│ ├── EvaluateRegression.py # Regression evaluation metrics
│ └── RemoveOutliers.py # Outlier handling functions
├── reports/
│ ├── model_performance.pdf # Performance metrics
│ └── drift_analysis.pdf # Drift detection results
├── requirements.txt
└── README.md
````

## ⚙️ Part 1: Pricing Engine - Predictive Model

### 📊 Data Science Pipeline

A robust 4-step process transforms raw insurance data into a production-ready pricing model:

1. **Exploratory Data Analysis** (`01_data_exploration_EDA.ipynb`)  
   - Analysis of 22,481 automobile insurance policies
   - Target variable: `PrimeCommerciale` (continuous)
   - Key features: Driver profile, vehicle characteristics, policy attributes

2. **Data Preprocessing** (`02_data_preprocessing.ipynb`)
   - Outlier removal using domain-specific thresholds (`src/RemoveOutliers.py`)
   - Feature engineering for temporal and categorical variables
   - Implementation of a Scikit-learn pipeline with `ColumnTransformer` to prevent data leakage:
     ```python
     preprocessor = ColumnTransformer(
         transformers=[
             ('num', StandardScaler(), numerical_features),
             ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
         ])
     ```

3. **Model Benchmarking** (`03_data_modeling.ipynb`)
   - Comparative evaluation of four regression algorithms:

   | Model               | RMSE    | R² Score |
   |---------------------|---------|----------|
   | **XGBoost Regressor** | **108.59** | **0.7482** |
   | CatBoost Regressor  | 111.33  | 0.7354   |
   | Random Forest       | 114.95  | 0.7179   |
   | Linear Regression   | 131.33  | 0.6318   |

4. **Hyperparameter Optimization** (`04_data_optimizer.ipynb`)
   - Optuna-based optimization (30 trials)
   - Final parameters: `n_estimators=941`, `max_depth=7`, `learning_rate=0.032`

### 📈 Key Model Insights

- **Top features** by importance:
  - `ClasseVehicule` (0.177) - Vehicle classification tier
  - `AgeVehicule` (0.082) - Vehicle age
  - `BonusMalus` (0.059) - Driver's claim history coefficient
- The model captures non-linear relationships between vehicle characteristics and pricing

## 🔍 Part 2: Drift Monitoring System

### 📊 Multi-Method Drift Detection Framework

To ensure model reliability in production, we implement a consensus-based approach combining four complementary methods:

| Method              | Type           | Implementation       | Key Advantage                          |
|---------------------|----------------|----------------------|----------------------------------------|
| **Custom PSI**      | Statistical    | `src/DriftComputePSI.py` | Full control over binning strategy     |
| **skorecard**       | Banking Std.   | `skorecard` library  | High sensitivity to distribution tails|
| **feature-engine**  | ML-Oriented    | `feature_engine`     | Adaptive binning algorithm             |
| **XGBoost Discriminator** | Supervised | Binary classifier    | Empirical evidence of drift            |

### 📈 Drift Analysis Results

**Population Stability Index (PSI) Results**  
*Thresholds: < 0.1 (Stable), 0.1-0.25 (Moderate), > 0.25 (Significant)*

| Feature           | Custom PSI | skorecard | feature_engine | Severity    |
|-------------------|------------|-----------|----------------|-------------|
| **BonusMalus**    | 4.04       | 8.72      | 0.0005         | **Extreme** |
| **AgeVehicule**   | 3.03       | 3.44      | 0.0017         | **Critical**|
| **ClasseVehicule**| 2.44       | 2.44      | -              | **Critical**|
| **AgeConducteur** | 0.60       | 1.06      | 0.0013         | **Significant**|

> **Key Insight**: `skorecard` proves most effective at detecting structural shifts in insurance data, while `feature_engine` shows limitations for this specific use case.

### 🧪 Discriminator Model Validation

A binary XGBoost classifier was trained to distinguish between training and production data:

```python
{
  "Accuracy": 0.9739,
  "Precision": 0.9568,
  "Recall": 0.9926,
  "F1-score": 0.9744,
  "AUC-ROC": 0.9980
}
````







