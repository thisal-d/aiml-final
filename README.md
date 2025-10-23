# 🩺 Hypertension Prediction — Model Comparison and Evaluation

## 📋 Table of Contents
- [📘 Project Overview](#-project-overview)
- [📊 Dataset Recommendations](#-dataset-recommendations)
- [🧠 Model Suitability Analysis](#-model-suitability-analysis)
- [⚙️ Implementation Details](️-implementation-details)
- [🔧 Hyperparameter Tuning](#-hyperparameter-tuning)
- [📊 Dataset Evolution](#-dataset-evolution)
- [🧩 Model-Dataset Pairing](#-model-dataset-pairing)
- [🧾 Key Insights](#-key-insights)
- [🚀 Next Steps](#-next-steps)

---

## 📘 Project Overview

This project aims to predict whether a person has **high blood pressure (hypertension)** based on demographic, lifestyle, and clinical features. Multiple classification algorithms were applied on **progressively preprocessed datasets** to analyze how preprocessing depth impacts model performance.

### 🔍 Research Question
How does the depth of data preprocessing affect the performance of different classification models in predicting hypertension?

### 🎯 Objectives
- Compare multiple classification algorithms
- Evaluate impact of preprocessing stages
- Identify optimal model-dataset pairings
- Provide insights for clinical prediction systems

---

## 📊 Dataset Recommendations

### Available Datasets (Preprocessing Pipeline)

| Level | Dataset Name | Preprocessing Steps | Best For |
|-------|-------------|---------------------|----------|
| 1 | `hypertension_dataset(encoded).csv` | Categorical encoding only | <span style="color:green">Tree-based models</span> |
| 2 | `hypertension_dataset(encoded-balanced).csv` | + SMOTE balancing | <span style="color:blue">Imbalance-sensitive models</span> |
| 3 | `hypertension_dataset(encoded-balanced-feature_engineered).csv` | + Feature engineering | <span style="color:purple">Domain-aware models</span> |
| 4 | `hypertension_dataset(encoded-balanced-feature_engineered-scaled).csv` | + Standardization | <span style="color:orange">Distance-based models</span> |
| 5 | `hypertension_dataset(encoded-balanced-feature_engineered-scaled-selected).csv` | + Feature selection | <span style="color:red">Interpretable models</span> |
| 6 | `hypertension_dataset(encoded-balanced-feature_engineered-scaled-selected-pca).csv` | + PCA transformation | <span style="color:teal">Dimensionality-sensitive models</span> |

### 🏆 Optimal Dataset Selection by Model

| Model | Recommended Dataset | Reasoning |
|-------|-------------------|-----------|
| <span style="color:blue">📊 Logistic Regression</span> | Level 5 | Scaling essential for regularization; feature selection improves interpretability |
| <span style="color:green">🔍 KNN</span> | Level 6 | Benefits from scaling and dimensionality reduction |
| <span style="color:purple">🌳 Decision Tree</span> | Level 3 | No scaling needed; feature engineering enhances splits |
| <span style="color:orange">🌲 Random Forest</span> | Level 5 | Feature selection focuses on important variables |
| <span style="color:red">🚀 XGBoost</span> | Level 5 | Feature selection focuses on important variables; handles non-linearity well |
| <span style="color:teal">🧠 Neural Network</span> | Level 6 | Scaling essential; PCA reduces input dimensionality |
| <span style="color:indigo">📈 Naïve Bayes</span> | Level 6 | PCA creates orthogonal features satisfying independence assumption |

---

## 🧠 Model Suitability Analysis

### 🎯 Model-Dataset Compatibility

| Model | Suitability | Recommended Dataset | Reason |
|-------|-------------|---------------------|--------|
| <span style="color:blue">📊 Logistic Regression</span> | <span style="color:green">✅ Highly suitable</span> | Level 6 | PCA ensures linear separability, scaling improves convergence |
| <span style="color:green">🔍 KNN</span> | <span style="color:green">✅ Suitable</span> | Level 6 | PCA + scaling improve Euclidean distance accuracy |
| <span style="color:purple">🌳 Decision Tree</span> | <span style="color:orange">⚠️ Partially suitable</span> | Level 5 | Trees prefer original features for interpretability |
| <span style="color:orange">🌲 Random Forest</span> | <span style="color:green">✅ Suitable</span> | Level 5 | Ensemble generalizes well with high-dimensional features |
| <span style="color:red">🚀 XGBoost</span> | <span style="color:green">✅ Very suitable</span> | Level 5 | Feature selection focuses on important variables; handles non-linearity well |
| <span style="color:teal">🧠 Neural Network</span> | <span style="color:orange">⚠️ Moderate</span> | Level 5 | Needs meaningful features; PCA may reduce interpretability |
| <span style="color:indigo">📈 Naïve Bayes</span> | <span style="color:green">✅ Excellent</span> | Level 6 | PCA creates Gaussian-distributed features |

---

## ⚙️ Implementation Details

### 🔧 Libraries Used

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
```

### 🔄 Pipeline Overview

1. **Data Preprocessing:** 
   ```mermaid
   graph LR
   A[Raw Data] --> B[Encoding]
   B --> C[Balancing]
   C --> D[Feature Engineering]
   D --> E[Scaling]
   E --> F[Feature Selection]
   F --> G[PCA]
   ```

2. **Model Training:** Each model trained on its optimal dataset variant

3. **Evaluation Metrics:**
   - ✅ Accuracy
   - ✅ Precision
   - ✅ Recall
   - ✅ F1-score
   - ✅ Confusion Matrix

4. **Tuning:** GridSearchCV for hyperparameter optimization

---

## 🔧 Hyperparameter Tuning

### 📊 Hyperparameter Configuration

| Model | Key Hyperparameters | Effect | Tuning Method |
|-------|-------------------|--------|---------------|
| <span style="color:blue">📊 Logistic Regression</span> | `C`, `solver`, `max_iter` | Regularization strength | GridSearchCV |
| <span style="color:green">🔍 KNN</span> | `n_neighbors`, `weights`, `metric` | Distance calculation | GridSearchCV |
| <span style="color:purple">🌳 Decision Tree</span> | `max_depth`, `min_samples_split` | Controls tree growth | GridSearchCV |
| <span style="color:orange">🌲 Random Forest</span> | `n_estimators`, `max_depth` | Ensemble complexity | GridSearchCV |
| <span style="color:red">🚀 XGBoost</span> | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | Ensemble complexity and regularization | GridSearchCV |
| <span style="color:teal">🧠 Neural Network</span> | `hidden_layer_sizes`, `activation` | Network architecture | Manual/GridSearchCV |
| <span style="color:indigo">📈 Naïve Bayes</span> | N/A | Simple probabilistic model | Not required |

### 💻 Example: XGBoost GridSearchCV

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)
```

---

## 📊 Dataset Evolution

### 🔄 Transformation Journey

| Dataset | Main Changes | Purpose |
|---------|-------------|---------|
| Level 1 | Label/One-hot encoding | Base dataset, minimal preprocessing |
| Level 2 | + SMOTE resampling | Remove class imbalance |
| Level 3 | + Derived features | Add domain knowledge |
| Level 4 | + Standardization | Prepare for distance-based models |
| Level 5 | + Feature selection | Reduce overfitting |
| Level 6 | + PCA transformation | Dimensionality reduction |

### 📈 Preprocessing Benefits

| Step | Benefits | Best For |
|------|----------|----------|
| **Encoding** | Converts categorical to numerical | All models |
| **Balancing** | Addresses class imbalance | Neural Networks, XGBoost |
| **Feature Engineering** | Creates domain-specific features | Tree-based models, XGBoost |
| **Scaling** | Standardizes feature ranges | Distance-based models |
| **Feature Selection** | Reduces noise, improves interpretability | XGBoost, Random Forest |
| **PCA** | Maximizes variance, reduces dimensionality | Curse of dimensionality |

---

## 🧩 Model-Dataset Pairing

### 🎯 Optimal Combinations

| Category | Recommended Dataset | Best Models | Reason |
|----------|---------------------|-------------|--------|
| **Raw + Balanced** | Level 2 | Decision Tree | Maintains interpretability |
| **Feature Engineered** | Level 3 | Random Forest, XGBoost, MLP | Benefits from richer feature relationships |
| **Scaled** | Level 4 | KNN, Logistic Regression | Improves distance-based calculations |
| **Selected Features** | Level 5 | Random Forest, XGBoost, MLP | Removes redundant predictors |
| **PCA Transformed** | Level 6 | Logistic Regression, Naïve Bayes | Low-dimensional, smooth representation |

---

## 🧾 Key Insights

### 🔍 Major Findings

1. **📊 PCA Impact**: 
   - ✅ Benefits linear models (Logistic Regression)
   - ❌ Reduces tree-based model performance (including XGBoost)
   - ✅ Creates Gaussian-like features for Naïve Bayes

2. **🌳 Tree-Based Models**:
   - ✅ Perform best on non-PCA datasets
   - ✅ Benefit from feature engineering
   - ❌ Don't require scaling
   - ✅ XGBoost shows strong performance with feature selection (Level 5)

3. **🔍 Distance-Based Models**:
   - ✅ Require scaling for optimal performance
   - ✅ Benefit from PCA dimensionality reduction
   - ✅ KNN most sensitive to preprocessing

4. **⚖️ Class Balance**:
   - ✅ SMOTE improves recall for minority class
   - ✅ Critical for Neural Networks and XGBoost
   - ⚠️ Less impact on tree-based models

5. **🧠 Feature Engineering**:
   - ✅ Enhances all model performances
   - ✅ Particularly valuable for clinical interpretation
   - ✅ Most impactful preprocessing step after encoding

6. **🚀 XGBoost**:
   - ✅ Performs well with feature selection (Level 5)
   - ✅ Handles non-linear relationships effectively
   - ✅ Less sensitive to scaling but benefits from feature engineering