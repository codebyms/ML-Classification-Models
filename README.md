# ML Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification models on the Breast Cancer dataset. The objective is to evaluate the performance of each model using comprehensive evaluation metrics including Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The comparison helps in understanding which models perform best for binary classification tasks and provides insights into model selection for real-world applications.

## Dataset Description

### Breast Cancer Dataset
- **Type**: Binary classification dataset
- **Features**: 30 features describing characteristics of cell nuclei
- **Samples**: 569 samples divided into malignant and benign classes
- **Target**: Binary classification (Malignant/Benign)
- **Source**: Wisconsin Breast Cancer Dataset

### Custom Dataset Support
The application also supports uploading custom CSV files for classification tasks, with automatic handling of categorical variables and target column selection.

## Models Used

### 1. Logistic Regression
- **Type**: Linear classification algorithm
- **Characteristics**: Probabilistic model using logistic function
- **Use Case**: Binary classification, interpretable results

### 2. Decision Tree Classifier
- **Type**: Tree-based algorithm
- **Characteristics**: Non-parametric model that splits data based on feature values
- **Use Case**: Handles non-linear relationships

### 3. K-Nearest Neighbor (KNN) Classifier
- **Type**: Instance-based learning algorithm
- **Characteristics**: Classifies based on majority class of k nearest neighbors
- **Use Case**: Pattern recognition

### 4. Naive Bayes Classifier (Gaussian)
- **Type**: Probabilistic algorithm based on Bayes' theorem
- **Characteristics**: Assumes feature independence, uses Gaussian distribution
- **Use Case**: Medical diagnosis, text classification

### 5. Random Forest (Ensemble)
- **Type**: Ensemble learning algorithm
- **Characteristics**: Combines multiple decision trees using bagging
- **Use Case**: Reduces overfitting, handles complex patterns

### 6. XGBoost (Ensemble)
- **Type**: Gradient boosting algorithm
- **Characteristics**: Optimized gradient boosting with regularization
- **Use Case**: High-performance classification

## Model Performance Comparison

### Performance Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9561 | 0.9925 | 0.9561 | 0.9561 | 0.9561 | 0.9086 |
| Decision Tree | 0.9298 | 0.9298 | 0.9315 | 0.9298 | 0.9299 | 0.8538 |
| K-Nearest Neighbor | 0.9561 | 0.9875 | 0.9561 | 0.9561 | 0.9561 | 0.9086 |
| Naive Bayes | 0.9474 | 0.9850 | 0.9492 | 0.9474 | 0.9473 | 0.8898 |
| Random Forest (Ensemble) | 0.9649 | 0.9947 | 0.9651 | 0.9649 | 0.9649 | 0.9265 |
| XGBoost (Ensemble) | 0.9649 | 0.9947 | 0.9651 | 0.9649 | 0.9649 | 0.9265 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Excellent performance on linearly separable data, highly interpretable, good baseline model with consistent results across datasets |
| Decision Tree | Good performance but prone to overfitting, provides clear decision rules, performance varies with dataset complexity |
| K-Nearest Neighbor | Strong performance on smaller datasets, sensitive to feature scaling, computationally expensive during prediction |
| Naive Bayes | Surprisingly good performance despite independence assumption, fast training and prediction, works well with probabilistic interpretation |
| Random Forest (Ensemble) | Best overall performance among all models, robust to overfitting, handles non-linear relationships well, feature importance available |
| XGBoost (Ensemble) | Competitive performance with Random Forest, faster training on large datasets, handles missing values well, requires careful tuning |
