# ML Classification Models Comparison

## a. Problem Statement

This project implements and compares six different machine learning classification models on the Breast Cancer dataset. The objective is to evaluate the performance of each model using comprehensive evaluation metrics including Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The comparison helps in understanding which models perform best for binary classification tasks and provides insights into model selection for real-world applications.

## b. Dataset Description

### Breast Cancer Wisconsin Dataset
- **Source**: University of Wisconsin Hospitals, Madison
- **Type**: Binary classification (Malignant vs Benign)
- **Samples**: 569 instances (357 Benign, 212 Malignant)
- **Features**: 30 numeric predictive attributes
- **Test/Train Split**: 85/15 ratio
- **Missing Values**: None
- **Target Variable**: Diagnosis (B/M)

### Custom Dataset Support
The application supports uploading custom CSV files with automatic categorical encoding, missing value handling, and target column selection.

## c. Model Performance Comparison

### Models Used

1. **Logistic Regression**: Linear classification algorithm, probabilistic model
2. **Decision Tree**: Tree-based algorithm, handles non-linear relationships
3. **K-Nearest Neighbor**: Instance-based learning, pattern recognition
4. **Naive Bayes**: Probabilistic algorithm based on Bayes' theorem
5. **Random Forest (Ensemble)**: Ensemble learning, reduces overfitting
6. **XGBoost (Ensemble)**: Gradient boosting, high-performance classification

### Performance Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9884 | 1.0000 | 0.9886 | 0.9884 | 0.9883 | 0.9753 |
| Decision Tree | 0.9767 | 0.9812 | 0.9776 | 0.9767 | 0.9766 | 0.9508 |
| K-Nearest Neighbor | 0.9535 | 0.9959 | 0.9540 | 0.9535 | 0.9532 | 0.9003 |
| Naive Bayes | 0.9419 | 0.9948 | 0.9417 | 0.9419 | 0.9417 | 0.8751 |
| Random Forest (Ensemble) | 0.9884 | 0.9988 | 0.9886 | 0.9884 | 0.9883 | 0.9753 |
| XGBoost (Ensemble) | 0.9884 | 1.0000 | 0.9886 | 0.9884 | 0.9883 | 0.9753 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Outstanding performance with perfect AUC score (1.0000), highly interpretable, excellent baseline model achieving top accuracy (98.84%) |
| Decision Tree | Strong performance (97.67% accuracy) with good interpretability, provides clear decision rules, slightly lower performance than ensemble methods |
| K-Nearest Neighbor | Good performance (95.35% accuracy) with excellent AUC (0.9959), sensitive to feature scaling, computationally expensive during prediction |
| Naive Bayes | Decent performance (94.19% accuracy) with high AUC (0.9948), fastest training and prediction, works well with probabilistic interpretation despite independence assumption |
| Random Forest (Ensemble) | Top-tier performance tied for best accuracy (98.84%) and precision (98.86%), robust to overfitting, handles non-linear relationships well, provides feature importance |
| XGBoost (Ensemble) | Elite performance with perfect AUC score (1.0000) tied for best accuracy (98.84%), optimized gradient boosting with regularization, handles complex patterns efficiently |

### Key Performance Insights

- **Three-Way Tie for Best Performance**: Logistic Regression, Random Forest, and XGBoost all achieved the highest accuracy (98.84%) and identical metrics across most measures
- **Perfect AUC Achievement**: Logistic Regression and XGBoost both achieved perfect AUC scores (1.0000), indicating excellent class separation capability
- **Ensemble Methods Excel**: Both Random Forest and XGBoost demonstrated why ensemble methods are preferred for complex classification tasks
- **Linear Model Competitiveness**: Logistic Regression proved that simpler models can compete with complex ensembles when the data is well-structured
- **Consistent High Performance**: All models achieved above 94% accuracy, indicating the dataset is well-suited for classification tasks

## Project Structure

```
ml_classification_project/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── streamlit_app.py                   # Main Streamlit web application
├── debug_csv.py                        # CSV debugging utility
├── data/                              # Data storage directory
│   ├── raw/                           # Raw dataset files
│   │   ├── .gitkeep                   # Directory placeholder
│   │   └── breast-cancer-Wisconsin-data.csv  # Main dataset
│   └── processed/                     # Processed data (if any)
├── models/                            # Machine learning models package
│   ├── __init__.py                    # Package initialization
│   ├── data_loader.py                 # Data loading and preprocessing
│   ├── evaluator.py                   # Model evaluation metrics
│   ├── pipeline.py                    # ML pipeline orchestration
│   ├── base_model.py                  # Base model interface
│   ├── logistic_regression.py         # Logistic Regression implementation
│   ├── decision_tree.py               # Decision Tree implementation
│   ├── knn.py                         # K-Nearest Neighbors implementation
│   ├── naive_bayes.py                 # Naive Bayes implementation
│   ├── random_forest.py               # Random Forest implementation
│   └── xgboost_model.py               # XGBoost implementation
└── saved_models/                      # Trained model checkpoints
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k_nearest_neighbor.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

## d. Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- XGBoost
- Matplotlib
- Seaborn
- Plotly

## e. Streamlit Application Features

- Modern UI with custom CSS styling
- Multiple dataset options (Built-in & Custom CSV)
- Six ML models with one-click training
- Comprehensive evaluation metrics
- Interactive visualizations and charts
- Performance comparison tables
- Confusion matrices and classification reports
- Custom dataset preprocessing
- Session state management
- Error handling and validation
