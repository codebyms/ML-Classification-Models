import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from xgboost import XGBClassifier

# Set page config
st.set_page_config(page_title="ML Classification Models Comparison", layout="wide")

# Title
st.title("Machine Learning Classification Models Comparison")
st.markdown("Implementation of 6 ML models with comprehensive evaluation metrics")

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
dataset_option = st.sidebar.selectbox(
    "Choose Dataset:",
    ["Breast Cancer", "Iris", "Wine", "Upload Custom Dataset"]
)

# Load dataset function
def load_dataset(option):
    if option == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        # sklearn uses 0=malignant,1=benign; flip to 0=Benign,1=Malignant
        df['target'] = 1 - data.target
        return df, "Breast Cancer Classification"
    elif option == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, "Iris Classification"
    elif option == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, "Wine Classification"
    else:
        return None, "Custom Dataset"

# Load the dataset
df, dataset_name = load_dataset(dataset_option)

if dataset_option != "Upload Custom Dataset":
    st.subheader(f"Dataset: {dataset_name}")
    
    # Simple Train/Test Summary
    st.markdown("### 📊 Train/Test Split Summary")
    total_samples = len(df)
    train_samples = int(total_samples * 0.85)
    test_samples = total_samples - train_samples
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Total Samples", f"{total_samples:,}")
    col2.metric("📚 Train Set", f"{train_samples:,} (85%)")
    col3.metric("🧪 Test Set", f"{test_samples:,} (15%)")
    st.markdown("---")
    
    # Display dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Features:** {df.shape[1] - 1}")
        st.write(f"**Classes:** {df['target'].nunique()}")
    
    with col2:
        st.write("**Class Distribution:**")
        class_dist = df['target'].value_counts()
        st.write(class_dist)
    
    # Display first few rows
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=5, min_samples_leaf=3),
        'K-Nearest Neighbor': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(var_smoothing=1e-9),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=3),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_weight=1, gamma=0.1)
    }
    
    # Train and evaluate models
    st.subheader("Model Training and Evaluation")
    
    results = {}
    
    for name, model in models.items():
        with st.expander(f"{name} Results"):
            # Train model
            if name in ['Logistic Regression', 'K-Nearest Neighbor', 'Naive Bayes']:  # Models that need scaling
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Calculate AUC (handle multiclass case)
            if len(np.unique(y_test)) == 2:
                if y_pred_proba is not None:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc = 0.0
            else:
                if y_pred_proba is not None:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    auc = 0.0
            
            # Store results
            results[name] = {
                'Accuracy': accuracy,
                'AUC': auc,
                'F1 Score': f1,
                'MCC': mcc
            }
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("AUC Score", f"{auc:.4f}")
            
            with col2:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC", f"{mcc:.4f}")
            
            # Save model
            model_dir = "model"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_filename = f"{model_dir}/{name.lower().replace(' ', '_').replace('-', '_')}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            
            st.success(f"Model saved as {model_filename}")
    
    # Create comparison table
    st.subheader("Model Comparison Table")
    
    comparison_df = pd.DataFrame(results).T
    st.dataframe(comparison_df.style.format("{:.4f}"))
    
    # Create visualization
    st.subheader("Performance Visualization")
    
    # Bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'AUC', 'F1 Score', 'MCC']
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results.keys()]
        models_list = list(results.keys())
        
        axes[i].bar(models_list, values)
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Best model analysis
    st.subheader("Best Model Analysis")
    
    best_accuracy = max(results[model]['Accuracy'] for model in results.keys())
    best_models_accuracy = [model for model in results.keys() if results[model]['Accuracy'] == best_accuracy]
    
    best_f1 = max(results[model]['F1 Score'] for model in results.keys())
    best_models_f1 = [model for model in results.keys() if results[model]['F1 Score'] == best_f1]
    
    best_auc = max(results[model]['AUC'] for model in results.keys())
    best_models_auc = [model for model in results.keys() if results[model]['AUC'] == best_auc]
    
    best_mcc = max(results[model]['MCC'] for model in results.keys())
    best_models_mcc = [model for model in results.keys() if results[model]['MCC'] == best_mcc]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Best Accuracy Models:**")
        for model in best_models_accuracy:
            st.write(f"- {model}: {best_accuracy:.4f}")
        st.write("**Best F1 Score Models:**")
        for model in best_models_f1:
            st.write(f"- {model}: {best_f1:.4f}")
    
    with col2:
        st.write("**Best AUC Models:**")
        for model in best_models_auc:
            st.write(f"- {model}: {best_auc:.4f}")
        st.write("**Best MCC Models:**")
        for model in best_models_mcc:
            st.write(f"- {model}: {best_mcc:.4f}")
    
    # Download results
    st.subheader("Download Results")
    
    # Convert results to CSV
    results_csv = comparison_df.to_csv()
    st.download_button(
        label="Download Results as CSV",
        data=results_csv,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )

else:
    st.subheader("Upload Custom Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")
            st.dataframe(custom_df.head())
            
            # Let user select target column
            target_col = st.selectbox("Select target column:", custom_df.columns)
            
            if st.button("Process Custom Dataset"):
                X_custom = custom_df.drop(target_col, axis=1)
                y_custom = custom_df[target_col]
                
                # Handle categorical variables
                label_encoders = {}
                for col in X_custom.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X_custom[col] = le.fit_transform(X_custom[col])
                    label_encoders[col] = le
                
                # Handle NaN values
                # Drop columns that are entirely NaN
                all_nan_cols = X_custom.columns[X_custom.isnull().all()].tolist()
                if all_nan_cols:
                    X_custom = X_custom.drop(columns=all_nan_cols)
                    st.warning(f"Dropped all-NaN columns: {all_nan_cols}")
                
                # Drop columns with >50% NaN
                high_nan_cols = X_custom.columns[X_custom.isnull().mean() > 0.5].tolist()
                if high_nan_cols:
                    X_custom = X_custom.drop(columns=high_nan_cols)
                    st.warning(f"Dropped high-NaN columns: {high_nan_cols}")
                
                # Drop rows with remaining NaN
                if X_custom.isnull().any().any():
                    before = len(X_custom)
                    valid_mask = ~X_custom.isnull().any(axis=1)
                    X_custom = X_custom[valid_mask]
                    y_custom = y_custom[valid_mask]
                    st.warning(f"Dropped {before - len(X_custom)} rows with missing values.")
                
                if len(X_custom) < 2:
                    st.error("Not enough samples after cleaning. Please fix your dataset.")
                else:
                    # Split and scale
                    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_custom, y_custom, test_size=0.15, random_state=42)
                    
                    scaler_c = StandardScaler()
                    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
                    X_test_c_scaled = scaler_c.transform(X_test_c)
                    
                    st.success("Custom dataset processed successfully! You can now run the models.")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.markdown("**Implementation for learning**")
st.markdown("6 ML Classification Models with Comprehensive Evaluation Metrics")
