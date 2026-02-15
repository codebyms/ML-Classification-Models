import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


class DataLoader:
    """Handles loading, preprocessing, and splitting datasets."""

    DATASETS = {
        "Breast Cancer": {
            "file": "breast_cancer_wisconsin.csv",
            "target": "target",
            "drop_cols": [],
            "target_labels": {0: "Benign", 1: "Malignant"},
        },
        "Iris": {
            "file": "iris.csv",
            "target": "target",
            "drop_cols": [],
            "target_labels": {0: "Setosa", 1: "Versicolor", 2: "Virginica"},
        },
        "Wine": {
            "file": "wine.csv",
            "target": "target",
            "drop_cols": [],
            "target_labels": {0: "Class 0", 1: "Class 1", 2: "Class 2"},
        },
    }

    def __init__(self):
        self.df = None
        self.dataset_name = ""
        self.X_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.target_labels = {}

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load a built-in sklearn dataset."""
        if name not in self.DATASETS:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self.DATASETS.keys())}")

        config = self.DATASETS[name]
        self.dataset_name = name
        self.target_labels = config["target_labels"]

        # Load sklearn built-in datasets
        if name == "Breast Cancer":
            data = load_breast_cancer()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            # sklearn uses 0=malignant,1=benign; flip to 0=Benign,1=Malignant
            # so it matches CSV LabelEncoder output (B=0, M=1)
            self.df["target"] = 1 - data.target
            
        elif name == "Iris":
            data = load_iris()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df["target"] = data.target
            
        elif name == "Wine":
            data = load_wine()
            self.df = pd.DataFrame(data.data, columns=data.feature_names)
            self.df["target"] = data.target

        return self.df

    def _encode_and_rename_target(self, target_col: str):
        """Encode categorical columns and rename target to 'target'."""
        # Encode categorical feature columns
        feature_cols = [c for c in self.df.columns if c != target_col]
        for col in self.df[feature_cols].select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])

        # Encode target if categorical
        if self.df[target_col].dtype == "object":
            le = LabelEncoder()
            self.df[target_col] = le.fit_transform(self.df[target_col])
            # Store mapping: encoded int -> original label
            if not self.target_labels:
                # Map B/M to Benign/Malignant for proper representation
                class_mapping = {}
                for i, label in enumerate(le.classes_):
                    if label.upper() == 'B':
                        class_mapping[i] = "Benign"
                    elif label.upper() == 'M':
                        class_mapping[i] = "Malignant"
                    else:
                        class_mapping[i] = label
                self.target_labels = class_mapping

        # Rename for consistency
        if target_col != "target":
            self.df = self.df.rename(columns={target_col: "target"})

    def load_custom(self, file, target_col: str) -> pd.DataFrame:
        """Load a user-uploaded CSV and encode categorical features."""
        self.df = pd.read_csv(file)
        self.dataset_name = "Custom Dataset"

        # Drop common ID-like columns that shouldn't be features
        id_cols = [c for c in self.df.columns
                   if c.lower() in ("id", "unnamed: 32") or c.lower().startswith("unnamed")]
        id_cols = [c for c in id_cols if c != target_col]
        if id_cols:
            self.df = self.df.drop(columns=id_cols, errors="ignore")

        self._encode_and_rename_target(target_col)
        return self.df

    def prepare(self, test_size: float = 0.15, random_state: int = 42):
        """Split data and create scaled versions."""
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_builtin or load_custom first.")

        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        # Handle NaN values (important for custom CSV uploads)
        if X.isnull().any().any():
            # Drop columns that are entirely NaN
            all_nan_cols = X.columns[X.isnull().all()].tolist()
            if all_nan_cols:
                X = X.drop(columns=all_nan_cols)

            # Drop columns with >50% NaN
            high_nan_cols = X.columns[X.isnull().mean() > 0.5].tolist()
            if high_nan_cols:
                X = X.drop(columns=high_nan_cols)

            # Drop remaining rows with any NaN
            valid_mask = ~X.isnull().any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            self.df = pd.concat([X, y], axis=1)

        if len(X) < 2:
            raise ValueError(
                "Not enough samples after removing missing values. "
                "Please clean your dataset before uploading."
            )

        stratify = y if y.nunique() > 1 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index,
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index,
        )
