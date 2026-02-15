import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


class Evaluator:
    """Computes classification evaluation metrics."""

    @staticmethod
    def compute_metrics(y_true, y_pred, y_pred_proba=None) -> dict:
        """Return a dict of all 6 evaluation metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        # AUC — handle binary vs multiclass
        auc = 0.0
        n_classes = len(np.unique(y_true))
        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(
                        y_true, y_pred_proba, multi_class="ovr", average="weighted"
                    )
            except ValueError:
                auc = 0.0

        return {
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
        }

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
        """Return the confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def compute_classification_report(y_true, y_pred, target_names=None) -> str:
        """Return the classification report as a string."""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    @staticmethod
    def best_models(results: dict, metric: str) -> tuple:
        """Return (best_value, list_of_model_names) for a given metric."""
        best_val = max(r[metric] for r in results.values())
        best_names = [name for name, r in results.items() if r[metric] == best_val]
        return best_val, best_names
