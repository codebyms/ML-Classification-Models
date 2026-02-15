from models.data_loader import DataLoader
from models.evaluator import Evaluator
from models.base_model import BaseModel


class Pipeline:
    """Orchestrates model training and evaluation."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.evaluator = Evaluator()
        self.results: dict[str, dict] = {}

    def run_all(self, all_models: dict) -> dict[str, dict]:
        """Train and evaluate every registered model. Returns results dict."""
        self.results = {}
        for name, model_cls in all_models.items():
            result = self.run_single(model_cls)
            self.results[name] = result
        return self.results

    def run_single(self, model_cls: type) -> dict:
        """Instantiate, train, predict, and evaluate a single model class.

        Returns a dict with keys: metrics, y_pred, y_test, confusion_matrix,
        classification_report.
        """
        model: BaseModel = model_cls()

        # Pick scaled or unscaled data based on model requirement
        if model.needs_scaling:
            X_train = self.data_loader.X_train_scaled
            X_test = self.data_loader.X_test_scaled
        else:
            X_train = self.data_loader.X_train
            X_test = self.data_loader.X_test

        y_train = self.data_loader.y_train
        y_test = self.data_loader.y_test

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        return {
            "metrics": self.evaluator.compute_metrics(y_test, y_pred, y_pred_proba),
            "y_pred": y_pred,
            "y_test": y_test,
            "confusion_matrix": self.evaluator.compute_confusion_matrix(y_test, y_pred),
            "classification_report": self.evaluator.compute_classification_report(
                y_test, y_pred,
                target_names=self._get_target_names(),
            ),
        }

    def _get_target_names(self):
        """Return target label names from the data loader, or None."""
        labels = self.data_loader.target_labels
        if labels:
            return [labels[k] for k in sorted(labels.keys())]
        return None
