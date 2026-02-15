from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all classification models."""

    def __init__(self, name: str, needs_scaling: bool = False):
        self.name = name
        self.needs_scaling = needs_scaling
        self._model = None

    @abstractmethod
    def _create_model(self):
        """Return a fresh sklearn-compatible estimator."""
        pass

    def fit(self, X_train, y_train):
        """Build the model from scratch and train it."""
        self._model = self._create_model()
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        """Return class predictions."""
        return self._model.predict(X_test)

    def predict_proba(self, X_test):
        """Return probability predictions if supported, else None."""
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X_test)
        return None
