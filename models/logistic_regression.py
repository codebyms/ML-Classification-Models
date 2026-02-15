from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier wrapper."""

    def __init__(self):
        super().__init__(name="Logistic Regression", needs_scaling=True)

    def _create_model(self):
        return LogisticRegression(random_state=42, max_iter=1000)
