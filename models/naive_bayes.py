from sklearn.naive_bayes import GaussianNB
from models.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """Gaussian Naive Bayes classifier wrapper."""

    def __init__(self):
        super().__init__(name="Naive Bayes", needs_scaling=True)

    def _create_model(self):
        return GaussianNB(var_smoothing=1e-9)
