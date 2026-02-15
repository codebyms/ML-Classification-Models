from sklearn.ensemble import RandomForestClassifier
from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest ensemble classifier wrapper."""

    def __init__(self):
        super().__init__(name="Random Forest", needs_scaling=False)

    def _create_model(self):
        return RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
        )
