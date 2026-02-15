from sklearn.tree import DecisionTreeClassifier
from models.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree classifier wrapper."""

    def __init__(self):
        super().__init__(name="Decision Tree", needs_scaling=False)

    def _create_model(self):
        return DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
        )
