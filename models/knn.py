from sklearn.neighbors import KNeighborsClassifier
from models.base_model import BaseModel


class KNNModel(BaseModel):
    """K-Nearest Neighbor classifier wrapper."""

    def __init__(self):
        super().__init__(name="K-Nearest Neighbor", needs_scaling=True)

    def _create_model(self):
        return KNeighborsClassifier()
