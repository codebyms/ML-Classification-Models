from xgboost import XGBClassifier
from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting classifier wrapper."""

    def __init__(self):
        super().__init__(name="XGBoost", needs_scaling=False)

    def _create_model(self):
        return XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=2.0,
            reg_lambda=2.0,
            min_child_weight=3,
            gamma=0.3,
        )
