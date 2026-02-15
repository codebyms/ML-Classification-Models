from models.data_loader import DataLoader
from models.evaluator import Evaluator
from models.pipeline import Pipeline
from models.logistic_regression import LogisticRegressionModel
from models.decision_tree import DecisionTreeModel
from models.knn import KNNModel
from models.naive_bayes import NaiveBayesModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel

ALL_MODELS = {
    "Logistic Regression": LogisticRegressionModel,
    "Decision Tree": DecisionTreeModel,
    "K-Nearest Neighbor": KNNModel,
    "Naive Bayes": NaiveBayesModel,
    "Random Forest (Ensemble)": RandomForestModel,
    "XGBoost (Ensemble)": XGBoostModel,
}
