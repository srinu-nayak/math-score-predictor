import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass

from src.mlproject.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def model_trainer(self, train_arr,test_arr):
        try:
            logging.info("Training model...")

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
            }

            params = {
                "Linear Regression": {},
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
            }

            evaluate_model(
                X_train, y_train, X_test, y_test, models, params
            )





        except Exception as e:
            raise CustomException(e)
