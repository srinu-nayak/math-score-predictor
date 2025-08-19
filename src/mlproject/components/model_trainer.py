import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass

from src.mlproject.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    model_path:str = os.path.join("artifacts", "model.pkl")

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
                "Decision Tree": DecisionTreeRegressor(),
            }

            params = {
                "Linear Regression": {},
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                }
            }

            r2_score_report, best_params_report, best_estimator_report = evaluate_model(
                X_train, y_train, X_test, y_test, models, params
            )

            best_model_score = max(sorted(r2_score_report.values()))
            logging.info("Best model score: {}".format(best_model_score))
            print("Best model score: {}".format(best_model_score))

            best_model_name = list(r2_score_report.keys())[list(r2_score_report.values()).index(best_model_score)]
            best_model = best_estimator_report[best_model_name]
            logging.info("Best model name: {}".format(best_model))
            print("Best model name: {}".format(best_model_name))

            best_hyperparameters = best_params_report[best_model_name]
            logging.info("Best hyperparameters: {}".format(best_hyperparameters))
            print("Best hyperparameters: {}".format(best_hyperparameters))

            save_object(
                self.model_trainer_config.model_path,
                best_model
            )


        except Exception as e:
            raise CustomException(e)
