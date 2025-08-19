import numpy as np

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
import os
import pandas as pd
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.mlproject.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_config = DataTransformationConfig()

    def transforming_data(self, input_feature_train_df):
        try:

            onehot_pipe = Pipeline(steps=[
                ("onehot_encoder",OneHotEncoder(handle_unknown="ignore", drop="first")),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ])

            ordinal_pipe = Pipeline(steps=[
                ("ordinal_encoder", OrdinalEncoder(categories=[["high school", "some high school", "some college", "bachelor's degree","associate's degree", "master's degree"]])),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ])

            numerical_pipe = Pipeline(steps=[
                ("numerical_pipe", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            ct_preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot_encoder", onehot_pipe, [0,1,3,4]),
                    ("ordinal_encoder", ordinal_pipe, [2]),
                    ("numerical_pipe", numerical_pipe, [5,6]),
                ],
                remainder="passthrough",
            )

            return ct_preprocessor

        except Exception as e:
            raise CustomException(e)

    def started_data_transformation(self, train_df, test_df):
        try:
            logging.info(f"getting train_df, test_df for transformation from app file")

            train_df = pd.read_csv(train_df)
            test_df = pd.read_csv(test_df)

            logging.info(f"seperating target feature from train_df")
            input_feature_train_df = train_df.drop(columns=["math_score"])
            target_feature_train_df = train_df["math_score"]
            target_feature_train_df = target_feature_train_df.values.reshape(-1,1)

            logging.info(f"seperating target feature from test_df")
            input_feature_test_df = test_df.drop(columns=["math_score"])
            target_feature_test_df = test_df["math_score"]
            target_feature_test_df = target_feature_test_df.values.reshape(-1,1)

            preprocessor_object = self.transforming_data(input_feature_train_df)


            logging.info(f"preprocessor object created")
            input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)

            logging.info(f"preprocessor object transformed")
            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_df), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_df), axis=1)


            logging.info(f"preprocessor object sending to utils file")

            save_object(
                self.preprocessor_obj_config.preprocessor_obj,
                preprocessor_object,
            )

            return (
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise CustomException(e)

