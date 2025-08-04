import os
from src.ml_project.utils import save_object
import pandas as pd

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.preprocessing_obj_config = DataTransformationConfig()

    def started_data_transformation(self, input_features_train_df):

        onehotencoder_pipeline = Pipeline(steps=[
            ('onehotencoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

        ordinalencoder_pipeline = Pipeline(steps=[
            ('ordinalencoder', OrdinalEncoder()),
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

        numerical_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='median')),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("onehotencoder", onehotencoder_pipeline, [0, 1, 3, 4]),
            ("ordinalencoder", ordinalencoder_pipeline, [2]),
            ("numerical", numerical_pipeline, [5, 6]),
        ], remainder='passthrough')


        return preprocessor

    def get_data_for_transformation(self, train_df, test_df):
        try:
            train_df = pd.read_csv(train_df)
            test_df = pd.read_csv(test_df)

            logging.info("dividing independent and dependent variables of train and test data")
            input_features_train_df = train_df.drop(columns=["math_score"])
            target_feature_train_df = train_df["math_score"]
            target_feature_train_df = target_feature_train_df.values.reshape(-1, 1)


            input_features_test_df = test_df.drop(columns=["math_score"])
            target_feature_test_df = test_df["math_score"]
            target_feature_test_df = target_feature_test_df.values.reshape(-1, 1)
            
            preprocessor_object = self.started_data_transformation(input_features_train_df)

            input_features_train_arr = preprocessor_object.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_object.transform(input_features_test_df)

            train_arr = np.concatenate((input_features_train_arr, target_feature_train_df), axis=1)
            test_arr = np.concatenate((input_features_test_arr, target_feature_test_df), axis=1)

            logging.info("preprocessor object saved to artifacts")
            save_object(
                self.preprocessing_obj_config.preprocessing_obj_file_path,
                preprocessor_object
            )

            return train_arr, test_arr


        except Exception as e:
            raise CustomException(e)