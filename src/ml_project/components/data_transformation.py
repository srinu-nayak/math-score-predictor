import pandas as pd

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging

class DataTransformation:
    def __init__(self):
        pass

    def get_data_for_transformation(self, train_df, test_df):
        try:
            train_df = pd.read_csv(train_df)
            test_df = pd.read_csv(test_df)

            print(train_df.head())
            print(test_df.shape)

        except Exception as e:
            raise CustomException(e)