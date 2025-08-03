import os
from dataclasses import dataclass
from src.ml_project.exception import CustomException
from src.ml_project.utils import get_data_from_database
from sklearn.model_selection import train_test_split
from src.ml_project.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def getting_mysql_data(self):
        try:
            df = get_data_from_database()

            logging.info("data splitted into train and test set")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("splitted data saved into the artifacts folder")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            train_df.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            return (
                self.data_ingestion_config.raw_data_path,
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )


        except Exception as e:
            raise CustomException(e)