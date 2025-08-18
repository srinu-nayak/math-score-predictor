from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
import os
from src.mlproject.utils import connecting_to_mysql_database
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def get_sql_data(self):
        
        try:
            
            logging.info(f"Ingecting the data from utils file")
            
            df = connecting_to_mysql_database()
            
            logging.info(f"saved rawdata")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)            
            df.to_csv(self.data_ingestion_config.raw_data_path, header=True, index= False)
            
            logging.info(f"splitting rawdata")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            logging.info(f"saved train data")
            train_df.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            
            logging.info(f"saved test data")
            test_df.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            
            return (
                self.data_ingestion_config.raw_data_path,
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e)
        

