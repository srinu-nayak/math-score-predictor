from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig

if __name__ == "__main__":
    try:
        raw_df, train_df, test_df = DataIngestion().get_sql_data()
        
        print(train_df)

        
    except Exception as e:
        raise CustomException(e)