from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer



if __name__ == "__main__":
    try:
        raw_df, train_df, test_df = DataIngestion().get_sql_data()

        logging.info("train_df,test_df sendind to data transformation process")
        train_arr,test_arr = DataTransformation().started_data_transformation(train_df, test_df)

        logging.info("train_arr,test_arr sendind to model training process")
        ModelTrainer().model_trainer(train_arr,test_arr)

        
    except Exception as e:
        raise CustomException(e)