from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_transformation import DataTransformation
from src.ml_project.exception import CustomException


if __name__ == '__main__':
    try:
        raw_df, train_df, test_df = DataIngestion().getting_mysql_data()
        DataTransformation().get_data_for_transformation(train_df, test_df)
    except Exception as e:
        raise CustomException(e)