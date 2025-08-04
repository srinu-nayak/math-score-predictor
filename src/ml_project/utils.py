from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import os
import mysql.connector
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import pickle
host = os.getenv("host")
password = os.getenv("password")
username = os.getenv("user")
database = os.getenv("database")


def get_data_from_database():
    try:
        mydb = mysql.connector.connect(
            host=host,
            user=username,
            passwd=password,
            database=database,
        )

        print("Connection established successfully!")
        logging.info("Connection established successfully!")
        df = pd.read_sql_query("SELECT * FROM students", mydb)
        return df

    except Exception as e:
        raise CustomException(e)

def save_object(filename, obj):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e)