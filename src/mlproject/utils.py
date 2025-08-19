import pickle

import mysql.connector
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

username = os.getenv("DB_USER")
hostname = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
password = os.getenv("DB_PASSWORD")




def connecting_to_mysql_database():
    try:
        logging.info("Connecting to MySQL database...")

        # Connect to MySQL
        mydb = mysql.connector.connect(
            host=hostname,
            user=username,
            password=password,
            database=database
        )

        logging.info("Database connection is successful!")

        # Fetch data into a DataFrame
        df = pd.read_sql_query("SELECT * FROM students", mydb)
        # print(df.head(5))

        # Close connection
        mydb.close()
        logging.info("Database connection closed.")
        return df

    except Exception as e:
        raise CustomException(e)

def save_object(preprocessor_obj,preprocessor_object):
    try:

        os.makedirs(os.path.dirname(preprocessor_obj), exist_ok=True)
        with open(preprocessor_obj, "wb") as f:
            pickle.dump(preprocessor_object, f)

    except Exception as e:
        raise CustomException(e)