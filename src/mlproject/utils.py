import pickle

import mysql.connector
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score
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

def save_object(filename,obj):
    try:

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e)

def evaluate_model(X_train, y_train, X_test, y_test, models:dict, params:dict):
    try:

        logging.info("Evaluating model...")

        r2_score_report ={}
        best_params_report = {}
        best_estimator_report = {}

        for model_name, model in models.items():

            param_grid = params.get(model_name, {})

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='r2')

            grid_search.fit(X_train, y_train)
            logging.info("GridSearchCV fitted!")

            y_test_pred = grid_search.predict(X_test)

            r2_score_test_model = r2_score(y_test,y_test_pred)
            r2_score_report[model_name] = r2_score_test_model

            best_params = grid_search.best_params_
            best_params_report[model_name] = best_params

            best_estimators = grid_search.best_estimator_
            best_estimator_report[model_name] = best_estimators


            return r2_score_report, best_params_report, best_estimator_report









    except Exception as e:
        raise CustomException(e)