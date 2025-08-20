import os
import pickle
import pandas as pd

# Current file directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up three levels to project root, then into artifacts
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "..", "..", "artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join(CURRENT_DIR, "..", "..", "..", "artifacts", "preprocessor.pkl")

class PredictionPipeline:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            self.preprocessor = pickle.load(f)

    def predict(self, input_data):
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Input should be a dictionary or pandas DataFrame")

        processed_data = self.preprocessor.transform(input_df)
        predictions = self.model.predict(processed_data)
        return predictions
