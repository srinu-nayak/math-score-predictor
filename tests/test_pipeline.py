# tests/test_pipeline.py
from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline

def test_pipeline_runs():
    pipeline = PredictionPipeline()
    sample_data = {
        "gender": "female",
        "race_ethnicity": "group A",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": 70,
        "writing_score": 75
    }
    result = pipeline.predict(sample_data)
    assert result is not None