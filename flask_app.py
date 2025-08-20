from flask import Flask, render_template, request
import pandas as pd
from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)
pipeline = PredictionPipeline()  # initialize the pipeline once

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # 1. Get form data
        data = {
            "gender": request.form.get("gender"),
            "race_ethnicity": request.form.get("race_ethnicity"),
            "parental_level_of_education": request.form.get("parental_level_of_education"),
            "lunch": request.form.get("lunch"),
            "test_preparation_course": request.form.get("test_preparation_course"),            "reading_score": int(request.form.get("reading_score")),
            "writing_score": int(request.form.get("writing_score"))
        }

        # 2. Get prediction
        prediction = pipeline.predict(data)[0]  # get first element if it's an array

        # 3. Return result in template
        return render_template("home.html", prediction_text=f"Predicted Result: {prediction}")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)