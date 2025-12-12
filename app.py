from flask import Flask, request, render_template
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app=application

## rout for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Validate all required fields are present and not empty
            gender = request.form.get('gender', '').strip()
            race_ethnicity = request.form.get('race_ethnicity', '').strip()
            parental_level_of_education = request.form.get('parental_level_of_education', '').strip()
            lunch = request.form.get('lunch', '').strip()
            test_preparation_course = request.form.get('test_preparation_course', '').strip()
            reading_score = request.form.get('reading_score', '').strip()
            writing_score = request.form.get('writing_score', '').strip()
            
            # Check if any field is empty
            if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score]):
                return render_template('home.html', error="Please fill all fields"), 400
            
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=float(reading_score),
                writing_score=float(writing_score),
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        except Exception as e:
            return render_template('home.html', error=str(e)), 500

if __name__=="__main__":
    app.run(host="127.0.0.1", port=8080)
