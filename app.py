from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipiline


application = Flask(__name__)

app = application

# Route for Home page

@app.route('/')
def index():
    return render_template('index.html')

# Route for Prediction page

@app.route('/predictscore', methods=['GET', 'POST'])
def predict_score():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            batting_team = request.form.get('batting_team'),
            bowling_team = request.form.get('bowling_team'),
            city = request.form.get('city'),
            current_score = request.form.get('current_score'),
            overs = request.form.get('overs'),
            wickets = request.form.get('wickets'), 
            last_five = request.form.get('last_five'),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipiline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results = str(int(results[0])))
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)