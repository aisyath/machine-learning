import os
import numpy as np
import joblib as joblib
from flask import Flask, request, render_template
# from fastapi import FastAPI, request, render_template

# Check if the files exist before loading
recommendation_model_path = 'svm_model.pkl'
predict_model_path = 'predict.pkl'
scaler_path = 'scaler.save'

if not os.path.exists(predict_model_path):
    raise FileNotFoundError(f"Model file not found: {predict_model_path}")
if not os.path.exists(recommendation_model_path):
    raise FileNotFoundError(f"Model file not found: {recommendation_model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

# Load the models and scaler
predict_model = joblib.load(predict_model_path)
recommendation_model = joblib.load(recommendation_model_path)
scaler = joblib.load(scaler_path)

app = Flask(__name__)

# Configure image folder
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input values from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        Temperature = float(request.form['Temperature'])
        ph = float(request.form['ph'])
        Humidity = float(request.form['Humidity'])
        Rainfall = float(request.form['Rainfall'])
        
        # Prepare input data for the models
        input_data = np.array([[N, P, K, Temperature, ph, Humidity, Rainfall]])
        scaled_data = scaler.transform(input_data)

        # Predict with the initial model
        initial_prediction = predict_model.predict(scaled_data)
        print(f'Initial prediction: {initial_prediction}')

        # Generate recommendation using the LightGBM model
        recommendation_prediction = recommendation_model.predict(scaled_data)
        print(f'Recommendation prediction: {recommendation_prediction}')

        # Construct image path based on prediction (assuming image names match predictions)
        image = f'{initial_prediction[0]}.png'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image)

        return render_template('index.html', prediction=initial_prediction[0], 
                               recommendation=recommendation_prediction[0], 
                               image=image_path)

if __name__ == '__main__':
    app.run(debug=True)
