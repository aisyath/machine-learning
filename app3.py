from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

# Define the input model for FastAPI
class InputData(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    pH: float
    Rainfall: float

# Load the XGBoost and SVM models and the scaler
with open('xgb_model.pkl', 'rb') as xgb_file:
    xgb_model = pickle.load(xgb_file)

with open('svm_model.pkl', 'rb') as svm_file:
    svm_model = pickle.load(svm_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the clusters data for SVM recommendations
data = pd.read_csv("Crop_recommendation.csv")
clusters = data.groupby("label").mean()

# Define feature columns for scaling
feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph", "Rainfall"]

# Initialize FastAPI app
app = FastAPI()

# Route for prediction and recommendation
@app.post("/predict")
def predict_crop(input_data: InputData):
    # Prepare data for XGBoost model
    xgb_input = np.array([[input_data.N, input_data.P, input_data.K, 
                           input_data.Temperature, input_data.Humidity, 
                           input_data.pH, input_data.Rainfall]])
    
    # Predict the crop using XGBoost model
    xgb_prediction = xgb_model.predict(xgb_input)
    crop_suggested = xgb_prediction[0]  # The crop label from XGBoost
    
    # Use the suggested crop as input for the SVM model
    input_scaled = scaler.transform([[input_data.N, input_data.P, input_data.K, 
                                      input_data.Temperature, input_data.Humidity, 
                                      input_data.pH, input_data.Rainfall]])
    
    # Predict the cluster for SVM model
    predicted_cluster = svm_model.predict(input_scaled)[0]
    
    # Get the ideal ranges for the predicted cluster
    ideal_ranges = clusters.loc[predicted_cluster]
    
    # Generate recommendation
    recommendation = []
    for parameter, value in zip(feature_cols, [input_data.N, input_data.P, input_data.K, 
                                               input_data.Temperature, input_data.Humidity, 
                                               input_data.pH, input_data.Rainfall]):
        if value < ideal_ranges[parameter] - 10:
            recommendation.append(f"{parameter} kekurangan, silakan tambahkan.")
        elif value > ideal_ranges[parameter] + 10:
            recommendation.append(f"{parameter} kelebihan, silakan kurangi.")
        else:
            recommendation.append(f"{parameter} sesuai dengan standar nasional.")
    
    # Add the crop suggestion from the previous model
    recommendation.append(f"Tanaman yang disarankan berdasarkan model XGBoost adalah: {crop_suggested}")
    
    return {"crop_suggested": crop_suggested, "recommendation": "\n".join(recommendation)}

# Run the FastAPI application:
# Save this file as `main.py`, then run the FastAPI app with the following command:
# uvicorn main:app --reload
