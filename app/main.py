import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Dict

app = FastAPI()

# Load Model 1 (XGBoost)
with open("xgb_model4.pkl", 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# Load Model 2 (SVM) and Scaler
# with open('svm_model2.pkl', 'rb') as model_file:
#     svm_model = pickle.load(model_file)

with open('scaler2.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load Label Encoder
label_encoder = LabelEncoder()
# Assuming you have the list of unique crop labels
unique_labels = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
label_encoder.fit(unique_labels)

# Load cluster data
clusters = pd.read_csv("Crop_Recommendation.csv").groupby("label").mean()

# Define input model with individual fields
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    Temperature: float
    Humidity: float
    ph: float
    Rainfall: float

class ModelOutput(BaseModel):
    model1_output: str
    model2_recommendation: str

# Route for POST method to predict using models
@app.post("/predict", response_model=ModelOutput)
async def predict(crop_input: CropInput):
    try:
        # Prepare input for Model 1 (XGBoost)
        model1_input = np.array([[crop_input.N, crop_input.P, crop_input.K,
                                  crop_input.Temperature, crop_input.Humidity,
                                  crop_input.ph, crop_input.Rainfall]])
        
        # Predict using Model 1 (XGBoost)
        model1_prediction = xgb_model.predict(model1_input)
        model1_output = label_encoder.inverse_transform(model1_prediction)[0]

        # Get ideal ranges for the predicted crop
        if model1_output not in clusters.index:
            raise HTTPException(status_code=404, detail=f"Predicted crop '{model1_output}' not found in clusters.")

        ideal_ranges = clusters.loc[model1_output]

        # Create recommendation based on ideal ranges
        feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph", "Rainfall"]
        recommendation = []
        for parameter, value in zip(feature_cols, model1_input[0]):
            if value < ideal_ranges[parameter] - 10:
                recommendation.append(f"{parameter} kekurangan, silakan tambahkan.")
            elif value > ideal_ranges[parameter] + 10:
                recommendation.append(f"{parameter} kelebihan, silakan kurangi.")
            else:
                recommendation.append(f"{parameter} sesuai dengan standar nasional.")

        # Add crop suggestion from Model 1
        recommendation.append(f"Tanaman yang disarankan berdasarkan model sebelumnya adalah: {model1_output}")
        model2_recommendation = "\n".join(recommendation)

        return ModelOutput(model1_output=model1_output, model2_recommendation=model2_recommendation)
    
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))

# Route for GET method to return some backend info or state
@app.get("/data", response_model=Dict[str, float])
async def get_cluster_averages():
    # Return the average of cluster values for all labels
    cluster_means = clusters.mean().to_dict()
    return cluster_means

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
