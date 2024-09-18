import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import sys  # Untuk menangani sys.exit()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Load the dataset
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'

if os.path.exists(DATASET_PATH):
    crop_data = pd.read_csv(DATASET_PATH)
else:
    print(f"File not found at: {DATASET_PATH}")
    sys.exit()  # Keluar dari script jika file tidak ditemukan

# Prepare the dataset
y = crop_data['label']
x = crop_data.drop(['label'], axis=1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# Define models
models = {
    "KNN": KNeighborsClassifier(),
    "DT": DecisionTreeClassifier(),
    "RFC": RandomForestClassifier(),
    "GBC": GradientBoostingClassifier(),
    "XGB": XGBClassifier()
}

# Store accuracy and confusion matrices for each model
model_results = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Decode the predictions for evaluation
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    # Generate classification report and confusion matrix
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    print(f"\nConfusion Matrix for {model_name}:")
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    print(cm)
    
    # Store results for comparison
    model_results[model_name] = {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

# Determine the best model based on accuracy
best_model_name = max(model_results, key=lambda x: model_results[x]["accuracy"])
best_model = model_results[best_model_name]["model"]
best_accuracy = model_results[best_model_name]["accuracy"]

print(f"\nThe best model is {best_model_name} with an accuracy of {best_accuracy:.4f}")

# Example of using the best model for prediction with correct feature names
feature_cols = ["N", "P", "K", "Temperature", "Humidity", "ph", "Rainfall"]
new_data = pd.DataFrame([[90, 40, 40, 20, 80, 7, 200]], columns=feature_cols)  # Example input with column names
prediction = best_model.predict(new_data)
prediction_decoded = label_encoder.inverse_transform(prediction)
print(f"The suggested crop for the given climatic condition is: {prediction_decoded[0]}")

# Save the decoded prediction to use as input for the next model
next_model_input = prediction_decoded[0]
print("Input for the next model (label):", next_model_input)

# ---------------------------------
# XGBoost Model and Recommendation Part
# ---------------------------------

# Define the clusters (assuming 'label' is the correct column to group by)
clusters = crop_data.groupby("label").mean()

# Prepare the data for XGBoost model
xgb_model4 = XGBClassifier()

# Encode the labels (y) for XGBoost
label_encoder_xgb = LabelEncoder()
y_encoded_xgb = label_encoder_xgb.fit_transform(crop_data["label"])

# Scale the data using StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(crop_data[feature_cols])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y_encoded_xgb, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model4.fit(X_train, y_train)

# Save the model and scaler using pickle
with open('xgb_model4.pkl', 'wb') as model_file:
    pickle.dump(xgb_model4, model_file)

with open('scaler2.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Define the recommendation function
def get_recommendation(N, P, K, Temperature, Humidity, ph, Rainfall, next_model_input):
    # Load the scaler and model
    with open('scaler2.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    with open('xgb_model4.pkl', 'rb') as model_file:
        xgb_model = pickle.load(model_file)
    
    # Create a DataFrame for the input with feature names
    input_data = pd.DataFrame([[N, P, K, Temperature, Humidity, ph, Rainfall]], columns=feature_cols)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Check if next_model_input (predicted crop) is valid and available in the dataset
    if next_model_input in clusters.index:
        # Get the ideal ranges for the predicted crop from Model 1 (next_model_input)
        ideal_ranges = clusters.loc[next_model_input]
    else:
        return f"Error: The predicted crop '{next_model_input}' is not available in the dataset."

    # Predict the crop label using the XGBoost model
    predicted_label_encoded = xgb_model.predict(input_data_scaled)[0]
    
    # Decode the predicted label back to the original label
    predicted_label = label_encoder_xgb.inverse_transform([predicted_label_encoded])[0]
    
    # Create a recommendation string based on the comparison between input values and ideal ranges
    recommendation = []
    for parameter, value in zip(feature_cols, [N, P, K, Temperature, Humidity, ph, Rainfall]):
        if value < ideal_ranges[parameter] - 10:
            recommendation.append(f"{parameter} kekurangan, silakan tambahkan.")
        elif value > ideal_ranges[parameter] + 10:
            recommendation.append(f"{parameter} kelebihan, silakan kurangi.")
        else:
            recommendation.append(f"{parameter} sesuai dengan standar nasional.")

    # Add information about the suggested crop from the previous model
    recommendation.append(f"Tanaman yang disarankan berdasarkan model sebelumnya adalah: {next_model_input}")
    
    return "\n".join(recommendation)

# Test the recommendation function using the predicted crop from the best model
print(get_recommendation(90, 40, 40, 20, 80, 7, 200, next_model_input))
