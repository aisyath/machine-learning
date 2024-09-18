import numpy as np
import joblib

# Paths
predict_model_path = 'xgb_model3.pkl'
recommendation_model_path = 'svm_model2.pkl'
scaler_path = 'scaler2.pkl'

# Load models and scaler
predict_model = joblib.load(predict_model_path)
recommendation_model = joblib.load(recommendation_model_path)
scaler = joblib.load(scaler_path)

# Sample input
input_data = np.array([[1.0, 2.0, 3.0, 25.0, 6.5, 60.0, 100.0]])
scaled_data = scaler.transform(input_data)

# Predict with the first model (XGBoost)
initial_prediction = predict_model.predict(scaled_data)
print(f"Initial Prediction: {initial_prediction}")

# Don't combine initial prediction, pass original scaled data
# Predict recommendation using the original 7 features (scaled_data)
recommendation_prediction = recommendation_model.predict(scaled_data)
print(f"Recommendation: {recommendation_prediction}")
