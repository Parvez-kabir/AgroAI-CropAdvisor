from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# -----------------------------
# Load model and encoder
# -----------------------------
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Crop Recommendation API",
    description="Predict the most suitable crop based on soil and weather data",
    version="1.0"
)

# -----------------------------
# Input schema
# -----------------------------
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "Crop Recommendation API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_crop(data: CropInput):
    input_data = np.array([[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    # prediction
    pred = model.predict(input_data)
    crop_name = le.inverse_transform(pred)[0]

    # confidence
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(float(np.max(probabilities)) * 100, 2)

    return {
        "recommended_crop": crop_name,
        "confidence_percent": confidence
    }
