from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.models.predict_model import ChurnPredictor


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting whether a customer will churn or not.",
    version="1.0.0"
)

predictor = None


def get_predictor():
    global predictor

    if predictor is None:
        predictor = ChurnPredictor()

    return predictor


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])

    model_predictor = get_predictor()
    prediction, probability = model_predictor.predict(input_df)

    result = "Churn" if prediction[0] == 1 else "Not Churn"

    return {
        "prediction": int(prediction[0]),
        "result": result,
        "churn_probability": float(probability[0])
    }