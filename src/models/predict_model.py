# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")


class ChurnPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

    def predict(self, input_data: pd.DataFrame):
        input_data = input_data.copy()

        if "customerID" in input_data.columns:
            input_data = input_data.drop(columns=["customerID"])

        input_data["TotalCharges"] = pd.to_numeric(
            input_data["TotalCharges"],
            errors="coerce"
        )

        processed_data = self.preprocessor.transform(input_data)

        feature_names = self.preprocessor.get_feature_names_out()
        processed_data = pd.DataFrame(processed_data, columns=feature_names)

        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)[:, 1]

        return prediction, probability


if __name__ == "__main__":
    sample = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }])

    predictor = ChurnPredictor()
    pred, proba = predictor.predict(sample)

    print("Prediction:", pred[0])
    print("Churn Probability:", proba[0])