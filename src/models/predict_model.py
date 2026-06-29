# -*- coding: utf-8 -*-
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.ubj")
JOBLIB_MODEL_PATH = os.getenv("JOBLIB_MODEL_PATH", "models/model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")


class ModelArtifactError(RuntimeError):
    """Raised when a required model artifact cannot be loaded."""


class NativeXGBoostModel:
    """Small adapter that gives a native XGBoost Booster a classifier API."""

    def __init__(self, model_path):
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)

    def predict_proba(self, input_data):
        probabilities = np.asarray(
            self.booster.predict(xgb.DMatrix(input_data))
        )
        probabilities = probabilities.reshape(-1)

        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, input_data):
        return (self.predict_proba(input_data)[:, 1] >= 0.5).astype(int)


class ChurnPredictor:
    def __init__(
        self,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        fallback_model_path=JOBLIB_MODEL_PATH
    ):
        self.model = self._load_model(model_path, fallback_model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)

    @staticmethod
    def _load_model(model_path, fallback_model_path):
        model_paths = [Path(model_path)]
        fallback_path = Path(fallback_model_path)

        if fallback_path not in model_paths:
            model_paths.append(fallback_path)

        errors = []

        for path in model_paths:
            if not path.exists():
                errors.append(f"{path}: file not found")
                continue

            try:
                if path.suffix.lower() in {".ubj", ".json"}:
                    return NativeXGBoostModel(path)

                return joblib.load(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        raise ModelArtifactError(
            "Unable to load churn model artifact. Tried "
            + "; ".join(errors)
        )

    @staticmethod
    def _load_preprocessor(preprocessor_path):
        path = Path(preprocessor_path)

        if not path.exists():
            raise ModelArtifactError(
                f"Unable to load preprocessor artifact. {path} was not found."
            )

        try:
            return joblib.load(path)
        except Exception as exc:
            raise ModelArtifactError(
                f"Unable to load preprocessor artifact from {path}: {exc}"
            ) from exc

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
