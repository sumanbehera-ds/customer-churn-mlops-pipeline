import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.exceptions import InconsistentVersionWarning

import app as app_module
from src.models.predict_model import ChurnPredictor, ModelArtifactError


client = TestClient(app_module.app)


class DummyPredictor:
    def predict(self, input_df):
        return [1], [0.95]


VALID_PAYLOAD = {
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
    "TotalCharges": 29.85,
}


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn Prediction API is running"


def test_healthz_endpoint():
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    app_module.predictor = DummyPredictor()

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "result" in data
    assert "churn_probability" in data
    assert data["prediction"] in [0, 1]
    assert data["result"] in ["Churn", "Not Churn"]
    assert 0 <= data["churn_probability"] <= 1


def test_predict_missing_required_field():
    app_module.predictor = DummyPredictor()

    payload = VALID_PAYLOAD.copy()
    payload.pop("tenure")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_numeric_type():
    app_module.predictor = DummyPredictor()

    payload = VALID_PAYLOAD.copy()
    payload["MonthlyCharges"] = "invalid_amount"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_real_predictor_loads_model_artifacts():
    input_df = pd.DataFrame([VALID_PAYLOAD])

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model_predictor = ChurnPredictor()

    prediction, probability = model_predictor.predict(input_df)

    assert prediction[0] in [0, 1]
    assert 0 <= probability[0] <= 1


def test_predictor_reports_missing_model_artifact(tmp_path):
    with pytest.raises(ModelArtifactError, match="Unable to load churn model artifact"):
        ChurnPredictor(
            model_path=tmp_path / "missing.ubj",
            fallback_model_path=tmp_path / "missing.pkl",
            preprocessor_path=tmp_path / "missing_preprocessor.pkl",
        )
