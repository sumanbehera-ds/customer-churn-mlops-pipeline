# Customer Churn MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![DVC](https://img.shields.io/badge/DVC-Google%20Drive-purple)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)

End-to-end customer churn prediction project with a reproducible ML pipeline, DVC data versioning, MLflow experiment tracking, FastAPI inference, Docker packaging, monitoring reports, and GitHub Actions CI.

## Business Problem

Telecom companies lose revenue when customers churn. This project predicts whether a customer is likely to churn so retention teams can prioritize at-risk customers.

## What Is Included

- DVC pipeline for data ingestion, preprocessing, and model training
- Google Drive DVC remote for data/cache storage
- MLflow logging during model training
- XGBoost, Scikit-learn, and imbalanced-learn model workflow
- Committed production inference artifacts for the FastAPI app
- FastAPI endpoints for health checks and churn prediction
- Dockerfile for containerized deployment
- Monitoring report artifacts under `reports/`
- GitHub Actions CI with import, DVC dry-run, file, and pytest validation

## Repository Structure

```text
customer-churn-mlops-pipeline/
|-- app.py
|-- data/
|   |-- raw/
|   |   `-- customer_churn.csv.dvc
|   `-- processed/
|-- docs/
|-- models/
|   |-- model.ubj
|   |-- model.pkl
|   `-- preprocessor.pkl
|-- monitoring/
|   `-- drift_report.py
|-- reports/
|   |-- metrics.json
|   |-- monitoring_report.html
|   `-- monitoring_summary.json
|-- src/
|   |-- data/
|   |-- features/
|   `-- models/
|-- tests/
|-- Dockerfile
|-- Makefile
|-- dvc.yaml
|-- dvc.lock
|-- requirements.txt
`-- README.md
```

## Pipeline

The DVC pipeline has three stages:

```text
data/raw/customer_churn.csv
        |
        v
data_ingestion
        |
        v
data_preprocessing
        |
        v
model_training
        |
        v
models/model.ubj + models/model.pkl + reports/metrics.json
```

Run the pipeline:

```powershell
.\.venv\Scripts\dvc.exe repro
```

Check pipeline status:

```powershell
.\.venv\Scripts\dvc.exe status --no-updates
```

Expected clean result:

```text
Data and pipelines are up to date.
```

## Model Performance

Current best model from `reports/metrics.json`:

| Metric | Value |
| --- | ---: |
| Best model | `XGBClassifier_scale_pos_weight` |
| Accuracy | 0.747 |
| Precision | 0.516 |
| Recall | 0.794 |
| F1 score | 0.625 |
| ROC-AUC | 0.847 |

## Model Artifact Policy

The FastAPI app loads these files directly:

```text
models/model.ubj
models/preprocessor.pkl
```

They are intentionally committed so the API can start from a normal Git clone without requiring an immediate `dvc pull`.

`models/model.ubj` is the preferred native XGBoost runtime artifact. `models/model.pkl` is kept as a joblib fallback and compatibility artifact.

The raw and processed datasets are managed by DVC and are not committed directly.

## Local Setup

Clone the repo:

```powershell
git clone https://github.com/sumanbehera-ds/customer-churn-mlops-pipeline.git
cd customer-churn-mlops-pipeline
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest tests
```

Expected result:

```text
8 passed
```

Run the API:

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## DVC Data Setup

The default DVC remote is a Google Drive folder:

```text
gdrive_remote    gdrive://1h8ZNguYynL7jgp8KNCKtb2YmrB7sSd_T
```

Pull data and cached pipeline outputs:

```powershell
.\.venv\Scripts\dvc.exe pull
```

Expected result when everything is current:

```text
Everything is up to date.
```

### Credentials

DVC credentials are private and are not committed.

The committed `.dvc/config` only stores the shared remote URL and repo-local cache setting. Authentication mode is configured locally per developer.

Ignored local credential paths:

```text
.dvc/config.local
.dvc/secrets/
```

For the current Google Drive remote, OAuth is the recommended upload method because service accounts cannot upload to a normal "My Drive" folder unless a Shared Drive or delegated setup is used.

Maintainer OAuth setup:

```powershell
.\.venv\Scripts\dvc.exe remote modify --local gdrive_remote gdrive_use_service_account false
.\.venv\Scripts\dvc.exe remote modify --local gdrive_remote gdrive_client_id "<google-oauth-client-id>"
.\.venv\Scripts\dvc.exe remote modify --local gdrive_remote gdrive_client_secret "<google-oauth-client-secret>"
.\.venv\Scripts\dvc.exe remote modify --local gdrive_remote gdrive_user_credentials_file secrets/gdrive-user-credentials.json
```

Then run:

```powershell
.\.venv\Scripts\dvc.exe pull
.\.venv\Scripts\dvc.exe push
```

For service-account read access, place the JSON file here and share the Drive folder with the service-account email:

```text
.dvc/secrets/gdrive-service-account.json
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Home endpoint |
| GET | `/healthz` | Health check |
| POST | `/predict` | Predict customer churn |

Example request:

```json
{
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
}
```

Example response:

```json
{
  "prediction": 1,
  "result": "Churn",
  "churn_probability": 0.95
}
```

## Testing

The automated tests cover:

- Home endpoint
- Health endpoint
- Prediction endpoint response shape
- Missing required fields
- Invalid numeric payload validation
- Real predictor loading committed model artifacts
- Clear error reporting for missing model artifacts
- Numeric PSI drift score calculation

Run:

```powershell
python -m pytest tests
```

## CI/CD

GitHub Actions runs on pushes and pull requests to `main`.

The CI workflow:

- Installs `requirements.txt`
- Verifies imports for `pandas`, `sklearn`, `imblearn`, `xgboost`, `fastapi`, `uvicorn`, `mlflow`, `dvc`, and `pytest`
- Validates the DVC graph with `dvc dag --dot`
- Runs DVC dry-run validation with `dvc repro --dry --allow-missing --ignore-errors`
- Checks required project files
- Runs `python -m pytest tests/`

The DVC dry run is credentials-free, so CI does not need Google Drive secrets.

## Docker

Build the image:

```powershell
docker build -t customer-churn .
```

Run the API:

```powershell
docker run -p 8000:8000 customer-churn
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Monitoring

Monitoring artifacts are stored in:

```text
reports/monitoring_report.html
reports/monitoring_summary.json
```

Current monitoring includes:

- Population Stability Index (PSI)
- Drift summary
- Feature stability report

## Useful Commands

```powershell
# Run tests
python -m pytest tests

# Check DVC status
.\.venv\Scripts\dvc.exe status --no-updates

# Pull DVC data/cache
.\.venv\Scripts\dvc.exe pull

# Reproduce the pipeline
.\.venv\Scripts\dvc.exe repro

# Start the API
uvicorn app:app --reload
```

## Resume Highlights

- Built a reproducible customer churn MLOps pipeline with DVC.
- Trained and compared multiple imbalance-aware models with MLflow tracking.
- Selected an XGBoost classifier optimized for churn recall and F1 score.
- Served predictions through a FastAPI inference API.
- Added Docker packaging, monitoring reports, and automated tests.
- Added CI checks for dependencies, DVC pipeline validation, and API behavior.

## License

MIT License
