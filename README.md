# Customer Churn MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![GitHub
Actions](https://img.shields.io/badge/CI-GitHub%20Actions-success)

Production-ready **Customer Churn Prediction MLOps Pipeline** built with
**Scikit-learn, DVC, MLflow, FastAPI, Docker, GitHub Actions, and
Monitoring**.

## Live Demo

-   **API:** https://customer-churn-mlops-pipeline-85tj.onrender.com
-   **Swagger UI:**
    https://customer-churn-mlops-pipeline-85tj.onrender.com/docs

------------------------------------------------------------------------

# Business Problem

Telecom companies lose revenue when customers leave (churn). This
project predicts churn probability so retention teams can proactively
target at-risk customers.

------------------------------------------------------------------------

# Features

-   End-to-end MLOps pipeline
-   Automated data preprocessing
-   Feature engineering
-   Gradient Boosting + SMOTEENN
-   MLflow experiment tracking
-   DVC pipeline orchestration
-   FastAPI inference API
-   Docker containerization
-   GitHub Actions CI
-   Automated API tests
-   Drift monitoring report
-   Health endpoint (`/healthz`)

------------------------------------------------------------------------

# Tech Stack

  Category          Technologies
  ----------------- -------------------------
  Language          Python
  ML                Scikit-learn
  API               FastAPI
  Tracking          MLflow
  Data Versioning   DVC
  Container         Docker
  CI/CD             GitHub Actions
  Testing           Pytest
  Monitoring        Custom Drift Monitoring

------------------------------------------------------------------------

# Project Architecture

``` text
Raw Data
    │
    ▼
Data Ingestion
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
MLflow Tracking
    │
    ▼
Model Artifact
    │
    ▼
FastAPI API
    │
    ▼
Docker
    │
    ▼
Render Deployment
    │
    ▼
Monitoring
```

------------------------------------------------------------------------

# Repository Structure

``` text
customer-churn-mlops/
├── data/
├── docs/
├── models/
├── monitoring/
├── notebooks/
├── reports/
├── src/
├── tests/
├── app.py
├── Dockerfile
├── dvc.yaml
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# Pipeline

1.  Data ingestion
2.  Data preprocessing
3.  Feature engineering
4.  Model training
5.  Evaluation
6.  MLflow logging
7.  FastAPI deployment
8.  Docker packaging
9.  GitHub Actions testing
10. Drift monitoring

------------------------------------------------------------------------

# Model Performance

  Metric       Score
  ---------- -------
  Recall       0.813
  F1 Score     0.607
  ROC-AUC      0.837

------------------------------------------------------------------------

# API Endpoints

  Method   Endpoint     Description
  -------- ------------ ------------------------
  GET      `/`          Home
  GET      `/healthz`   Health check
  POST     `/predict`   Predict customer churn

Example request:

``` json
{
  "gender":"Female",
  "SeniorCitizen":0,
  "tenure":1,
  "MonthlyCharges":29.85,
  "TotalCharges":29.85
}
```

------------------------------------------------------------------------

# Monitoring

The project generates:

-   `reports/monitoring_report.html`
-   `reports/monitoring_summary.json`

Current monitoring includes:

-   Population Stability Index (PSI)
-   Drift summary
-   Feature stability report

------------------------------------------------------------------------

# Testing

Current automated tests include:

-   Home endpoint
-   Health endpoint
-   Prediction endpoint
-   Invalid payload validation
-   Missing field validation

Run:

``` bash
pytest tests/
```

------------------------------------------------------------------------

# CI/CD

GitHub Actions automatically:

-   Installs dependencies
-   Validates imports
-   Checks project files
-   Runs pytest
-   Verifies monitoring artifacts

------------------------------------------------------------------------

# Docker

``` bash
docker build -t customer-churn .
docker run -p 8000:8000 customer-churn
```

------------------------------------------------------------------------

# Local Setup

``` bash
git clone https://github.com/sumanbehera-ds/customer-churn-mlops-pipeline.git
cd customer-churn-mlops-pipeline

python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn app:app --reload
```

------------------------------------------------------------------------

# Future Improvements

-   Cloud DVC remote
-   Prometheus + Grafana
-   Model Registry automation
-   Drift alerts
-   Kubernetes deployment

------------------------------------------------------------------------

# Resume Highlights

-   Built an end-to-end production-ready MLOps pipeline.
-   Automated data versioning with DVC.
-   Tracked experiments using MLflow.
-   Developed a FastAPI inference service.
-   Containerized the application with Docker.
-   Implemented CI/CD using GitHub Actions.
-   Added monitoring and automated testing.

------------------------------------------------------------------------

# License

MIT License