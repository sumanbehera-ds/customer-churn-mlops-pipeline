# Customer Churn MLOps Pipeline

End-to-end MLOps project for predicting customer churn using machine learning, DVC pipelines, MLflow experiment tracking, FastAPI deployment, Docker containerization, and GitHub Actions CI.

This project converts a customer churn machine learning experiment into a reproducible production-style ML pipeline.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave a service.  
This project uses the Telco Customer Churn dataset to build a machine learning pipeline that predicts whether a customer is likely to churn.

The project covers the complete MLOps workflow:

- Data versioning with DVC
- Data ingestion pipeline
- Feature engineering and preprocessing
- Model training with imbalance handling
- Experiment tracking with MLflow
- Model serialization
- FastAPI prediction API
- Dockerized deployment
- GitHub Actions CI workflow

---

## Tech Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Imbalance Handling | SMOTEENN |
| Model | GradientBoostingClassifier |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI, Pydantic, Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Version Control | Git, GitHub |

---

## Machine Learning Workflow

```text
Raw Data
   ↓
Data Ingestion
   ↓
Train/Test Split
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
SMOTEENN Imbalance Handling
   ↓
Gradient Boosting Model Training
   ↓
Evaluation
   ↓
MLflow Tracking
   ↓
Model Serialization
   ↓
FastAPI Prediction API
   ↓
Docker Deployment

customer-churn-mlops-pipeline/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dvc/
│   └── config
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv.dvc
│   ├── processed/
│   ├── interim/
│   └── external/
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── reports/
│   └── metrics.json
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   │
│   └── visualization/
│       └── visualize.py
│
├── app.py
├── Dockerfile
├── .dockerignore
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
└── setup.py