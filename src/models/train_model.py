# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import click
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


TARGET_COLUMN = "Churn"


@click.command()
@click.argument("train_filepath", type=click.Path(exists=True))
@click.argument("test_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path())
@click.argument("metrics_filepath", type=click.Path())
def main(train_filepath, test_filepath, model_filepath, metrics_filepath):
    """Train SMOTEENN + GradientBoostingClassifier, save model, metrics, and log to MLflow."""

    logger = logging.getLogger(__name__)

    logger.info("Loading processed train and test data")

    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model_params = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "subsample": 0.8,
        "random_state": 42
    }

    logger.info("Creating SMOTEENN + GradientBoosting model")

    model = Pipeline(steps=[
        ("smoteenn", SMOTEENN(random_state=42)),
        ("gb", GradientBoostingClassifier(**model_params))
    ])

    mlflow.set_experiment("customer_churn_prediction")

    with mlflow.start_run(run_name="SMOTEENN_GradientBoosting"):
        logger.info("Logging parameters to MLflow")

        mlflow.log_param("model_name", "GradientBoostingClassifier")
        mlflow.log_param("imbalance_method", "SMOTEENN")

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        logger.info("Training model")
        model.fit(X_train, y_train)

        logger.info("Predicting on test data")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model_name": "GradientBoostingClassifier_SMOTEENN",
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                output_dict=True
            )
        }

        logger.info("Logging metrics to MLflow")

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])

        model_path = Path(model_filepath)
        model_path.mkdir(parents=True, exist_ok=True)

        metrics_path = Path(metrics_filepath)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        local_model_path = model_path / "model.pkl"

        logger.info("Saving local model and metrics")

        joblib.dump(model, local_model_path)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Logging model artifact to MLflow")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        mlflow.log_artifact(str(local_model_path))
        mlflow.log_artifact(str(metrics_path))

        logger.info("Model training completed")
        logger.info(f"Model saved to: {local_model_path}")
        logger.info(f"Metrics saved to: {metrics_path}")
        logger.info(f"Accuracy: {metrics['accuracy']}")
        logger.info(f"Recall: {metrics['recall']}")
        logger.info(f"F1 Score: {metrics['f1_score']}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
