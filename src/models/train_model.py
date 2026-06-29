# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import click
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

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
MODEL_PICKLE_NAME = "model.pkl"
MODEL_NATIVE_NAME = "model.ubj"


def log_model_to_mlflow(model, artifact_path):
    """Log models with the MLflow flavor that matches the estimator type."""

    if isinstance(model, XGBClassifier):
        mlflow.xgboost.log_model(
            xgb_model=model.get_booster(),
            artifact_path=artifact_path,
            model_format="ubj"
        )
        return

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path
    )


@click.command()
@click.argument("train_filepath", type=click.Path(exists=True))
@click.argument("test_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path())
@click.argument("metrics_filepath", type=click.Path())
def main(train_filepath, test_filepath, model_filepath, metrics_filepath):
    """Train multiple models, log each to MLflow, select best model, and save artifacts."""

    logger = logging.getLogger(__name__)

    logger.info("Loading processed train and test data")

    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "LogisticRegression_SMOTEENN": Pipeline(steps=[
            ("smoteenn", SMOTEENN(random_state=42)),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "DecisionTree_SMOTEENN": Pipeline(steps=[
            ("smoteenn", SMOTEENN(random_state=42)),
            ("model", DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=8,
                random_state=42
            ))
        ]),

        "RandomForest_SMOTEENN": Pipeline(steps=[
            ("smoteenn", SMOTEENN(random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "GradientBoosting_SMOTEENN": Pipeline(steps=[
            ("smoteenn", SMOTEENN(random_state=42)),
            ("model", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            ))
        ]),

        # XGBoost handles imbalance with scale_pos_weight, not SMOTEENN
        "XGBClassifier_scale_pos_weight": XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
    }

    mlflow.set_experiment("customer_churn_prediction")

    best_model = None
    best_model_name = None
    best_f1 = -1
    all_results = {}

    for model_name, model in models.items():
        logger.info(f"Training {model_name}")

        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

            mlflow.log_param("model_name", model_name)

            if "SMOTEENN" in model_name:
                mlflow.log_param("imbalance_method", "SMOTEENN")
            else:
                mlflow.log_param("imbalance_method", "scale_pos_weight")

            mlflow.log_metrics(metrics)

            log_model_to_mlflow(model, artifact_path="model")

            all_results[model_name] = metrics

            logger.info(f"{model_name} metrics: {metrics}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model
                best_model_name = model_name

    logger.info(f"Best model selected: {best_model_name}")

    final_y_pred = best_model.predict(X_test)
    final_y_proba = best_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "best_model": best_model_name,
        "accuracy": accuracy_score(y_test, final_y_pred),
        "precision": precision_score(y_test, final_y_pred),
        "recall": recall_score(y_test, final_y_pred),
        "f1_score": f1_score(y_test, final_y_pred),
        "roc_auc": roc_auc_score(y_test, final_y_proba),
        "confusion_matrix": confusion_matrix(y_test, final_y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            final_y_pred,
            labels=[0, 1],
            output_dict=True
        ),
        "all_model_results": all_results
    }

    model_path = Path(model_filepath)
    model_path.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(metrics_filepath)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    local_model_path = model_path / MODEL_PICKLE_NAME
    native_model_path = None

    logger.info("Saving best model and metrics")

    joblib.dump(best_model, local_model_path)

    if isinstance(best_model, XGBClassifier):
        native_model_path = model_path / MODEL_NATIVE_NAME
        best_model.get_booster().save_model(str(native_model_path))

    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    with mlflow.start_run(run_name=f"BEST_MODEL_{best_model_name}"):
        mlflow.log_param("best_model", best_model_name)

        mlflow.log_metrics({
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "roc_auc": final_metrics["roc_auc"]
        })

        log_model_to_mlflow(best_model, artifact_path="model")

        mlflow.log_artifact(str(local_model_path))
        if native_model_path is not None and native_model_path.exists():
            mlflow.log_artifact(str(native_model_path))
        mlflow.log_artifact(str(metrics_path))

    logger.info("Model training completed")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Model saved to: {local_model_path}")
    if native_model_path is not None and native_model_path.exists():
        logger.info(f"Native XGBoost model saved to: {native_model_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Accuracy: {final_metrics['accuracy']}")
    logger.info(f"Recall: {final_metrics['recall']}")
    logger.info(f"F1 Score: {final_metrics['f1_score']}")
    logger.info(f"ROC-AUC: {final_metrics['roc_auc']}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

