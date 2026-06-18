# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]


@click.command()
@click.argument("train_filepath", type=click.Path(exists=True))
@click.argument("test_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("model_filepath", type=click.Path())
def main(train_filepath, test_filepath, output_filepath, model_filepath):
    """Build processed train/test datasets and save preprocessing object."""

    logger = logging.getLogger(__name__)

    logger.info("Loading train and test data")
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    logger.info("Converting TotalCharges to numeric")
    train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
    test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

    logger.info("Separating features and target")

    X_train = train_df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    X_test = test_df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Numeric features: {numeric_features}")
    logger.info(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # For newer sklearn versions
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # For older sklearn versions
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    logger.info("Fitting preprocessor on train data")
    X_train_processed = preprocessor.fit_transform(X_train)

    logger.info("Transforming test data")
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrix to dense array if needed
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # Get feature names safely
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    logger.info("Creating processed dataframes")

    train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    train_processed[TARGET_COLUMN] = y_train.values

    test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    test_processed[TARGET_COLUMN] = y_test.values

    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_filepath)
    model_path.mkdir(parents=True, exist_ok=True)

    train_output_path = output_path / "train_processed.csv"
    test_output_path = output_path / "test_processed.csv"
    preprocessor_output_path = model_path / "preprocessor.pkl"

    logger.info("Saving processed files")

    train_processed.to_csv(train_output_path, index=False)
    test_processed.to_csv(test_output_path, index=False)

    logger.info("Saving preprocessor")
    joblib.dump(preprocessor, preprocessor_output_path)

    logger.info("Feature building completed")
    logger.info(f"Train processed saved to: {train_output_path}")
    logger.info(f"Test processed saved to: {test_output_path}")
    logger.info(f"Preprocessor saved to: {preprocessor_output_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()