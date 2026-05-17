# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Split raw churn data into train.csv and test.csv."""

    logger = logging.getLogger(__name__)
    logger.info("Loading raw data")

    df = pd.read_csv(input_filepath)

    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Splitting data into train and test")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Churn"]
    )

    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    logger.info("Data ingestion completed")
    logger.info(f"Train saved to {output_path / 'train.csv'}")
    logger.info(f"Test saved to {output_path / 'test.csv'}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()