"""Data ingestion stage: download the raw dataset and split into train/test.

Reads ``params.yaml`` for the train/test split ratio, downloads the raw
tweet-emotion dataset, filters it down to the two sentiment classes used by
this project, and writes ``data/raw/train.csv`` / ``data/raw/test.csv``.
"""

import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config.config import (
    DATA_SOURCE_URL,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SENTIMENT_CLASSES_TO_KEEP,
    SENTIMENT_LABEL_MAP,
)
from src.exceptions import DataIngestionError
from src.logger import get_logger

logger = get_logger(__name__, log_filename="data_ingestion.log")


def load_params(params_path: str) -> dict:
    """Load pipeline parameters from a YAML file.

    Args:
        params_path: Path to ``params.yaml``.

    Returns:
        Parsed parameters as a dictionary.

    Raises:
        DataIngestionError: If the file is missing or cannot be parsed.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError as exc:
        logger.error("File not found: %s", params_path)
        raise DataIngestionError(f"Params file not found: {params_path}") from exc
    except yaml.YAMLError as exc:
        logger.error("YAML error: %s", exc)
        raise DataIngestionError(f"Invalid YAML in {params_path}") from exc


def load_data(data_url: str) -> pd.DataFrame:
    """Load the raw dataset from a URL or local path.

    Args:
        data_url: CSV location (URL or filesystem path).

    Returns:
        The loaded DataFrame.

    Raises:
        DataIngestionError: If the CSV cannot be read or parsed.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse the CSV file: %s", exc)
        raise DataIngestionError(f"Failed to parse CSV at {data_url}") from exc
    except Exception as exc:
        logger.error("Unexpected error occurred while loading the data: %s", exc)
        raise DataIngestionError("Unexpected error while loading data") from exc


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the two sentiment classes used by this project and encode them.

    Args:
        df: Raw dataframe containing ``tweet_id`` and ``sentiment`` columns.

    Returns:
        A dataframe restricted to happiness/sadness tweets with sentiment
        encoded as 1/0.

    Raises:
        DataIngestionError: If expected columns are missing.
    """
    try:
        df = df.drop(columns=["tweet_id"])
        final_df = df[df["sentiment"].isin(SENTIMENT_CLASSES_TO_KEEP)].copy()
        final_df["sentiment"] = final_df["sentiment"].replace(SENTIMENT_LABEL_MAP)
        logger.debug("Data preprocessing completed")
        return final_df
    except KeyError as exc:
        logger.error("Missing column in the dataframe: %s", exc)
        raise DataIngestionError(f"Missing expected column: {exc}") from exc


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Persist the train/test split to ``<data_path>/raw``.

    Args:
        train_data: Training split.
        test_data: Test split.
        data_path: Base data directory.

    Raises:
        DataIngestionError: If the files cannot be written.
    """
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug("Train and test data saved to %s", raw_data_path)
    except OSError as exc:
        logger.error("Unexpected error occurred while saving the data: %s", exc)
        raise DataIngestionError("Failed to save raw train/test data") from exc


def main() -> None:
    """Run the data ingestion stage end to end."""
    try:
        params = load_params(params_path="params.yaml")
        test_size = params["data_ingestion"]["test_size"]

        df = load_data(data_url=DATA_SOURCE_URL)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=RANDOM_SEED
        )
        save_data(train_data, test_data, data_path=str(RAW_DATA_DIR.parent))
    except DataIngestionError as exc:
        logger.error("Failed to complete the data ingestion process: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
