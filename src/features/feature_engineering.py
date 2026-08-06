"""Feature engineering stage: Bag-of-Words vectorization.

Fits a ``CountVectorizer`` on the processed training text, transforms both
train and test splits, persists the fitted vectorizer (tracked by DVC), and
writes the resulting feature matrices to ``data/processed``.
"""

import os
import pickle

import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer

from src.config.config import (INTERIM_DATA_DIR, PARAMS_PATH,
                               PROCESSED_DATA_DIR, VECTORIZER_PATH)
from src.exceptions import FeatureEngineeringError
from src.logger import get_logger

logger = get_logger(__name__, log_filename="feature_engineering.log")


def load_params(params_path: str) -> dict:
    """Load pipeline parameters from a YAML file.

    Raises:
        FeatureEngineeringError: If the file is missing or invalid.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError as exc:
        logger.error("File not found: %s", params_path)
        raise FeatureEngineeringError(f"Params file not found: {params_path}") from exc
    except yaml.YAMLError as exc:
        logger.error("YAML error: %s", exc)
        raise FeatureEngineeringError(f"Invalid YAML in {params_path}") from exc


def load_data(file_path: str) -> pd.DataFrame:
    """Load a processed CSV and fill missing text with an empty string.

    Raises:
        FeatureEngineeringError: If the CSV cannot be read.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaNs filled from %s", file_path)
        return df
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse the CSV file: %s", exc)
        raise FeatureEngineeringError(f"Failed to parse CSV at {file_path}") from exc


def apply_bow(
    train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int
) -> tuple:
    """Fit a Bag-of-Words vectorizer on train data and transform train/test.

    Persists the fitted vectorizer to ``VECTORIZER_PATH`` so the Flask app
    and evaluation stages can reuse the exact same feature space.

    Raises:
        FeatureEngineeringError: If vectorization or persistence fails.
    """
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        x_train = train_data["content"].values
        y_train = train_data["sentiment"].values
        x_test = test_data["content"].values
        y_test = test_data["sentiment"].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df["label"] = y_test

        os.makedirs(VECTORIZER_PATH.parent, exist_ok=True)
        with open(VECTORIZER_PATH, "wb") as file:
            pickle.dump(vectorizer, file)

        logger.debug("Bag of Words applied and data transformed")
        return train_df, test_df
    except KeyError as exc:
        logger.error("Missing expected column during BoW transformation: %s", exc)
        raise FeatureEngineeringError("Missing expected column for BoW") from exc


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Persist ``df`` to ``file_path`` as CSV, creating parent dirs as needed.

    Raises:
        FeatureEngineeringError: If the write fails.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)
    except OSError as exc:
        logger.error("Unexpected error occurred while saving the data: %s", exc)
        raise FeatureEngineeringError(f"Failed to save data to {file_path}") from exc


def main() -> None:
    """Run the feature engineering stage end to end."""
    try:
        params = load_params(str(PARAMS_PATH))
        max_features = params["feature_engineering"]["max_features"]

        train_data = load_data(str(INTERIM_DATA_DIR / "train_processed.csv"))
        test_data = load_data(str(INTERIM_DATA_DIR / "test_processed.csv"))

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df, str(PROCESSED_DATA_DIR / "train_bow.csv"))
        save_data(test_df, str(PROCESSED_DATA_DIR / "test_bow.csv"))
    except FeatureEngineeringError as exc:
        logger.error("Failed to complete the feature engineering process: %s", exc)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
