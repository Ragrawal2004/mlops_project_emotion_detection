"""Model building stage: train the Logistic Regression classifier."""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config.config import MODEL_PATH, PROCESSED_DATA_DIR
from src.exceptions import ModelBuildingError
from src.logger import get_logger

logger = get_logger(__name__, log_filename="model_building.log")


def load_data(file_path: str) -> pd.DataFrame:
    """Load a processed feature CSV.

    Raises:
        ModelBuildingError: If the CSV cannot be read.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse the CSV file: %s", exc)
        raise ModelBuildingError(f"Failed to parse CSV at {file_path}") from exc


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    Raises:
        ModelBuildingError: If training fails.
    """
    try:
        clf = LogisticRegression(C=1, solver="liblinear", penalty="l2")
        clf.fit(x_train, y_train)
        logger.debug("Model training completed")
        return clf
    except ValueError as exc:
        logger.error("Error during model training: %s", exc)
        raise ModelBuildingError("Model training failed") from exc


def save_model(model: LogisticRegression, file_path: str) -> None:
    """Pickle the trained model to ``file_path``.

    Raises:
        ModelBuildingError: If persistence fails.
    """
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except OSError as exc:
        logger.error("Error occurred while saving the model: %s", exc)
        raise ModelBuildingError(f"Failed to save model to {file_path}") from exc


def main() -> None:
    """Run the model building stage end to end."""
    try:
        train_data = load_data(str(PROCESSED_DATA_DIR / "train_bow.csv"))
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(x_train, y_train)

        save_model(clf, str(MODEL_PATH))
    except ModelBuildingError as exc:
        logger.error("Failed to complete the model building process: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
