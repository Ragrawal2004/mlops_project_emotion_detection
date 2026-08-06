"""Data preprocessing stage: normalize raw tweet text.

Loads ``data/raw/{train,test}.csv``, applies text normalization (lower
casing, stop-word removal, punctuation/URL/number stripping, lemmatization)
via :mod:`src.features.text_processing`, and writes the cleaned data to
``data/interim``.
"""

import os

import pandas as pd

from src.config.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.exceptions import DataPreprocessingError
from src.features.text_processing import normalize_dataframe
from src.logger import get_logger

logger = get_logger(__name__, log_filename="data_preprocessing.log")


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the ``content`` column of ``df``.

    Thin wrapper around :func:`src.features.text_processing.normalize_dataframe`
    kept here so the stage's public API is unchanged.

    Raises:
        DataPreprocessingError: If normalization fails.
    """
    try:
        df = normalize_dataframe(df, column="content")
        logger.debug("Text normalization completed")
        return df
    except Exception as exc:
        logger.error("Error during text normalization: %s", exc)
        raise DataPreprocessingError("Failed to normalize text") from exc


def main() -> None:
    """Run the data preprocessing stage end to end."""
    try:
        train_data = pd.read_csv(RAW_DATA_DIR / "train.csv")
        test_data = pd.read_csv(RAW_DATA_DIR / "test.csv")
        logger.debug("data loaded properly")

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
        train_processed_data.to_csv(
            INTERIM_DATA_DIR / "train_processed.csv", index=False
        )
        test_processed_data.to_csv(INTERIM_DATA_DIR / "test_processed.csv", index=False)

        logger.debug("Processed data saved to %s", INTERIM_DATA_DIR)
    except DataPreprocessingError as exc:
        logger.error("Failed to complete the data transformation process: %s", exc)
        raise SystemExit(1) from exc
    except OSError as exc:
        logger.error("Failed to read/write processed data: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
