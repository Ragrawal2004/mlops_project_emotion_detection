"""Model evaluation stage: score the trained model and log to MLflow.

Loads the trained model and test features, computes accuracy/precision/
recall/AUC, logs metrics + params + the model itself to the DagsHub-hosted
MLflow tracking server, and writes ``reports/metrics.json`` and
``reports/experiment_info.json`` for the downstream registration stage.
"""

import json
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.config.config import (
    EXPERIMENT_INFO_PATH,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    configure_mlflow_tracking,
)
from src.exceptions import ModelEvaluationError
from src.logger import get_logger

logger = get_logger(__name__, log_filename="model_evaluation.log")


def load_model(file_path: str):
    """Unpickle the trained model.

    Raises:
        ModelEvaluationError: If the model file is missing or unreadable.
    """
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except FileNotFoundError as exc:
        logger.error("File not found: %s", file_path)
        raise ModelEvaluationError(f"Model file not found: {file_path}") from exc


def load_data(file_path: str) -> pd.DataFrame:
    """Load a processed feature CSV.

    Raises:
        ModelEvaluationError: If the CSV cannot be read.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse the CSV file: %s", exc)
        raise ModelEvaluationError(f"Failed to parse CSV at {file_path}") from exc


def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute accuracy, precision, recall, and AUC for ``clf`` on test data.

    Raises:
        ModelEvaluationError: If prediction/scoring fails.
    """
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }
        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except ValueError as exc:
        logger.error("Error during model evaluation: %s", exc)
        raise ModelEvaluationError("Model evaluation failed") from exc


def save_metrics(metrics: dict, file_path: str) -> None:
    """Write ``metrics`` to ``file_path`` as JSON.

    Raises:
        ModelEvaluationError: If the write fails.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except OSError as exc:
        logger.error("Error occurred while saving the metrics: %s", exc)
        raise ModelEvaluationError(f"Failed to save metrics to {file_path}") from exc


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Write the MLflow run id + model artifact path to ``file_path``.

    Raises:
        ModelEvaluationError: If the write fails.
    """
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except OSError as exc:
        logger.error("Error occurred while saving the model info: %s", exc)
        raise ModelEvaluationError(f"Failed to save model info to {file_path}") from exc


def main() -> None:
    """Run the model evaluation stage end to end."""
    configure_mlflow_tracking()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        try:
            clf = load_model(str(MODEL_PATH))
            test_data = load_data(str(PROCESSED_DATA_DIR / "test_bow.csv"))

            x_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, x_test, y_test)

            save_metrics(metrics, str(METRICS_PATH))

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(clf, "get_params"):
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(clf, "model")

            save_model_info(run.info.run_id, "model", str(EXPERIMENT_INFO_PATH))

            mlflow.log_artifact(str(METRICS_PATH))
            # NOTE: the original script also tried to log a non-existent
            # 'reports/model_info.json' here (the real file written above is
            # EXPERIMENT_INFO_PATH). That mismatched path would raise at
            # runtime; we log the file that actually exists instead.
            mlflow.log_artifact(str(EXPERIMENT_INFO_PATH))

        except ModelEvaluationError as exc:
            logger.error("Failed to complete the model evaluation process: %s", exc)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
