"""Model registration stage: register the evaluated model into the MLflow
Model Registry and move it to the ``Staging`` stage."""

import json

import mlflow

from src.config.config import (
    EXPERIMENT_INFO_PATH,
    MODEL_STAGE_STAGING,
    REGISTERED_MODEL_NAME,
    configure_mlflow_tracking,
)
from src.exceptions import ModelRegistrationError
from src.logger import get_logger

logger = get_logger(__name__, log_filename="model_registration.log")


def load_model_info(file_path: str) -> dict:
    """Load the run id / model path written by the evaluation stage.

    Raises:
        ModelRegistrationError: If the file is missing or invalid JSON.
    """
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError as exc:
        logger.error("File not found: %s", file_path)
        raise ModelRegistrationError(f"Model info file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in model info file: %s", exc)
        raise ModelRegistrationError(f"Invalid JSON in {file_path}") from exc


def register_model(model_name: str, model_info: dict) -> None:
    """Register a model version from a run and move it to ``Staging``.

    Raises:
        ModelRegistrationError: If registration or the stage transition fails.
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=MODEL_STAGE_STAGING,
        )

        logger.debug(
            "Model %s version %s registered and transitioned to %s.",
            model_name,
            model_version.version,
            MODEL_STAGE_STAGING,
        )
    except KeyError as exc:
        logger.error("Malformed model info payload: %s", exc)
        raise ModelRegistrationError("Model info missing required keys") from exc
    except mlflow.exceptions.MlflowException as exc:
        logger.error("Error during model registration: %s", exc)
        raise ModelRegistrationError("MLflow model registration failed") from exc


def main() -> None:
    """Run the model registration stage end to end."""
    configure_mlflow_tracking()
    try:
        model_info = load_model_info(str(EXPERIMENT_INFO_PATH))
        register_model(REGISTERED_MODEL_NAME, model_info)
    except ModelRegistrationError as exc:
        logger.error("Failed to complete the model registration process: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
