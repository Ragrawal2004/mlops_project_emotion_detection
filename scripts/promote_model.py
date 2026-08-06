"""Promote the latest registered model version to ``Production``.

Run after the model tests pass in CI. Finds the highest version number
registered for ``REGISTERED_MODEL_NAME`` and transitions it to
``Production``, archiving any previously-production versions.
"""

import mlflow

from src.config.config import (MODEL_STAGE_PRODUCTION, REGISTERED_MODEL_NAME,
                               configure_mlflow_tracking)
from src.exceptions import ModelRegistrationError
from src.logger import get_logger

logger = get_logger(__name__, log_filename="promote_model.log")


def promote_model() -> None:
    """Promote the latest version of the registered model to Production.

    Raises:
        ModelRegistrationError: If no versions are registered or the
            transition fails.
    """
    configure_mlflow_tracking()

    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise ModelRegistrationError(
            f"No registered versions found for model '{REGISTERED_MODEL_NAME}'"
        )

    versions = sorted(versions, key=lambda version: int(version.version))
    latest_version = versions[-1].version

    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=latest_version,
            stage=MODEL_STAGE_PRODUCTION,
            archive_existing_versions=True,
        )
    except mlflow.exceptions.MlflowException as exc:
        logger.error("Failed to promote model version %s: %s", latest_version, exc)
        raise ModelRegistrationError("Failed to promote model to Production") from exc

    logger.debug("Version %s promoted to %s", latest_version, MODEL_STAGE_PRODUCTION)
    print(f"Version {latest_version} promoted to Production")


if __name__ == "__main__":
    try:
        promote_model()
    except ModelRegistrationError as exc:
        logger.error("Model promotion failed: %s", exc)
        raise SystemExit(1) from exc
