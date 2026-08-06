"""Quick standalone check that DagsHub + MLflow credentials are wired up
correctly, independent of the full DVC pipeline.

Usage:
    python scripts/mlflow_connection_check.py

Requires DAGSHUB_PAT to be set (in .env locally, or as an env var in CI).
Logs one dummy param/metric to a throwaway run so you can see it show up
under Experiments on https://dagshub.com/Ragrawal2004/mlops_project_emotion_detection.
"""

import mlflow

from src.config.config import configure_mlflow_tracking
from src.logger import get_logger

logger = get_logger(__name__, log_filename="mlflow_connection_check.log")


def main() -> None:
    """Log a single dummy param/metric to confirm the tracking connection works."""
    configure_mlflow_tracking()
    logger.debug("Tracking URI: %s", mlflow.get_tracking_uri())

    with mlflow.start_run():
        mlflow.log_param("parameter name", "value")
        mlflow.log_metric("metric name", 1)

    logger.debug("Connection check complete — see the Experiments tab on DagsHub.")
    print("MLflow/DagsHub connection OK. Check the Experiments tab on DagsHub.")


if __name__ == "__main__":
    main()
