"""Centralized configuration for the sentiment-analysis MLOps project.

Every hardcoded value that was previously scattered across scripts (paths,
MLflow/DagsHub settings, the model name, the random seed, etc.) lives here.
Import from this module instead of hardcoding literals in pipeline stages,
the Flask app, or the tests.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (DAGSHUB_PAT, etc.) if present.
load_dotenv()

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "model.pkl"
VECTORIZER_PATH: Path = MODELS_DIR / "vectorizer.pkl"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
EXPERIMENT_INFO_PATH: Path = REPORTS_DIR / "experiment_info.json"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

LOGS_DIR: Path = PROJECT_ROOT / "logs"

PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
DATA_SOURCE_URL: str = (
    "https://raw.githubusercontent.com/campusx-official/"
    "jupyter-masterclass/main/tweet_emotions.csv"
)

# The two sentiment classes kept from the raw multi-class dataset, and how
# they are encoded as binary labels.
SENTIMENT_LABEL_MAP: dict = {"happiness": 1, "sadness": 0}
SENTIMENT_CLASSES_TO_KEEP = list(SENTIMENT_LABEL_MAP.keys())

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED: int = 44

# --------------------------------------------------------------------------
# MLflow / DagsHub configuration
# --------------------------------------------------------------------------
DAGSHUB_REPO_OWNER: str = "Ragrawal2004"
DAGSHUB_REPO_NAME: str = "mlops_project_emotion_detection"
MLFLOW_TRACKING_URI: str = (
    f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
)
MLFLOW_EXPERIMENT_NAME: str = "dvc-pipeline"

REGISTERED_MODEL_NAME: str = "my_model"
MODEL_STAGE_STAGING: str = "Staging"
MODEL_STAGE_PRODUCTION: str = "Production"

# --------------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------------
DAGSHUB_PAT_ENV_VAR: str = "DAGSHUB_PAT"


def get_dagshub_token() -> str:
    """Return the DagsHub personal access token from the environment.

    Raises:
        EnvironmentError: If the ``DAGSHUB_PAT`` environment variable is not set.
    """
    token = os.getenv(DAGSHUB_PAT_ENV_VAR)
    if not token:
        raise EnvironmentError(f"{DAGSHUB_PAT_ENV_VAR} environment variable is not set")
    return token


def configure_mlflow_tracking() -> None:
    """Configure MLflow to authenticate against and track to DagsHub.

    Uses ``dagshub.init(..., mlflow=True)`` to wire up MLflow's tracking
    URI and credentials against this project's DagsHub-hosted MLflow
    server (https://dagshub.com/Ragrawal2004/mlops_project_emotion_detection.mlflow).
    The DagsHub token is read from the ``DAGSHUB_PAT`` environment variable
    (set locally via ``.env``, or as a CI secret) and passed through as
    ``DAGSHUB_USER_TOKEN`` so ``dagshub.init`` authenticates non-interactively
    instead of trying to open a browser — required for CI runners and any
    other headless environment.

    Centralizing this avoids duplicating the same setup in
    ``model_evaluation.py``, ``register_model.py``, ``promote_model.py``,
    ``app.py`` and the test suite.
    """
    token = get_dagshub_token()
    os.environ["DAGSHUB_USER_TOKEN"] = token

    import dagshub

    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True
    )
