# Project Architecture

## Overview

This project is a Bag-of-Words + Logistic Regression sentiment classifier
(happiness vs. sadness) for tweets, wired into an MLOps pipeline:

```
raw data --> preprocessing --> feature engineering --> training --> evaluation --> registration --> promotion --> Flask serving
```

## Pipeline stages (DVC-orchestrated, see `dvc.yaml`)

| Stage | Module | Reads | Writes |
|---|---|---|---|
| Data ingestion | `src/data/data_ingestion.py` | remote CSV | `data/raw/{train,test}.csv` |
| Preprocessing | `src/data/data_preprocessing.py` | `data/raw` | `data/interim/{train,test}_processed.csv` |
| Feature engineering | `src/features/feature_engineering.py` | `data/interim` | `data/processed/{train,test}_bow.csv`, `models/vectorizer.pkl` |
| Model building | `src/model/model_building.py` | `data/processed` | `models/model.pkl` |
| Model evaluation | `src/model/model_evaluation.py` | `models/model.pkl`, `data/processed` | `reports/metrics.json`, `reports/experiment_info.json`, logs to MLflow |
| Model registration | `src/model/register_model.py` | `reports/experiment_info.json` | registers a model version in MLflow, stage=Staging |

`scripts/promote_model.py` runs after CI tests pass and promotes the latest
registered version to `Production` (archiving the prior production
version).

## Shared modules

- `src/config/config.py` — every path, constant, and environment-derived
  setting (MLflow tracking URI, DagsHub repo, random seed, registered
  model name, stage names, etc.) lives here.
- `src/logger.py` — one logger factory used by every stage; writes to
  `logs/<module>.log` plus the console.
- `src/exceptions.py` — one exception type per pipeline stage so failures
  are traceable to their source.
- `src/features/text_processing.py` — the single implementation of text
  normalization (lower-casing, stop-word removal, punctuation/URL/number
  stripping, lemmatization) used identically by the offline pipeline and
  the Flask app, preventing training/serving skew.
- `src/visualization/visualize.py` — confusion matrix and classification
  report generation, writing into `reports/`.

## Serving

`flask_app/app.py` loads the current highest-versioned model from the
MLflow Model Registry (`models:/<name>/<version>`) plus the fitted
vectorizer from `models/vectorizer.pkl`, and exposes a single form for
interactive predictions at `/predict`.

## Containerized serving (Docker)

The Flask app can be built and run as a container:

```bash
docker build -t mlops-mini-project .
docker run -p 5001:5001 \
  -e DAGSHUB_PAT=your_token \
  -v $(pwd)/models:/app/models:ro \
  mlops-mini-project
```

or, for local development, `docker compose up --build` (reads `DAGSHUB_PAT`
from `.env` and mounts `models/` so a locally-trained `vectorizer.pkl` is
picked up without rebuilding the image).

The image only bundles code — `models/` is mounted at runtime rather than
baked in, since those files are DVC-tracked artifacts that change every
time the pipeline retrains. `data/`, `.env`, and DVC's local cache are
excluded from the build context via `.dockerignore`.

In CI (`.github/workflows/ci.yaml`), a second job (`docker-build`) builds
the image after tests pass, downloads the `model.pkl`/`vectorizer.pkl`
produced by that run's `dvc repro`, runs the container, and hits `/` to
confirm it serves before the workflow succeeds. Pushing the image to a
registry is not enabled by default — see the commented-out login step in
the workflow for how to wire up Docker Hub or GHCR when you're ready.

## Experiment tracking

MLflow tracking is hosted on DagsHub at
https://dagshub.com/Ragrawal2004/mlops_project_emotion_detection.mlflow.
Authentication uses `dagshub.init(repo_owner=..., repo_name=..., mlflow=True)`
via `src.config.config.configure_mlflow_tracking()`, driven by the
`DAGSHUB_PAT` environment variable (loaded from `.env` locally, or from a
repository secret in CI — passed through as `DAGSHUB_USER_TOKEN` so
`dagshub.init` doesn't try to open an interactive browser login).

To sanity-check credentials independently of the full pipeline:

```bash
python scripts/mlflow_connection_check.py
```

## Data & model versioning

Datasets and pickled artifacts (`models/*.pkl`) are tracked with DVC, not
Git — see `.gitignore` and `dvc.yaml`. Only code, configuration, and DVC
metadata files (`.dvc`, `dvc.lock`) are committed to Git.
