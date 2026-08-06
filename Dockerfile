# syntax=docker/dockerfile:1

# --------------------------------------------------------------------------
# Base image
# --------------------------------------------------------------------------
FROM python:3.10-slim AS base

# Don't buffer stdout/stderr (so logs show up immediately in `docker logs`),
# and don't write .pyc files into the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --------------------------------------------------------------------------
# System dependencies
# --------------------------------------------------------------------------
# gcc/build-essential are needed to build a couple of the ML wheels on slim
# images; removed again in the same layer to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Python dependencies (separate layer so code changes don't bust the cache)
# --------------------------------------------------------------------------
COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

# Pre-download the NLTK corpora used by src/features/text_processing.py so
# the container doesn't hit the network on first request.
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# --------------------------------------------------------------------------
# Application code
# --------------------------------------------------------------------------
COPY flask_app ./flask_app
COPY models ./models

# The vectorizer must exist at build/run time — it's produced by the DVC
# pipeline (feature_engineering stage) and is tracked by DVC, not baked
# into the image build itself. Mount it in, or run `dvc pull` before
# `docker build` (see README/docs/architecture.md for details).

# --------------------------------------------------------------------------
# Runtime configuration
# --------------------------------------------------------------------------
EXPOSE 5001

# DAGSHUB_PAT must be supplied at `docker run` time (-e DAGSHUB_PAT=...),
# never baked into the image.
ENV FLASK_APP=flask_app/app.py

# gunicorn instead of the Flask dev server for anything beyond local testing.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--timeout", "120", "flask_app.app:app"]
