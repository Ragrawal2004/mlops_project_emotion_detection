# syntax=docker/dockerfile:1
# ============================================================
# STAGE 1 — BUILD
# ============================================================
FROM python:3.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/opt/nltk_data \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Build deps + venv, cached apt layer via BuildKit mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && python -m venv /opt/venv

# Dependency metadata first (cache layer)
COPY requirements.txt setup.py ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile -r requirements.txt

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile . --no-deps

# NLTK resources
RUN python - <<'PY'
import nltk
download_dir = "/opt/nltk_data"
for resource in ("stopwords", "wordnet"):
    if not nltk.download(resource, download_dir=download_dir, quiet=False):
        raise SystemExit(f"Failed to download NLTK resource: {resource}")
nltk.data.path.insert(0, download_dir)
for resource in ("corpora/stopwords", "corpora/wordnet"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.data.find(resource + ".zip")
print("NLTK data verified successfully.")
PY

# ============================================================
# STAGE 2 — RUNTIME
# ============================================================
FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA=/opt/nltk_data \
    FLASK_APP=flask_app/app.py \
    WEB_CONCURRENCY=2 \
    APP_ROOT=/app

WORKDIR /app

RUN addgroup --system --gid 10001 appgroup \
    && adduser --system --uid 10001 --gid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appgroup /app/logs

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appgroup /opt/nltk_data /opt/nltk_data
COPY --chown=appuser:appgroup flask_app ./flask_app
COPY --chown=appuser:appgroup models ./models

USER appuser
EXPOSE 80

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health', timeout=3)" \
    || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:80", \
     "--workers", "2", \
     "--timeout", "120", \
     "flask_app.app:app"]