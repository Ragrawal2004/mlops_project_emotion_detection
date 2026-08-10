# syntax=docker/dockerfile:1
# ============================================================
# STAGE 1 — BUILD
# ============================================================
FROM python:3.10-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/opt/nltk_data
WORKDIR /app

# Build dependencies only exist in this stage
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && python -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH"

# Copy only dependency-related files first
COPY requirements.txt setup.py ./

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile -r requirements.txt

# Install application
COPY src ./src
RUN pip install --no-cache-dir --no-compile . --no-deps

# Download required NLTK resources, then VALIDATE them immediately.
# This is critical: nltk.download() can silently write a truncated/corrupt
# zip on a flaky connection. Without validation, the build "succeeds" but
# the app crashes at runtime with zipfile.BadZipFile the first time a
# gunicorn worker tries to lazily load the corpus — much harder to debug
# than a failed build.
RUN python -c "import nltk; \
    nltk.download('stopwords', download_dir='/opt/nltk_data'); \
    nltk.download('wordnet', download_dir='/opt/nltk_data')" \
    && python -c "import nltk; nltk.data.path.insert(0, '/opt/nltk_data'); \
    nltk.data.find('corpora/wordnet.zip'); \
    nltk.data.find('corpora/stopwords.zip'); \
    print('NLTK data verified OK')"

# Remove pip/setuptools caches and unnecessary metadata
RUN rm -rf \
    /root/.cache \
    /tmp/* \
    /opt/venv/lib/python3.10/site-packages/pip* \
    /opt/venv/lib/python3.10/site-packages/setuptools*

# ============================================================
# STAGE 2 — RUNTIME
# ============================================================
FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA=/opt/nltk_data \
    FLASK_APP=flask_app/app.py
WORKDIR /app

# Copy only runtime dependencies
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /opt/nltk_data

# Application files only
COPY src ./src
COPY flask_app ./flask_app
COPY models ./models

# Create non-root user and required directory (logs dir must exist and be
# writable by this UID before the app starts, or gunicorn workers crash
# with PermissionError on boot)
RUN mkdir -p /app/logs \
    && chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 5001

# No curl required — uses Python's stdlib for the healthcheck
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/', timeout=3)" || exit 1

# DAGSHUB_PAT must be supplied at `docker run` time, never baked into the image.

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5001", \
     "--workers", "2", \
     "--timeout", "120", \
     "flask_app.app:app"]