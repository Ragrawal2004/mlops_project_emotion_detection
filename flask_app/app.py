"""Flask serving app for the sentiment-analysis model.

Loads the current ``Production`` (falls back to latest) model version from
the MLflow Model Registry and the Bag-of-Words vectorizer used at training
time, and exposes a single-page form for interactive predictions.
"""

import pickle

import mlflow
import pandas as pd
from flask import Flask, render_template, request

from src.config.config import (
    REGISTERED_MODEL_NAME,
    VECTORIZER_PATH,
    configure_mlflow_tracking,
)
from src.exceptions import ConfigurationError
from src.features.text_processing import ensure_nltk_resources, normalize_text
from src.logger import get_logger

logger = get_logger(__name__, log_filename="flask_app.log")

configure_mlflow_tracking()
logger.debug("Tracking URI: %s", mlflow.get_tracking_uri())

# Load WordNet/stopwords once, synchronously, at startup — before any
# request thread can touch them. See src/features/text_processing.py for
# why concurrent lazy-loading of the WordNet corpus crashes with
# AttributeError: 'WordNetCorpusReader' object has no attribute
# '_LazyCorpusLoader__args'
ensure_nltk_resources()

app = Flask(__name__)


def get_latest_model_version(model_name: str) -> str:
    """Return the highest registered version number for ``model_name``.

    Raises:
        ConfigurationError: If no versions are registered for the model.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ConfigurationError(f"No versions found for model '{model_name}'")

    latest = max(versions, key=lambda version: int(version.version))
    logger.debug("Using model version: %s", latest.version)
    return latest.version


def load_model_and_vectorizer():
    """Load the latest registered model and the fitted vectorizer."""
    model_version = get_latest_model_version(REGISTERED_MODEL_NAME)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
    logger.debug("Loading model from: %s", model_uri)

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    with open(VECTORIZER_PATH, "rb") as file:
        loaded_vectorizer = pickle.load(file)

    return loaded_model, loaded_vectorizer


model, vectorizer = load_model_and_vectorizer()


@app.route("/")
def home():
    """Render the input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Normalize the submitted text, vectorize it, and return a prediction."""
    text = request.form["text"]
    normalized_text = normalize_text(text)

    features = vectorizer.transform([normalized_text])
    features_df = pd.DataFrame(
        features.toarray(), columns=[str(i) for i in range(features.shape[1])]
    )

    result = model.predict(features_df)

    return render_template("index.html", result=result[0])


if __name__ == "__main__":
    app.run(debug=True, port=5001)
