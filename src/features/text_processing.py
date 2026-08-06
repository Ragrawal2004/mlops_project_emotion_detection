"""Text normalization utilities.

This module used to be duplicated verbatim in
``src/data/data_preprocessing.py`` and ``flask_app/preprocessing_utility.py``.
Both now import from here, so cleaning logic only has to be fixed/updated in
one place and training/serving skew can't creep in between the pipeline and
the Flask app.
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_RESOURCES_READY = False


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora required for lemmatization/stopwords.

    Idempotent and cheap to call multiple times; NLTK itself no-ops if the
    resource is already present, but we additionally guard with a module
    level flag to avoid the network/filesystem check on every call.
    """
    global _NLTK_RESOURCES_READY
    if _NLTK_RESOURCES_READY:
        return
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    _NLTK_RESOURCES_READY = True


def lemmatize(text: str) -> str:
    """Lemmatize each whitespace-separated token in ``text``."""
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())


def remove_stop_words(text: str) -> str:
    """Remove English stop words from ``text``."""
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in str(text).split() if word not in stop_words)


def remove_numbers(text: str) -> str:
    """Strip digit characters from ``text``."""
    return "".join(char for char in text if not char.isdigit())


def to_lower_case(text: str) -> str:
    """Lower-case every whitespace-separated token in ``text``."""
    return " ".join(word.lower() for word in text.split())


def remove_punctuation(text: str) -> str:
    """Remove punctuation (and stray Arabic semicolons) from ``text``."""
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("\u061b", "")  # Arabic semicolon
    return re.sub(r"\s+", " ", text).strip()


def remove_urls(text: str) -> str:
    """Strip http(s)/www URLs from ``text``."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_small_sentences(df: pd.DataFrame, min_words: int = 3) -> None:
    """Blank out rows in ``df['text']`` with fewer than ``min_words`` words.

    Mutates ``df`` in place, matching the original behaviour.
    """
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < min_words:
            df.text.iloc[i] = None


def normalize_text(text: str) -> str:
    """Run the full normalization pipeline on a single string.

    Used by both the offline DVC preprocessing stage (on a DataFrame column)
    and the Flask app (on a single user-submitted string), guaranteeing
    identical preprocessing at train and inference time.
    """
    ensure_nltk_resources()
    text = to_lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_urls(text)
    text = lemmatize(text)
    return text


def normalize_dataframe(df: pd.DataFrame, column: str = "content") -> pd.DataFrame:
    """Apply :func:`normalize_text` to every row of ``df[column]``."""
    ensure_nltk_resources()
    df[column] = df[column].apply(to_lower_case)
    df[column] = df[column].apply(remove_stop_words)
    df[column] = df[column].apply(remove_numbers)
    df[column] = df[column].apply(remove_punctuation)
    df[column] = df[column].apply(remove_urls)
    df[column] = df[column].apply(lemmatize)
    return df
