"""Text normalization utilities.

This module used to be duplicated verbatim in
``src/data/data_preprocessing.py`` and ``flask_app/preprocessing_utility.py``.
Both now import from here, so cleaning logic only has to be fixed/updated in
one place and training/serving skew can't creep in between the pipeline and
the Flask app.
"""

import re
import string
import threading

import nltk
import pandas as pd
from nltk.corpus import stopwords as _stopwords_corpus
from nltk.corpus import wordnet as _wordnet_corpus
from nltk.stem import WordNetLemmatizer




_NLTK_RESOURCES_READY = False
_NLTK_LOCK = threading.Lock()

# Built once, after the corpora are confirmed loaded. NLTK's WordNet corpus
# reader lazy-loads on first access, and that lazy-load is NOT thread-safe:
# if two Flask requests hit lemmatize()/remove_stop_words() at roughly the
# same time (the dev server handles requests in separate threads), both
# threads can trigger the lazy load simultaneously and corrupt the loader's
# internal state, raising
# AttributeError: 'WordNetCorpusReader' object has no attribute
# '_LazyCorpusLoader__args'
# Forcing the load once at startup (under a lock) and reusing a single
# lemmatizer/stopword-set instance avoids that race entirely.
_lemmatizer: WordNetLemmatizer | None = None
_stop_words: set | None = None


def ensure_nltk_resources() -> None:
    """Download and eagerly load the NLTK corpora used for text cleaning.

    Thread-safe and idempotent: guarded by a lock so concurrent Flask
    request threads can't race on NLTK's lazy corpus loading.
    """
    global _NLTK_RESOURCES_READY, _lemmatizer, _stop_words

    if _NLTK_RESOURCES_READY:
        return

    with _NLTK_LOCK:
        if _NLTK_RESOURCES_READY:  # re-check after acquiring the lock
            return

        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)

        # Force the lazy loaders to fully resolve now, while safely inside
        # the lock, instead of on first use from arbitrary request threads.
        _wordnet_corpus.ensure_loaded()
        _stop_words = set(_stopwords_corpus.words("english"))
        _lemmatizer = WordNetLemmatizer()

        _NLTK_RESOURCES_READY = True


def lemmatize(text: str) -> str:
    """Lemmatize each whitespace-separated token in ``text``."""
    ensure_nltk_resources()
    return " ".join(_lemmatizer.lemmatize(word) for word in text.split())


def remove_stop_words(text: str) -> str:
    """Remove English stop words from ``text``."""
    ensure_nltk_resources()
    return " ".join(word for word in str(text).split() if word not in _stop_words)


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
