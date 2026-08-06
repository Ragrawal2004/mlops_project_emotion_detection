"""Backward-compatible shim.

All text-normalization logic now lives in
:mod:`src.features.text_processing` (shared with the DVC pipeline). This
module re-exports the same functions under their original names so any
existing code/imports referencing ``flask_app.preprocessing_utility`` keep
working without duplicating the implementation.
"""

from src.features.text_processing import (
    lemmatize as lemmatization,  # noqa: F401
    normalize_text,  # noqa: F401
    remove_numbers as removing_numbers,  # noqa: F401
    remove_punctuation as removing_punctuations,  # noqa: F401
    remove_small_sentences,  # noqa: F401
    remove_stop_words,  # noqa: F401
    remove_urls as removing_urls,  # noqa: F401
    to_lower_case as lower_case,  # noqa: F401
)

__all__ = [
    "lemmatization",
    "normalize_text",
    "removing_numbers",
    "removing_punctuations",
    "remove_small_sentences",
    "remove_stop_words",
    "removing_urls",
    "lower_case",
]