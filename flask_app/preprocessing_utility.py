"""Backward-compatible shim.

All text-normalization logic now lives in
:mod:`src.features.text_processing` (shared with the DVC pipeline). This
module re-exports the same functions under their original names so any
existing code/imports referencing ``flask_app.preprocessing_utility`` keep
working without duplicating the implementation.
"""

from src.features.text_processing import (  # noqa: F401
    lemmatize as lemmatization,
    normalize_text,
    remove_numbers as removing_numbers,
    remove_punctuation as removing_punctuations,
    remove_small_sentences,
    remove_stop_words,
    remove_urls as removing_urls,
    to_lower_case as lower_case,
)
