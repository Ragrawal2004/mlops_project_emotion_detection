"""Backward-compatible shim.

All text-normalization logic now lives in
:mod:`src.features.text_processing` (shared with the DVC pipeline). This
module re-exports the same functions under their original names so any
existing code/imports referencing ``flask_app.preprocessing_utility`` keep
working without duplicating the implementation.
"""

from src.features.text_processing import \
    lemmatize as lemmatization  # noqa: F401
from src.features.text_processing import normalize_text
from src.features.text_processing import remove_numbers as removing_numbers
from src.features.text_processing import \
    remove_punctuation as removing_punctuations
from src.features.text_processing import (remove_small_sentences,
                                          remove_stop_words)
from src.features.text_processing import remove_urls as removing_urls
from src.features.text_processing import to_lower_case as lower_case
