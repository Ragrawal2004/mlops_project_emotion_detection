"""Shared logging configuration.

Every pipeline stage previously duplicated ~15 lines of logging
boilerplate (console handler + its own file handler + formatter) and wrote
its error log into the project root. This module centralizes that setup:

* One console handler (DEBUG) shared by all loggers.
* One rotating-free file handler per named logger, but all files now live
  under ``logs/`` instead of the project root.
* A single formatter used everywhere.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
from pathlib import Path

from src.config.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_FORMATTER = logging.Formatter(_LOG_FORMAT)

# Cache so repeated calls with the same name don't attach duplicate handlers.
_CONFIGURED_LOGGERS: dict = {}


def get_logger(name: str, log_filename: str | None = None) -> logging.Logger:
    """Return a configured logger that writes to console and to ``logs/``.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_filename: Optional explicit log file name (stored under
            ``LOGS_DIR``). Defaults to ``<name>.log``.

    Returns:
        A ``logging.Logger`` configured with a DEBUG console handler and an
        ERROR file handler.
    """
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(_FORMATTER)

    log_file = Path(log_filename) if log_filename else Path(f"{name}.log")
    file_path = LOGS_DIR / log_file.name
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(_FORMATTER)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS[name] = logger
    return logger
