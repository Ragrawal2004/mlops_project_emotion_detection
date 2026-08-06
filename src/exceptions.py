"""Custom exception hierarchy for the sentiment-analysis MLOps project.

Replacing bare ``except Exception`` / re-raises with these types makes
failures easier to diagnose (a ``DataIngestionError`` vs a
``ModelRegistrationError`` tells you exactly which pipeline stage broke)
and lets callers (e.g. the CI workflow or the Flask app) catch specific
failure modes instead of swallowing everything.
"""


class ProjectBaseError(Exception):
    """Base class for all custom exceptions raised by this project."""


class ConfigurationError(ProjectBaseError):
    """Raised when required configuration or environment variables are missing."""


class DataIngestionError(ProjectBaseError):
    """Raised when data loading, validation, or persistence fails."""


class DataPreprocessingError(ProjectBaseError):
    """Raised when text normalization / cleaning fails."""


class FeatureEngineeringError(ProjectBaseError):
    """Raised when feature extraction (e.g. Bag-of-Words) fails."""


class ModelBuildingError(ProjectBaseError):
    """Raised when model training or persistence fails."""


class ModelEvaluationError(ProjectBaseError):
    """Raised when model evaluation or metric logging fails."""


class ModelRegistrationError(ProjectBaseError):
    """Raised when registering or promoting a model in the MLflow registry fails."""
