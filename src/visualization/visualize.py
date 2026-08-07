"""Evaluation visualizations: confusion matrix and classification report.

Previously ``src/visualization`` was an empty placeholder (just a
``.gitkeep``) while the actual confusion-matrix/report generation lived
ad-hoc inside a notebook. This module makes that reusable from the CLI or
from other scripts, and writes figures into ``reports/figures`` (already a
DVC-tracked location) instead of ``notebooks/``.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from src.config.config import FIGURES_DIR, REPORTS_DIR
from src.logger import get_logger

logger = get_logger(__name__, log_filename="visualization.log")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str | None = None
) -> str:
    """Render and save a confusion matrix heatmap for the given predictions.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        output_path: Where to save the PNG. Defaults to
            ``reports/figures/confusion_matrix.png``.

    Returns:
        The path the figure was saved to.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or str(FIGURES_DIR / "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["sadness", "happiness"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.debug("Confusion matrix saved to %s", output_path)
    return output_path


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str | None = None
) -> str:
    """Write a text classification report (precision/recall/F1 per class).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        output_path: Where to save the report. Defaults to
            ``reports/classification_report.txt``.

    Returns:
        The path the report was saved to.
    """
    output_path = output_path or str(REPORTS_DIR / "classification_report.txt")
    report = classification_report(
        y_true, y_pred, target_names=["sadness", "happiness"]
    )
    with open(output_path, "w") as file:
        file.write(report)

    logger.debug("Classification report saved to %s", output_path)
    return output_path


def save_metrics_summary(metrics: dict, output_path: str | None = None) -> str:
    """Persist a metrics dict as pretty-printed JSON for quick inspection."""
    output_path = output_path or str(REPORTS_DIR / "metrics_summary.json")
    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)
    return output_path
