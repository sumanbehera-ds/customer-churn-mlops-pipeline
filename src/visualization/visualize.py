# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_confusion_matrix(
    confusion_matrix,
    labels=("Not Churn", "Churn"),
    output_path=None,
    title="Confusion Matrix"
):
    """Plot a confusion matrix and optionally save it to disk."""

    matrix = np.asarray(confusion_matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be a square 2D array.")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:g}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_roc_curve(
    y_true,
    y_score,
    output_path=None,
    title="ROC Curve"
):
    """Plot an ROC curve from true labels and positive-class scores."""

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#1f77b4",
        linewidth=2,
        label=f"ROC AUC = {roc_auc:.3f}",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linewidth=1,
        linestyle="--",
        label="Random classifier",
    )

    ax.set(
        title=title,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")

    return fig, ax, roc_auc
