"""Reusable, non-interactive research visualizations."""

from .autoencoder import (
    plot_coadaptation,
    plot_hidden_activation_distribution,
    plot_reconstruction_comparison,
    plot_representational_sparsity,
)
from .curves import plot_classification_history

__all__ = [
    "plot_classification_history",
    "plot_coadaptation",
    "plot_hidden_activation_distribution",
    "plot_reconstruction_comparison",
    "plot_representational_sparsity",
]
