"""Headless integration checks for reusable research figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from boostdropout.models import OneHiddenLayerAutoencoder
from boostdropout.visualization import (
    plot_classification_history,
    plot_coadaptation,
    plot_reconstruction_comparison,
    plot_representational_sparsity,
)
from boostdropout.visualization.autoencoder import make_weight_grid


def test_figures_reconstruct_from_existing_arrays_and_model(tmp_path):
    model = OneHiddenLayerAutoencoder(hidden_dim=8).eval()
    inputs = torch.rand(3, 1, 28, 28)
    history = {
        "train_loss": [1.0, 0.7],
        "val_loss": [1.1, 0.8],
        "train_acc": [0.3, 0.6],
        "val_acc": [0.2, 0.5],
    }
    curves = plot_classification_history(history, save_path=tmp_path / "curves.png")
    filters = plot_coadaptation(model, save_path=tmp_path / "coadaptation.png")
    sparsity, summary = plot_representational_sparsity(
        model, inputs, save_path=tmp_path / "sparsity.png"
    )
    reconstruction = plot_reconstruction_comparison(
        {"baseline": model}, inputs, save_path=tmp_path / "reconstruction.png"
    )
    assert 0 <= summary["fraction_exact_zero"] <= 1
    assert all(
        (tmp_path / name).stat().st_size > 0
        for name in ("curves.png", "coadaptation.png", "sparsity.png", "reconstruction.png")
    )
    for figure in (curves, filters, sparsity, reconstruction):
        plt.close(figure)


def test_filter_grid_preserves_global_symmetric_scaling():
    weights = np.zeros((2, 784), dtype=np.float32)
    weights[0, 0] = -2.0
    weights[1, 0] = 2.0
    grid, _ = make_weight_grid(weights, n_cols=2)
    assert grid[1, 1] == 0.0
    assert grid[1, 30] == 1.0
