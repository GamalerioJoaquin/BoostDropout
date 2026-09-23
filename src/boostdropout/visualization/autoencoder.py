"""Autoencoder analysis figures based on explicit models and input batches."""

from collections.abc import Mapping
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from torch import nn


def _save(figure: Figure, save_path: str | Path | None, dpi: int = 220) -> None:
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")


def make_weight_grid(
    weights: np.ndarray,
    n_cols: int = 16,
    image_shape: tuple[int, int] = (28, 28),
    pad: int = 1,
    gray_floor: float = 0.25,
) -> tuple[np.ndarray, LinearSegmentedColormap]:
    """Tile encoder weights with the thesis's global symmetric scaling."""
    array = np.asarray(weights, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != image_shape[0] * image_shape[1]:
        raise ValueError("Weights must have one flattened image per row")
    if n_cols < 1 or pad < 0:
        raise ValueError("n_cols must be positive and pad nonnegative")
    scale = float(np.max(np.abs(array)))
    normalized = np.full_like(array, 0.5) if scale < 1e-8 else (array / scale + 1.0) / 2.0
    rows = ceil(len(array) / n_cols)
    height, width = image_shape
    grid = np.full(
        (rows * height + (rows + 1) * pad, n_cols * width + (n_cols + 1) * pad),
        -1.0,
        dtype=np.float32,
    )
    for index, flat in enumerate(normalized):
        top = pad + index // n_cols * (height + pad)
        left = pad + index % n_cols * (width + pad)
        grid[top : top + height, left : left + width] = flat.reshape(image_shape)
    floor = float(np.clip(gray_floor, 0.0, 1.0))
    cmap = LinearSegmentedColormap.from_list(
        "gray_floor", [(floor, floor, floor), (1.0, 1.0, 1.0)], N=256
    ).with_extremes(under="black")
    return grid, cmap


def plot_coadaptation(
    model: nn.Module,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 10),
    gray_floor: float = 0.25,
) -> Figure:
    """Plot the input-space encoder filters of a fitted autoencoder."""
    weights = model.encoder.weight.detach().cpu().numpy()
    grid, cmap = make_weight_grid(weights, gray_floor=gray_floor)
    figure, axis = plt.subplots(figsize=figsize)
    axis.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
    axis.axis("off")
    _save(figure, save_path, dpi=200)
    return figure


@torch.no_grad()
def plot_representational_sparsity(
    model: nn.Module,
    inputs: torch.Tensor,
    save_path: str | Path | None = None,
    bins_mean: int = 60,
    bins_activation: int = 80,
    activation_xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8.2, 5.8),
) -> tuple[Figure, dict[str, float]]:
    """Plot hidden activations before regularization for a fixed input batch."""
    model.eval()
    device = next(model.parameters()).device
    hidden = model.encode_pre_dropout(inputs.to(device)).detach().cpu().numpy()
    return plot_hidden_activation_distribution(
        hidden,
        save_path=save_path,
        bins_mean=bins_mean,
        bins_activation=bins_activation,
        activation_xlim=activation_xlim,
        figsize=figsize,
    )


def plot_hidden_activation_distribution(
    hidden: np.ndarray,
    save_path: str | Path | None = None,
    bins_mean: int = 60,
    bins_activation: int = 80,
    activation_xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (8.2, 5.8),
) -> tuple[Figure, dict[str, float]]:
    """Plot and summarize a precomputed batch of hidden activations."""
    hidden = np.asarray(hidden)
    if hidden.ndim != 2 or hidden.shape[0] == 0:
        raise ValueError("Expected a nonempty batch of hidden activations")
    per_unit = hidden.mean(axis=0)
    activations = hidden.ravel()
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].hist(per_unit, bins=bins_mean, edgecolor="black", linewidth=0.6)
    axes[0].set(ylabel="Count", title="Mean activation")
    bins = (
        np.linspace(*activation_xlim, bins_activation + 1)
        if activation_xlim is not None
        else bins_activation
    )
    axes[1].hist(activations, bins=bins, edgecolor="black", linewidth=0.6)
    axes[1].set(title="Activation")
    if activation_xlim is not None:
        axes[1].set_xlim(*activation_xlim)
    figure.tight_layout()
    _save(figure, save_path)
    summary = {
        "mean_activation_mean": float(per_unit.mean()),
        "mean_activation_std": float(per_unit.std()),
        "all_activation_mean": float(activations.mean()),
        "all_activation_std": float(activations.std()),
        "fraction_exact_zero": float((activations == 0.0).mean()),
        "fraction_gt_1": float((activations > 1.0).mean()),
        "fraction_gt_2": float((activations > 2.0).mean()),
    }
    return figure, summary


@torch.no_grad()
def plot_reconstruction_comparison(
    models: Mapping[str, nn.Module],
    inputs: torch.Tensor,
    save_path: str | Path | None = None,
    image_shape: tuple[int, int] = (28, 28),
) -> Figure:
    """Display identical inputs and reconstructions from several autoencoders."""
    if not models or inputs.ndim != 4 or len(inputs) == 0:
        raise ValueError("Provide at least one model and a nonempty NCHW input batch")
    rows, columns = len(models) + 1, len(inputs)
    figure, axes = plt.subplots(rows, columns, figsize=(1.5 * columns, 1.7 * rows), squeeze=False)
    images: list[tuple[str, np.ndarray]] = [("Input", inputs.cpu().numpy())]
    for label, model in models.items():
        model.eval()
        device = next(model.parameters()).device
        reconstruction, _, _ = model(inputs.to(device))
        images.append((label, reconstruction.cpu().numpy().reshape(-1, 1, *image_shape)))
    for row, (label, batch) in enumerate(images):
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center")
        for column in range(columns):
            axes[row, column].imshow(batch[column, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.tight_layout()
    _save(figure, save_path)
    return figure
