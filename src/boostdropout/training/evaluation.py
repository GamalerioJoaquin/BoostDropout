"""Checkpoint evaluation helpers."""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import ClassificationMetrics
from .trainer import evaluate_one_epoch


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> ClassificationMetrics:
    """Load a model checkpoint and evaluate it on a labelled data loader."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return evaluate_one_epoch(model, loader, loss_fn, device)
