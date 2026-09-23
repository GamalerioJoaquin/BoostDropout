"""Deterministic-friendly classification training loops."""

from collections.abc import Callable
from time import perf_counter

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import ClassificationMetricAccumulator, ClassificationMetrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> ClassificationMetrics:
    training = optimizer is not None
    model.train(training)
    accumulator = ClassificationMetricAccumulator()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            accumulator.update(loss, logits, targets)
    return accumulator.compute()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    device: torch.device,
) -> ClassificationMetrics:
    """Train one epoch and return loss and top-1 accuracy."""
    return _run_epoch(model, loader, loss_fn, device, optimizer)


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> ClassificationMetrics:
    """Evaluate one epoch without modifying model parameters."""
    return _run_epoch(model, loader, loss_fn, device)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    on_epoch_end: Callable[[int, ClassificationMetrics, ClassificationMetrics], None] | None = None,
) -> dict[str, list[float]]:
    """Train a classifier while retaining the thesis-compatible history structure."""
    if epochs < 1:
        raise ValueError("epochs must be at least one")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        validation_metrics = evaluate_one_epoch(model, validation_loader, loss_fn, device)
        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(validation_metrics.loss)
        history["train_acc"].append(train_metrics.accuracy)
        history["val_acc"].append(validation_metrics.accuracy)
        if on_epoch_end is not None:
            on_epoch_end(epoch, train_metrics, validation_metrics)
    return history


def run_classifier_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    on_epoch_end: Callable[[int, ClassificationMetrics, ClassificationMetrics], None],
) -> float:
    """Run the thesis epoch loop and return compute time, excluding artifact callbacks."""
    if epochs < 1:
        raise ValueError("epochs must be at least one")
    elapsed_seconds = 0.0
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        validation_metrics = evaluate_one_epoch(model, validation_loader, loss_fn, device)
        elapsed_seconds += perf_counter() - started
        on_epoch_end(epoch, train_metrics, validation_metrics)
    return elapsed_seconds
