"""Declarative classification experiment execution."""

from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .configuration import ExperimentConfig, ModelConfig, validate_config
from .data import load_small_mnist_data, make_train_validation_dataloaders
from .models import build_classification_model
from .reproducibility import SeedConfiguration, set_global_determinism
from .training import RunArtifactStore, evaluate_checkpoint, train_classifier


def _synthetic_datasets(config: ExperimentConfig) -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(config.experiment_seed)
    samples = config.data.synthetic_samples
    inputs = torch.randn(samples, 1, 28, 28, generator=generator)
    targets = torch.randint(0, 10, (samples,), generator=generator)
    split = max(1, int(samples * (1 - config.data.validation_fraction)))
    return (
        TensorDataset(inputs[:split], targets[:split]),
        TensorDataset(inputs[split:], targets[split:]),
    )


def _datasets(config: ExperimentConfig):
    if config.data.kind == "synthetic":
        return _synthetic_datasets(config)
    cache_dir = Path(config.output_dir) / "cache"
    return load_small_mnist_data(
        data_dir=config.data.root,
        cache_dir=cache_dir,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.validation_fraction,
        normalize=config.data.normalize,
        split_seed=config.split_seed,
    )


def run_experiment(config: ExperimentConfig, project_root: str | Path = ".") -> Path:
    """Run one validated configuration and return its traceable artefact directory."""
    validate_config(config)
    set_global_determinism(config.experiment_seed)
    device = torch.device(config.training.device)
    train_dataset, validation_dataset = _datasets(config)
    train_loader, validation_loader = make_train_validation_dataloaders(
        train_dataset,
        validation_dataset,
        batch_size=config.training.batch_size,
        seed=config.experiment_seed,
        num_workers=config.training.num_workers,
        device=device,
    )
    model = build_classification_model(
        config.model.name,
        hidden_size=config.model.hidden_size,
        p=config.model.p,
        lambd=config.model.lambd,
        mask_normalization=config.model.mask_normalization,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    store = RunArtifactStore(config.output_dir, config.name, project_root=project_root)
    seeds = SeedConfiguration(config.experiment_seed, config.split_seed).to_dict()
    store.save_metadata(config.to_dict(), seeds)
    history = train_classifier(
        model,
        train_loader,
        validation_loader,
        nn.CrossEntropyLoss(),
        optimizer,
        device,
        config.training.epochs,
    )
    store.save_history(history)
    store.save_checkpoint(model, optimizer, config.training.epochs, history)
    store.finalize()
    return store.path


def evaluate_run(run_dir: str | Path, config: ExperimentConfig) -> dict[str, float | int]:
    """Evaluate a saved run using the declared dataset and model configuration."""
    device = torch.device(config.training.device)
    _, validation_dataset = _datasets(config)
    _, validation_loader = make_train_validation_dataloaders(
        validation_dataset,
        validation_dataset,
        batch_size=config.training.batch_size,
        seed=config.experiment_seed,
        num_workers=config.training.num_workers,
        device=device,
    )
    model = build_classification_model(
        config.model.name,
        hidden_size=config.model.hidden_size,
        p=config.model.p,
        lambd=config.model.lambd,
        mask_normalization=config.model.mask_normalization,
    )
    return evaluate_checkpoint(
        Path(run_dir) / "last.pt", model, validation_loader, nn.CrossEntropyLoss(), device
    ).to_dict()


def run_grid_search(config: ExperimentConfig, parameter_grid: dict[str, list[Any]]) -> list[Path]:
    """Run a compact grid over model fields while retaining one artefact per trial."""
    unknown = set(parameter_grid) - {"p", "lambd", "hidden_size"}
    if unknown:
        raise ValueError(f"Unsupported grid parameters: {', '.join(sorted(unknown))}.")
    if not parameter_grid or any(not values for values in parameter_grid.values()):
        raise ValueError("The grid must contain at least one nonempty parameter list.")
    keys = list(parameter_grid)
    paths: list[Path] = []
    for values in product(*(parameter_grid[key] for key in keys)):
        overrides = dict(zip(keys, values, strict=True))
        model = ModelConfig(**{**config.model.__dict__, **overrides})
        trial = ExperimentConfig(**{**config.__dict__, "model": model})
        paths.append(run_experiment(trial))
    return paths
