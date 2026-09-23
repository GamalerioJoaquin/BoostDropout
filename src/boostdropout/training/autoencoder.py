"""Legacy-compatible autoencoder training protocol."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from boostdropout.models import (
    BoostDropoutAutoencoder,
    DropConnectAutoencoder,
    OneHiddenLayerAutoencoder,
)
from boostdropout.reproducibility import seed_everything

from .artifacts import collect_runtime_metadata, save_json


@dataclass
class AutoencoderRunConfig:
    """Parameters and defaults retained from the exploratory notebook."""

    seed: int = 1234
    dataset_name: str = "MNIST"
    model_type: str = "baseline"
    model_name: str = "baseline"
    input_dim: int = 28 * 28
    hidden_dim: int = 256
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"
    dropout_p: float = 0.0
    dropconnect_p: float = 0.5
    boost_p: float = 0.5
    boost_lambd: float = 0.6
    mask_normalization: bool = False
    batch_size: int = 512
    epochs: int = 60
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    loss_name: str = "BCE"
    weight_decay: float = 0.0
    use_nesterov: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    save_every_epoch: bool = False


def stable_run_hash(config: AutoencoderRunConfig, extra: dict[str, Any] | None = None) -> str:
    """Retain the historical twelve-character run-directory hash."""
    payload = asdict(config).copy()
    if extra is not None:
        payload["extra"] = extra
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_autoencoder(config: AutoencoderRunConfig) -> nn.Module:
    """Instantiate an autoencoder from the thesis-compatible configuration."""
    model_type = config.model_type.lower()
    common = {"input_dim": config.input_dim, "hidden_dim": config.hidden_dim}
    if model_type == "baseline":
        return OneHiddenLayerAutoencoder(**common, dropout_p=0.0)
    if model_type == "dropout":
        return OneHiddenLayerAutoencoder(**common, dropout_p=config.dropout_p)
    if model_type == "dropconnect":
        return DropConnectAutoencoder(**common, p=config.dropconnect_p)
    if model_type == "boostdropout":
        return BoostDropoutAutoencoder(
            **common,
            p=config.boost_p,
            lambd=config.boost_lambd,
            mask_normalization=config.mask_normalization,
        )
    raise ValueError(f"Unsupported model_type: {config.model_type}")


@torch.no_grad()
def evaluate_autoencoder(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Return reconstruction BCE and MSE per example."""
    model.eval()
    total_bce = total_mse = 0.0
    total_items = 0
    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        flat = inputs.view(inputs.size(0), -1)
        reconstruction, logits, _ = model(inputs)
        total_bce += functional.binary_cross_entropy_with_logits(
            logits, flat, reduction="sum"
        ).item()
        total_mse += functional.mse_loss(reconstruction, flat, reduction="sum").item()
        total_items += inputs.size(0)
    if total_items == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    return {"bce_per_example": total_bce / total_items, "mse_per_example": total_mse / total_items}


def _checkpoint(
    path: Path,
    model: nn.Module,
    config: AutoencoderRunConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
            "torch_version": torch.__version__,
        },
        path,
    )


def train_autoencoder(
    config: AutoencoderRunConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    root_runs: str | Path,
    device: torch.device,
    project_root: str | Path = ".",
) -> Path:
    """Run the existing BCE protocol and persist its historical artefact layout."""
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.optimizer.upper() != "ADAM":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    seed_everything(config.seed, deterministic=False)
    run_dir = Path(root_runs) / stable_run_hash(config) / config.model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_autoencoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float | int]] = []
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            flat = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad(set_to_none=True)
            _, logits, _ = model(inputs)
            loss = functional.binary_cross_entropy_with_logits(logits, flat, reduction="mean")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * inputs.size(0)
            count += inputs.size(0)
        if count == 0:
            raise ValueError("Cannot train on an empty data loader")
        row: dict[str, float | int] = {"epoch": epoch, "train_bce_epoch": loss_sum / count}
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            train_metrics = evaluate_autoencoder(model, train_loader, device)
            test_metrics = evaluate_autoencoder(model, test_loader, device)
            row.update(
                train_bce_eval=train_metrics["bce_per_example"],
                train_mse_eval=train_metrics["mse_per_example"],
                test_bce_eval=test_metrics["bce_per_example"],
                test_mse_eval=test_metrics["mse_per_example"],
            )
        history.append(row)
        if config.save_every_epoch:
            _checkpoint(run_dir / f"model_epoch_{epoch:03d}.pt", model, config, epoch, row)
    last_eval = next(row for row in reversed(history) if "test_bce_eval" in row)
    final_metrics = {
        "train_bce_per_example": last_eval["train_bce_eval"],
        "train_mse_per_example": last_eval["train_mse_eval"],
        "test_bce_per_example": last_eval["test_bce_eval"],
        "test_mse_per_example": last_eval["test_mse_eval"],
        "training_seconds": time.monotonic() - started,
        "device": str(device),
    }
    model_path = run_dir / "model.pt"
    history_path = run_dir / "history.json"
    _checkpoint(model_path, model, config, config.epochs, final_metrics)
    save_json(history_path, {"history": history})
    save_json(
        run_dir / "metrics.json",
        {
            "config": asdict(config),
            "final_metrics": final_metrics,
            "paths": {"model_path": str(model_path), "history_path": str(history_path)},
        },
    )
    save_json(
        run_dir / "run_metadata.json",
        {
            "configuration": asdict(config),
            "seeds": {"experiment_seed": config.seed},
            **collect_runtime_metadata(project_root),
            "artifacts": {
                "model": str(model_path),
                "history": str(history_path),
                "metrics": str(run_dir / "metrics.json"),
            },
        },
    )
    return model_path
