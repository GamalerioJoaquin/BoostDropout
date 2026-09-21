"""Validated declarative experiment configuration."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection and deterministic synthetic-data settings."""

    kind: str = "mnist"
    root: str = "data"
    train_fraction: float = 0.10
    validation_fraction: float = 0.25
    normalize: bool = True
    synthetic_samples: int = 64


@dataclass(frozen=True)
class ModelConfig:
    """Classifier architecture and regularization parameters."""

    name: str = "OverfitNet"
    hidden_size: int = 256
    p: float = 0.5
    lambd: float = 0.5
    mask_normalization: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and execution parameters."""

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete validated configuration for one classification experiment."""

    name: str
    output_dir: str = "runs"
    experiment_seed: int = 42
    split_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(payload: dict[str, Any], key: str, type_: type[Any]) -> Any:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping.")
    try:
        return type_(**value)
    except TypeError as error:
        raise ValueError(f"Invalid '{key}' configuration: {error}") from error


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate semantic constraints before any files or compute are created."""
    if config.data.kind not in {"mnist", "synthetic"}:
        raise ValueError("data.kind must be 'mnist' or 'synthetic'.")
    if config.model.name.lower() not in {
        "overfitnet",
        "dropoutnet",
        "dropconnectnet",
        "boostdropoutnet",
    }:
        raise ValueError("model.name is not a supported classification architecture.")
    if config.model.hidden_size < 1 or config.training.epochs < 1:
        raise ValueError("model.hidden_size and training.epochs must be positive.")
    if config.training.batch_size < 1 or config.training.learning_rate <= 0:
        raise ValueError("training.batch_size and training.learning_rate must be positive.")
    if not 0 <= config.model.p <= 1:
        raise ValueError("model.p must be between 0 and 1.")
    if config.model.name.lower() == "dropconnectnet" and config.model.p == 1:
        raise ValueError("DropConnect requires model.p to be less than 1.")
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The configuration document must contain a YAML mapping.")
    try:
        config = ExperimentConfig(
            name=payload["name"],
            output_dir=payload.get("output_dir", "runs"),
            experiment_seed=payload.get("experiment_seed", 42),
            split_seed=payload.get("split_seed", 42),
            data=_section(payload, "data", DataConfig),
            model=_section(payload, "model", ModelConfig),
            training=_section(payload, "training", TrainingConfig),
        )
    except KeyError as error:
        raise ValueError("The configuration requires a 'name' field.") from error
    return validate_config(config)
