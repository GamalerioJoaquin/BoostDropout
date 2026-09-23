"""Validated declarative experiment configuration."""

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from re import fullmatch
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
    if not isinstance(config.name, str) or not config.name.strip():
        raise ValueError("name must be a nonempty string.")
    if fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config.name) is None:
        raise ValueError("name may contain only letters, digits, dots, underscores, and hyphens.")
    if not isinstance(config.output_dir, str) or not config.output_dir.strip():
        raise ValueError("output_dir must be a nonempty string.")
    for seed in (config.experiment_seed, config.split_seed):
        if type(seed) is not int or seed < 0 or seed >= 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32).")
    if config.data.kind not in {"mnist", "synthetic"}:
        raise ValueError("data.kind must be 'mnist' or 'synthetic'.")
    if not 0 < config.data.train_fraction <= 1:
        raise ValueError("data.train_fraction must be in (0, 1].")
    if not 0 < config.data.validation_fraction < 1:
        raise ValueError("data.validation_fraction must be in (0, 1).")
    if type(config.data.synthetic_samples) is not int or config.data.synthetic_samples < 2:
        raise ValueError("data.synthetic_samples must be at least two.")
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
    if not isfinite(config.training.learning_rate) or not isfinite(config.training.weight_decay):
        raise ValueError("Optimizer parameters must be finite.")
    if config.training.weight_decay < 0 or config.training.num_workers < 0:
        raise ValueError("weight_decay and num_workers must be nonnegative.")
    if not isinstance(config.training.device, str):
        raise ValueError("training.device must be a string.")
    if config.training.device not in {"cpu", "cuda", "mps"} and not (
        config.training.device.startswith("cuda:") and config.training.device[5:].isdigit()
    ):
        raise ValueError("training.device must be cpu, cuda, cuda:N, or mps.")
    if not 0 <= config.model.p <= 1:
        raise ValueError("model.p must be between 0 and 1.")
    if not isfinite(config.model.lambd):
        raise ValueError("model.lambd must be finite.")
    if config.model.name.lower() == "dropconnectnet" and config.model.p == 1:
        raise ValueError("DropConnect requires model.p to be less than 1.")
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The configuration document must contain a YAML mapping.")
    unknown = set(payload) - {
        "name",
        "output_dir",
        "experiment_seed",
        "split_seed",
        "data",
        "model",
        "training",
        "grid",
    }
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}.")
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
