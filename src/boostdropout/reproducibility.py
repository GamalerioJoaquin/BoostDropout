"""Explicit random-state controls for reproducible experiments."""

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch

DEFAULT_SPLIT_SEED = 42
DEFAULT_EXPERIMENT_SEED = 42


@dataclass(frozen=True)
class SeedConfiguration:
    """Serializable seed settings that define an experimental run."""

    experiment_seed: int = DEFAULT_EXPERIMENT_SEED
    split_seed: int = DEFAULT_SPLIT_SEED
    deterministic_algorithms: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        """Return plain values suitable for run metadata."""
        return asdict(self)


def set_global_determinism(seed: int, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy and PyTorch and configure deterministic cuDNN settings."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


def seed_everything(seed: int = 1234, deterministic: bool = False) -> None:
    """Seed all random sources using the autoencoder protocol's configuration."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker_factory(base_seed: int) -> Callable[[int], None]:
    """Create a deterministic ``DataLoader`` worker initialization function."""

    def seed_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker
