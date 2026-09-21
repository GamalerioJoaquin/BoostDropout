"""MNIST data access, deterministic splits, and data-loader construction."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from boostdropout.reproducibility import seed_worker_factory


def get_mnist_transform(
    normalize: bool = True, mean: float = 0.1307, std: float = 0.3081
):
    """Return the MNIST tensor transform used by the classification protocol."""
    from torchvision import transforms

    operations = [transforms.ToTensor()]
    if normalize:
        operations.append(transforms.Normalize((mean,), (std,)))
    return transforms.Compose(operations)


def count_classes_from_indices(targets: Sequence[int], indices: Sequence[int]) -> dict[int, int]:
    """Count labels represented by a set of dataset indices."""
    selected_targets = np.asarray(targets)[np.asarray(indices, dtype=int)]
    labels, counts = np.unique(selected_targets, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def create_stratified_split(
    targets: Sequence[int],
    train_fraction: float = 0.10,
    val_fraction: float = 0.25,
    split_seed: int = 42,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the exact per-class random split used by the thesis protocol."""
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    rng = np.random.RandomState(split_seed)
    targets_array = np.asarray(targets)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for label in range(num_classes):
        label_indices = np.where(targets_array == label)[0]
        rng.shuffle(label_indices)
        selected_count = int(round(len(label_indices) * train_fraction))
        validation_count = int(round(selected_count * val_fraction))
        selected = label_indices[:selected_count]
        validation_indices.extend(selected[:validation_count])
        train_indices.extend(selected[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return np.asarray(train_indices, dtype=int), np.asarray(validation_indices, dtype=int)


def _split_cache_path(
    cache_dir: Path,
    cache_prefix: str,
    train_fraction: float,
    val_fraction: float,
    normalize: bool,
    mean: float,
    std: float,
    split_seed: int,
) -> Path:
    filename = (
        f"{cache_prefix}_frac-{train_fraction}_val-{val_fraction}_norm-{normalize}"
        f"_mean-{mean}_std-{std}_splitseed-{split_seed}.npz"
    )
    return cache_dir / filename


def load_small_mnist_data(
    data_dir: str | Path,
    cache_dir: str | Path,
    train_fraction: float = 0.10,
    val_fraction: float = 0.25,
    normalize: bool = True,
    mean: float = 0.1307,
    std: float = 0.3081,
    split_seed: int = 42,
    cache_prefix: str = "mnist",
) -> tuple[Subset, Subset]:
    """Download MNIST if required and return deterministic train/validation subsets."""
    from torchvision import datasets

    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=get_mnist_transform(normalize, mean, std)
    )
    cache_path = _split_cache_path(
        cache_dir,
        cache_prefix,
        train_fraction,
        val_fraction,
        normalize,
        mean,
        std,
        split_seed,
    )
    if cache_path.exists():
        cached = np.load(cache_path)
        train_indices, validation_indices = cached["train_idx"], cached["val_idx"]
    else:
        targets = dataset.targets.cpu().numpy()
        train_indices, validation_indices = create_stratified_split(
            targets,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
        np.savez(cache_path, train_idx=train_indices, val_idx=validation_indices)
    return Subset(dataset, train_indices.tolist()), Subset(dataset, validation_indices.tolist())


def compute_dataset_mean_std(dataset: Dataset) -> tuple[float, float]:
    """Compute scalar mean and standard deviation across a tensor dataset."""
    loader = DataLoader(dataset, batch_size=1_024, shuffle=False)
    channel_sum = 0.0
    squared_sum = 0.0
    total_values = 0
    for inputs, _ in loader:
        channel_sum += inputs.sum().item()
        squared_sum += (inputs**2).sum().item()
        total_values += inputs.numel()
    mean = channel_sum / total_values
    std = (squared_sum / total_values - mean**2) ** 0.5
    return mean, std


def _pin_memory_for(device: torch.device | None, pin_memory: bool | None) -> bool:
    if pin_memory is not None:
        return pin_memory
    return device is not None and device.type == "cuda"


def make_train_validation_dataloaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build reproducible train and validation loaders for the classification protocol."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init_fn = seed_worker_factory(seed)
    resolved_pin_memory = _pin_memory_for(device, pin_memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, validation_loader


def make_test_dataloader(
    test_dataset: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build a deterministic non-shuffled test loader."""
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin_memory_for(device, pin_memory),
        worker_init_fn=seed_worker_factory(seed),
    )


def load_mnist_test_dataset(
    data_dir: str | Path, normalize: bool = True, mean: float = 0.1307, std: float = 0.3081
) -> Dataset:
    """Download MNIST's test split if required and return it as a dataset."""
    from torchvision import datasets

    return datasets.MNIST(
        root=Path(data_dir),
        train=False,
        download=True,
        transform=get_mnist_transform(normalize, mean, std),
    )


def build_mnist_datasets(data_root: str | Path = "./data") -> tuple[Dataset, Dataset]:
    """Build the unnormalized MNIST datasets used by the autoencoder protocol."""
    from torchvision import datasets

    transform = get_mnist_transform(normalize=False)
    root = Path(data_root)
    return (
        datasets.MNIST(root=root, train=True, download=True, transform=transform),
        datasets.MNIST(root=root, train=False, download=True, transform=transform),
    )


def build_autoencoder_mnist_dataloaders(
    data_root: str | Path = "./data",
    batch_size: int = 512,
    test_batch_size: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build the train/test loaders used by the autoencoder protocol."""
    train_dataset, test_dataset = build_mnist_datasets(data_root)
    if test_batch_size is None:
        test_batch_size = batch_size
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **loader_kwargs),
    )
