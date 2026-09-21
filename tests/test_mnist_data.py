"""Tests for data utilities that do not require downloading MNIST."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from boostdropout.data import (
    compute_dataset_mean_std,
    count_classes_from_indices,
    create_stratified_split,
    make_train_validation_dataloaders,
)


def test_stratified_split_is_repeatable_disjoint_and_balanced():
    targets = np.repeat(np.arange(10), 20)
    train_indices, validation_indices = create_stratified_split(
        targets, train_fraction=0.5, val_fraction=0.2, split_seed=17
    )
    repeated_train, repeated_validation = create_stratified_split(
        targets, train_fraction=0.5, val_fraction=0.2, split_seed=17
    )
    assert np.array_equal(train_indices, repeated_train)
    assert np.array_equal(validation_indices, repeated_validation)
    assert not set(train_indices).intersection(validation_indices)
    assert count_classes_from_indices(targets, train_indices) == {label: 8 for label in range(10)}
    assert count_classes_from_indices(targets, validation_indices) == {
        label: 2 for label in range(10)
    }


def test_loaders_have_deterministic_train_order_and_dataset_statistics():
    features = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    labels = torch.arange(12)
    dataset = TensorDataset(features, labels)
    first_loader, _ = make_train_validation_dataloaders(dataset, dataset, 3, seed=91)
    second_loader, _ = make_train_validation_dataloaders(dataset, dataset, 3, seed=91)
    first_order = torch.cat([batch_labels for _, batch_labels in first_loader])
    second_order = torch.cat([batch_labels for _, batch_labels in second_loader])
    assert torch.equal(first_order, second_order)
    mean, std = compute_dataset_mean_std(dataset)
    assert mean == torch.mean(features).item()
    assert np.isclose(std, torch.std(features, unbiased=False).item())
