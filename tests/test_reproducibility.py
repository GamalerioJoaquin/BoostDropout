"""Tests for explicit control of experimental random state."""

import random

import numpy as np
import torch

from boostdropout.reproducibility import (
    SeedConfiguration,
    seed_worker_factory,
    set_global_determinism,
)


def test_global_seed_reproduces_all_random_sources():
    set_global_determinism(73)
    first = (random.random(), np.random.rand(), torch.rand(3))
    set_global_determinism(73)
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_worker_factory_and_seed_metadata_are_deterministic():
    seed_worker_factory(41)(2)
    observed = (random.random(), np.random.rand())
    seed_worker_factory(41)(2)
    assert observed == (random.random(), np.random.rand())
    assert SeedConfiguration(experiment_seed=7, split_seed=11).to_dict() == {
        "experiment_seed": 7,
        "split_seed": 11,
        "deterministic_algorithms": False,
    }
