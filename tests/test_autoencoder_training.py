"""Short runs preserve the existing autoencoder checkpoint schema."""

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from boostdropout.training.autoencoder import AutoencoderRunConfig, train_autoencoder


@pytest.mark.parametrize("model_type", ["baseline", "dropout", "dropconnect", "boostdropout"])
def test_short_autoencoder_run_is_traceable_and_loadable(tmp_path, model_type):
    inputs = torch.rand(8, 1, 28, 28)
    labels = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    config = AutoencoderRunConfig(
        model_type=model_type,
        model_name=model_type,
        hidden_dim=8,
        epochs=1,
        num_workers=0,
    )
    checkpoint_path = train_autoencoder(
        config, loader, loader, tmp_path / "runs", torch.device("cpu"), project_root=tmp_path
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["config"]["model_type"] == model_type
    assert payload["epoch"] == 1
    directory = checkpoint_path.parent
    assert json.loads((directory / "history.json").read_text())["history"]
    metadata = json.loads((directory / "run_metadata.json").read_text())
    assert metadata["seeds"]["experiment_seed"] == 1234
    assert "commit" in metadata["git"]
