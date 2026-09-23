"""Integration tests for short, traceable classification runs."""

import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from boostdropout.models import OverfitNet
from boostdropout.training import RunArtifactStore, evaluate_checkpoint, train_classifier


def test_short_run_produces_history_checkpoint_and_traceable_metadata(tmp_path):
    torch.manual_seed(12)
    inputs = torch.randn(12, 1, 28, 28)
    targets = torch.randint(0, 10, (12,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = OverfitNet(hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = RunArtifactStore(tmp_path, "smoke", project_root=tmp_path)
    metadata_path = store.save_metadata({"model": "OverfitNet"}, {"experiment_seed": 12})
    history = train_classifier(
        model, loader, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), epochs=1
    )
    history_path = store.save_history(history)
    checkpoint_path = store.save_checkpoint(model, optimizer, epoch=1, history=history)
    store.finalize()

    assert len(history["train_loss"]) == 1
    assert history_path.exists() and checkpoint_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert {"configuration", "seeds", "git", "runtime", "created_at"} <= metadata.keys()
    assert metadata["artifacts"]["last.pt"]["sha256"]
    assert metadata["runtime"]["boostdropout"]
    metrics = evaluate_checkpoint(
        checkpoint_path,
        OverfitNet(hidden_size=8),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    assert metrics.samples == 12
