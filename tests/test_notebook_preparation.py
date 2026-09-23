"""Notebook preparation cells remain executable without fetching MNIST."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]


def _execute_cells(name: str, indices: list[int], namespace: dict) -> dict:
    notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
    for index in indices:
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"{name}:cell-{index}", "exec"), namespace)
    return namespace


def test_main_protocol_preparation_and_compatibility_wrappers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = _execute_cells("main_protocol.ipynb", [2, 4, 6, 8, 9, 11, 13, 16, 17], {})
    inputs = torch.randn(8, 1, 28, 28)
    targets = torch.randint(0, 10, (8,))
    dataset = TensorDataset(inputs, targets)
    train, validation = namespace["make_dataloaders"](
        dataset, dataset, batch_size=4, num_workers=0, experiment_seed=42
    )
    assert len(next(iter(train))[0]) == 4
    assert len(next(iter(validation))[0]) == 4
    path = tmp_path / "artifact.json"
    namespace["save_json"]({"result": 1}, path)
    assert json.loads(path.read_text()) == {"result": 1}
    assert namespace["build_model"]("BoostDropoutNet", hidden_size=8)(inputs).shape == (8, 10)


def test_autoencoder_protocol_definitions_load_without_dataset_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = _execute_cells("autoencoder_protocol.ipynb", [2, 4, 8, 10, 11, 13, 15, 34, 47], {})
    config = namespace["RunConfig"](model_type="baseline", hidden_dim=8)
    assert namespace["build_model"](config)(torch.rand(2, 1, 28, 28))[0].shape == (2, 784)
    assert len(namespace["stable_run_hash"](config)) == 12


def test_main_protocol_short_training_preserves_legacy_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = _execute_cells("main_protocol.ipynb", [2, 4, 6, 8, 9, 11, 13, 16, 17], {})
    dataset = TensorDataset(torch.rand(12, 1, 28, 28), torch.arange(12) % 10)
    namespace["load_small_data"] = lambda **kwargs: (
        Subset(dataset, list(range(8))),
        Subset(dataset, list(range(8, 12))),
    )
    namespace["datasets"] = SimpleNamespace(MNIST=lambda **kwargs: dataset)
    namespace["evaluate_all_artifacts_on_test"] = lambda training_dir: {}
    history, model = namespace["train_model_on_mnist_subset"](
        model_name="OverfitNet",
        hidden_size=8,
        num_epochs=1,
        batch_size=4,
        num_workers=0,
        normalize=False,
        plot_curves=False,
        save_training_plots=False,
        save_final_csv=False,
        base_save_dir=tmp_path / "runs",
    )
    run_directory = tmp_path / "runs" / history["training_id"]
    assert model.name == "OverfitNet"
    assert (run_directory / "best_model.pth").exists()
    assert (run_directory / "last_model.pth").exists()
    assert (run_directory / "run_metadata.json").exists()


def test_reconstruction_notebooks_load_their_preparation_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
        "reconstruct_gridsearch_comparison_plots.ipynb",
        "reconstruct_final_evaluation_plots.ipynb",
    ):
        namespace = _execute_cells(name, [1, 2, 3, 4, 5, 6], {})
        assert "plt" in namespace
