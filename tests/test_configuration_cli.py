"""End-to-end checks for the declarative configuration and CLI."""

import json
from pathlib import Path

import pytest

from boostdropout.cli import main
from boostdropout.configuration import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    load_experiment_config,
    validate_config,
)


def _config(path: Path) -> Path:
    path.write_text(
        """name: cli-smoke
output_dir: OUTPUT_DIR
experiment_seed: 19
split_seed: 19
data:
  kind: synthetic
  synthetic_samples: 16
  validation_fraction: 0.25
model:
  name: OverfitNet
  hidden_size: 8
training:
  epochs: 1
  batch_size: 4
  device: cpu
""".replace("OUTPUT_DIR", str(path.parent / "runs").replace("\\", "/")),
        encoding="utf-8",
    )
    return path


def test_configured_short_run_evaluation_and_figure_reproduction(tmp_path, capsys):
    config_path = _config(tmp_path / "experiment.yaml")
    assert load_experiment_config(config_path).name == "cli-smoke"
    assert main(["train", "--config", str(config_path)]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    assert (run_dir / "metadata.json").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["configuration"]["name"] == "cli-smoke"
    assert main(["evaluate", "--config", str(config_path), "--run-dir", str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 4
    assert main(["reproduce-figures", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "validation-loss.svg").exists()


@pytest.mark.parametrize("command", ["train", "grid-search", "evaluate", "reproduce-figures"])
def test_each_cli_command_exposes_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "config",
    [
        ExperimentConfig(name="../outside"),
        ExperimentConfig(name="valid", data=DataConfig(validation_fraction=1)),
        ExperimentConfig(name="valid", data=DataConfig(synthetic_samples=1)),
        ExperimentConfig(name="valid", model=ModelConfig(name="DropConnectNet", p=1)),
    ],
)
def test_invalid_configuration_is_rejected_before_a_run(config):
    with pytest.raises(ValueError):
        validate_config(config)
