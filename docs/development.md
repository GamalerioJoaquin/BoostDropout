# Development guide

## Supported Python versions

The engineering foundation supports Python 3.10 and 3.11. Continuous
integration validates both versions.

## Set up a development environment

```bash
python -m venv .venv
```

Activate the virtual environment using the command appropriate for your shell,
then install the package and development tooling:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch,config,visualization]"
python -m pre_commit install
```

To work with the existing notebooks, also install their optional dependencies:

```bash
python -m pip install -e ".[notebooks,torch,config,visualization,dev]"
```

PyTorch and torchvision remain explicit installations because their appropriate
build depends on the target CPU or CUDA environment. Follow the official
PyTorch installation instructions for your platform.

## Required checks

Run these commands before opening a pull request:

```bash
ruff check .
python -m pytest
python -m build
```

Run all configured pre-commit hooks when changing repository metadata or
documentation:

```bash
python -m pre_commit run --all-files
```

## Scope discipline during the refactor

The restructuring plan separates engineering extraction from research changes.
Do not combine a refactor with altered data, hyperparameters, architecture
behavior, or methodological claims. Record validation evidence for every phase
in the local restructuring progress log.

## Release readiness

Before tagging a release, run the quality checks above from a clean virtual
environment and execute `boostdropout train --config
configs/classification-smoke.yaml`. See [Versioning and release policy](versioning.md)
and the [changelog](../CHANGELOG.md).
