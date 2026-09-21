# Running declarative experiments

The `boostdropout` command executes a validated YAML configuration and writes a
self-contained run directory. Each run records the full configuration, seeds,
Git commit and working-tree state, Python and PyTorch versions, metric history,
and a checkpoint.

Install the required optional dependencies:

```bash
python -m pip install -e ".[torch,config]"
```

Run the committed CPU-only smoke configuration:

```bash
boostdropout train --config configs/classification-smoke.yaml
```

The smoke configuration intentionally uses deterministic synthetic data. It is
appropriate for checking the software path, not for reporting scientific
results. Use `data.kind: mnist` for an MNIST experiment.

Given the printed run directory, evaluate its checkpoint or reproduce its
portable validation-loss figure:

```bash
boostdropout evaluate --config configs/classification-smoke.yaml --run-dir runs/<run-id>
boostdropout reproduce-figures --run-dir runs/<run-id>
```

For a compact parameter grid, add a `grid` mapping to a configuration. The
supported model keys are `p`, `lambd`, and `hidden_size`:

```yaml
grid:
  p: [0.2, 0.5]
  lambd: [0.4, 0.6]
```

Run it with `boostdropout grid-search --config path/to/experiment.yaml`. Every
trial receives an independent, traceable run directory.
