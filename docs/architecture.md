# Architecture and research boundaries

## Purpose

BoostDropout is evolving from a thesis-oriented notebook repository into a
reproducible research package for studying stochastic regularization in neural
networks. The refactor is intentionally incremental: every migration step must
be independently testable and must preserve the established experimental
protocol unless a later research change explicitly states otherwise.

## Historical baseline

The academic snapshot used for the thesis is preserved in the original
repository at commit
[`2d6fd9636b69670b5d7efddcd50630cb6c04b8c5`](https://github.com/joacogamalerio/tesis-project-repository/commit/2d6fd9636b69670b5d7efddcd50630cb6c04b8c5).
The published manuscript is available through the
[UNC institutional repository](https://rdu.unc.edu.ar/items/ca464d63-34b0-4314-897e-17bd103af950).

## Current boundary

At this stage, the notebooks remain the source of the experimental logic. The
`src/boostdropout` package establishes an importable and testable boundary but
does not yet contain model, training, or data code. This is deliberate: the
first phase introduces engineering controls before any scientific behavior is
migrated.

## Target architecture

```text
src/boostdropout/
├── models/          # Baseline, Dropout, DropConnect, and BoostDropout
├── data/            # Datasets, transforms, splits, and normalization
├── training/        # Seeding, loops, evaluation, metrics, and artifacts
├── experiments/     # Reproducible experiment orchestration
└── visualization/   # Curves, sparsity, co-adaptation, reconstructions
```

Notebooks will become consumers of these modules. They will remain useful for
exploration and analysis, but reusable logic must live in `src/`.

## Reproducibility contract

Every future experiment run must record:

- the effective configuration and configuration source;
- the Git commit and package version;
- random seeds for Python, NumPy, PyTorch, and data loading;
- dataset and split identifiers;
- software and hardware environment details; and
- paths and checksums for the generated artifacts when practical.

## Quality gates

Changes must pass the checks defined in [Development guide](development.md):
formatting and linting with Ruff, automated tests with pytest, and a package
build. The CI workflow enforces these checks on pull requests and `main`.
