# Contributing to BoostDropout

Thank you for contributing to BoostDropout. This project is evolving into a
reproducible research codebase, so reviewability and traceability are as
important as feature delivery.

## Before you start

- Read the [architecture guide](docs/architecture.md) and
  [development guide](docs/development.md).
- Keep one conceptual change per branch and Pull Request.
- Do not mix refactoring with experimental or methodological changes.
- Never commit datasets, generated runs, model checkpoints, or local secrets.

## Pull Request expectations

Every Pull Request should explain the motivation, summarize the change, and
include the commands used for verification. Changes that affect experimental
behavior must state their expected effect on reproducibility and include an
appropriate regression or integration test.

## Code and documentation standards

- Keep reusable Python code in `src/boostdropout/`.
- Use notebooks for exploration, analysis, and presentation rather than for
  duplicated business logic.
- Write documentation and user-facing text in English.
- Run Ruff, pytest, and the package build before requesting review.

## Research integrity

The thesis snapshot is a historical reference. New work should clearly identify
the commit, configuration, seeds, dataset split, software environment, and
artifact outputs required to reproduce it.
