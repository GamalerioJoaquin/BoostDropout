# Project commands

This directory is reserved for thin command-line entry points.

During the engineering-foundation phase, no executable workflow is introduced.
In Phase 5, scripts will delegate to the `boostdropout` package for training,
hyperparameter search, evaluation, and figure reconstruction. They must not
duplicate experimental logic.
