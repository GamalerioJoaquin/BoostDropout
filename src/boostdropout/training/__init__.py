"""Training, evaluation, metrics, and traceable run artefacts."""

from .artifacts import RunArtifactStore, collect_runtime_metadata
from .autoencoder import AutoencoderRunConfig, build_autoencoder, train_autoencoder
from .evaluation import evaluate_checkpoint
from .trainer import evaluate_one_epoch, run_classifier_epochs, train_classifier, train_one_epoch

__all__ = [
    "AutoencoderRunConfig",
    "RunArtifactStore",
    "build_autoencoder",
    "collect_runtime_metadata",
    "evaluate_checkpoint",
    "evaluate_one_epoch",
    "run_classifier_epochs",
    "train_autoencoder",
    "train_classifier",
    "train_one_epoch",
]
