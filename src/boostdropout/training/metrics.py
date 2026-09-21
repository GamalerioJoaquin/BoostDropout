"""Metrics shared by training and evaluation loops."""

from dataclasses import asdict, dataclass

import torch


@dataclass
class ClassificationMetrics:
    """Aggregated scalar metrics for one dataset pass."""

    loss: float
    accuracy: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ClassificationMetricAccumulator:
    """Accumulate batch loss and top-1 accuracy without retaining tensors."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.correct = 0
        self.samples = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = targets.size(0)
        self.loss_sum += loss.item() * batch_size
        self.correct += (logits.argmax(dim=1) == targets).sum().item()
        self.samples += batch_size

    def compute(self) -> ClassificationMetrics:
        if self.samples == 0:
            raise ValueError("Cannot compute metrics for an empty data loader.")
        return ClassificationMetrics(
            loss=self.loss_sum / self.samples,
            accuracy=self.correct / self.samples,
            samples=self.samples,
        )
