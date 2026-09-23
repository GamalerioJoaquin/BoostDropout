"""Training-history figures shared by scripts and notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_classification_history(
    history: dict[str, list[float]],
    metric: str = "loss",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot loss, accuracy, or error from a thesis-compatible history."""
    if metric not in {"loss", "accuracy", "error"}:
        raise ValueError("metric must be 'loss', 'accuracy', or 'error'")
    if metric == "loss":
        train, validation = history["train_loss"], history["val_loss"]
        label = "Loss"
    else:
        train, validation = history["train_acc"], history["val_acc"]
        if metric == "error":
            train = [1.0 - value for value in train]
            validation = [1.0 - value for value in validation]
        label = metric.title()
    if len(train) != len(validation) or not train:
        raise ValueError("Training and validation histories must be nonempty and equally long")

    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train) + 1)
    axis.plot(epochs, train, label=f"Train {label}")
    axis.plot(epochs, validation, linestyle="--", label=f"Validation {label}")
    axis.set(xlabel="Epoch", ylabel=label, title=f"{label} curves")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200, bbox_inches="tight")
    return figure
