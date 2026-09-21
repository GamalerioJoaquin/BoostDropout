"""MNIST classification architectures used in the thesis protocol."""

from torch import Tensor, nn
from torch.nn import functional as functional

from .regularization import BoostDropout, DropConnect


class OverfitNet(nn.Module):
    """Four-layer fully connected MNIST baseline without regularization."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.name = "OverfitNet"
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.flatten(inputs)
        outputs = functional.relu(self.fc1(outputs))
        outputs = functional.relu(self.fc2(outputs))
        outputs = functional.relu(self.fc3(outputs))
        return self.fc4(outputs)


class DropoutNet(OverfitNet):
    """Baseline architecture with dropout after its first two hidden layers."""

    def __init__(self, hidden_size: int = 256, p: float = 0.5):
        super().__init__(hidden_size=hidden_size)
        self.name = "DropoutNet"
        self.dropout = nn.Dropout(p=p)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.flatten(inputs)
        outputs = self.dropout(functional.relu(self.fc1(outputs)))
        outputs = self.dropout(functional.relu(self.fc2(outputs)))
        outputs = functional.relu(self.fc3(outputs))
        return self.fc4(outputs)


class BoostDropoutNet(OverfitNet):
    """Baseline architecture with BoostDropout after its first two hidden layers."""

    def __init__(
        self,
        hidden_size: int = 256,
        p: float = 0.5,
        lambd: float = 0.5,
        mask_normalization: bool = False,
    ):
        super().__init__(hidden_size=hidden_size)
        self.name = "BoostDropoutNet"
        self.boost_dropout = BoostDropout(
            p=p, lambd=lambd, mask_normalization=mask_normalization
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.flatten(inputs)
        outputs = self.boost_dropout(functional.relu(self.fc1(outputs)))
        outputs = self.boost_dropout(functional.relu(self.fc2(outputs)))
        outputs = functional.relu(self.fc3(outputs))
        return self.fc4(outputs)


class DropConnectNet(nn.Module):
    """MNIST classifier with DropConnect on the first two affine layers."""

    def __init__(self, hidden_size: int = 256, p: float = 0.5):
        super().__init__()
        self.name = "DropConnectNet"
        self.flatten = nn.Flatten()
        self.fc1 = DropConnect(28 * 28, hidden_size, p=p)
        self.fc2 = DropConnect(hidden_size, hidden_size, p=p)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.flatten(inputs)
        outputs = functional.relu(self.fc1(outputs))
        outputs = functional.relu(self.fc2(outputs))
        outputs = functional.relu(self.fc3(outputs))
        return self.fc4(outputs)


def build_classification_model(
    model_name: str,
    hidden_size: int = 256,
    p: float = 0.5,
    lambd: float = 0.5,
    mask_normalization: bool = False,
) -> nn.Module:
    """Build one of the classification models used by the protocol."""
    factories = {
        "overfitnet": lambda: OverfitNet(hidden_size=hidden_size),
        "dropoutnet": lambda: DropoutNet(hidden_size=hidden_size, p=p),
        "dropconnectnet": lambda: DropConnectNet(hidden_size=hidden_size, p=p),
        "boostdropoutnet": lambda: BoostDropoutNet(
            hidden_size=hidden_size,
            p=p,
            lambd=lambd,
            mask_normalization=mask_normalization,
        ),
    }
    try:
        return factories[model_name.lower()]()
    except KeyError as error:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}.") from error
