"""One-hidden-layer autoencoder architectures used in the exploratory protocol."""

from torch import Tensor, nn

from .regularization import BoostDropout, DropConnect


class OneHiddenLayerAutoencoder(nn.Module):
    """MNIST autoencoder with optional activation dropout."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, dropout_p: float = 0.0):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode_pre_dropout(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs.flatten(start_dim=1))

    def encode(self, inputs: Tensor) -> Tensor:
        return self.dropout(self.encode_pre_dropout(inputs))

    def decode_logits(self, hidden: Tensor) -> Tensor:
        return self.decoder(hidden)

    def decode(self, hidden: Tensor) -> Tensor:
        return self.decode_logits(hidden).sigmoid()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_pre_regularization = self.encode_pre_dropout(inputs)
        hidden = self.dropout(hidden_pre_regularization)
        logits = self.decode_logits(hidden)
        return logits.sigmoid(), logits, hidden_pre_regularization


class DropConnectAutoencoder(OneHiddenLayerAutoencoder):
    """Autoencoder with DropConnect applied to the encoder weights."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, p: float = 0.5):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = DropConnect(input_dim, hidden_dim, p=p)
        self.reset_parameters()


class BoostDropoutAutoencoder(OneHiddenLayerAutoencoder):
    """Autoencoder with BoostDropout applied to encoder activations."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        p: float = 0.5,
        lambd: float = 0.5,
        mask_normalization: bool = False,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.boost_dropout = BoostDropout(p=p, lambd=lambd, mask_normalization=mask_normalization)

    def encode(self, inputs: Tensor) -> Tensor:
        return self.boost_dropout(self.encode_pre_dropout(inputs))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_pre_regularization = self.encode_pre_dropout(inputs)
        hidden = self.boost_dropout(hidden_pre_regularization)
        logits = self.decode_logits(hidden)
        return logits.sigmoid(), logits, hidden_pre_regularization
