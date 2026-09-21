"""Behavioural tests for the extracted model implementations."""

import pytest

torch = pytest.importorskip("torch")

from boostdropout.models import (  # noqa: E402
    BoostDropout,
    BoostDropoutAutoencoder,
    BoostDropoutNet,
    DropConnect,
    DropConnectAutoencoder,
    DropConnectNet,
    DropoutNet,
    OneHiddenLayerAutoencoder,
    OverfitNet,
    build_classification_model,
)


@pytest.mark.parametrize(
    ("model_class", "kwargs"),
    [
        (OverfitNet, {}),
        (DropoutNet, {"p": 0.2}),
        (DropConnectNet, {"p": 0.2}),
        (BoostDropoutNet, {"p": 0.2, "lambd": 0.6}),
    ],
)
def test_classifiers_return_mnist_logits(model_class, kwargs):
    model = model_class(hidden_size=16, **kwargs)
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 10)


def test_boost_dropout_matches_expected_scale_at_probability_one():
    layer = BoostDropout(p=1.0, lambd=0.5)
    layer.train()
    inputs = torch.tensor([[-2.0, 4.0]])
    assert torch.equal(layer(inputs), inputs * 1.5)


def test_boost_dropout_is_identity_when_evaluating_or_disabled():
    inputs = torch.randn(4, 5)
    enabled = BoostDropout(p=0.5, lambd=0.6)
    enabled.eval()
    assert torch.equal(enabled(inputs), inputs)
    assert torch.equal(BoostDropout(p=0.0, lambd=0.6)(inputs), inputs)


def test_dropconnect_is_dense_during_evaluation():
    layer = DropConnect(3, 2, p=0.5)
    layer.eval()
    inputs = torch.randn(4, 3)
    assert torch.equal(layer(inputs), torch.nn.functional.linear(inputs, layer.weight, layer.bias))


def test_factory_and_autoencoder_interfaces():
    assert isinstance(build_classification_model("BoostDropoutNet"), BoostDropoutNet)
    with pytest.raises(ValueError, match="Unknown model"):
        build_classification_model("unknown")

    inputs = torch.randn(2, 1, 28, 28)
    for model in (
        OneHiddenLayerAutoencoder(hidden_dim=12),
        DropConnectAutoencoder(hidden_dim=12, p=0.2),
        BoostDropoutAutoencoder(hidden_dim=12, p=0.2, lambd=0.5),
    ):
        reconstruction, logits, hidden = model(inputs)
        assert reconstruction.shape == logits.shape == (2, 784)
        assert hidden.shape == (2, 12)
