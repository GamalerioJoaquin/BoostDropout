"""Stochastic regularization layers used in the experiments."""

import torch
from torch import nn
from torch.nn import functional as functional


class BoostDropout(nn.Module):
    """Amplify independently sampled activations during training.

    Selected activations are multiplied by ``1 + lambd`` with probability ``p``.
    Optional normalization preserves the expected activation scale.
    """

    def __init__(self, p: float = 0.5, lambd: float = 0.2, mask_normalization: bool = False):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        self.p = p
        self.lambd = lambd
        self.mask_normalization = mask_normalization

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return inputs

        mask = torch.ones_like(inputs)
        mask[torch.rand_like(inputs) < self.p] = 1 + self.lambd
        if self.mask_normalization:
            return inputs * (mask / (1 + self.p * self.lambd))
        return inputs * mask


class DropConnect(nn.Linear):
    """A linear layer that stochastically masks weights during training."""

    def __init__(self, in_features: int, out_features: int, p: float = 0.5, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        if not 0 <= p < 1:
            raise ValueError("p must satisfy 0 <= p < 1")
        self.p = p

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.training and self.p > 0:
            keep_probability = 1 - self.p
            mask = (torch.rand_like(weight) < keep_probability).to(weight.dtype)
            weight = weight * mask / keep_probability
        return functional.linear(inputs, weight, self.bias)
