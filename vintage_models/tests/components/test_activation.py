import pytest
import torch

from vintage_models.components.activation import ScaledTanh


@pytest.fixture()
def input():
    return torch.Tensor([0.5])


class TestScaledTanh:
    def test_simple(self, input):
        scaled_tanh = ScaledTanh(scale=2)
        assert torch.allclose(scaled_tanh(input), torch.tanh(input) * 2)
