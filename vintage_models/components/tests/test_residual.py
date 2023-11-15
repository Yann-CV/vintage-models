import pytest

import torch

from torch.nn import LayerNorm
from vintage_models.components.attention import ScaledDotProductAttention
from vintage_models.components.residual import ResidualWithSelfAttention


@pytest.fixture
def input():
    return torch.tensor(data=[[0, 0.5, 0, 0], [0.5, 0, 0.5, 0.5]], dtype=torch.float32)


class TestScaledDotProductAttention:
    residual = ResidualWithSelfAttention(
        layers=[LayerNorm(4), ScaledDotProductAttention(dk=4)],
    )

    def test_simple(self, input):
        output = self.residual(input)
        assert output.shape == (2, 4)
        assert not torch.allclose(output, input)
