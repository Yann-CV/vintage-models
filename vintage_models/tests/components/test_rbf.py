import pytest
import torch

from vintage_models.components.rbf import EuclideanDistanceRBF


@pytest.fixture
def input():
    return torch.arange(start=0, end=1 * 3, dtype=torch.float32).reshape((1, 3))


class TestEuclideanDistanceRBF:
    def test_simple(self, input):
        rbf = EuclideanDistanceRBF(3, 10)
        assert rbf(input).shape == (
            1,
            10,
        )
