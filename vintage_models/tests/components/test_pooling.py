import pytest
import torch

from vintage_models.components.pooling import AddPool2D, TrainableAddPool2D


@pytest.fixture()
def input():
    return torch.arange(start=0, end=2 * 4 * 4, dtype=torch.float32).reshape(
        (2, 1, 4, 4)
    )


class TestAddPool2D:
    def test_simple_usage(self, input):
        pooler = AddPool2D(
            kernel_size=2,
            in_channels=1,
        )
        output = pooler(input)
        assert output.shape == (2, 1, 2, 2)
        assert output[0, 0, 0, 0] == 0 + 1 + 4 + 5


class TestTrainableAddPool2D:
    def test_simple_usage(self, input):
        pooler = TrainableAddPool2D(
            kernel_size=2, in_channels=1, activation=torch.nn.Sigmoid()
        )
        output = pooler(input)
        assert output.shape == (2, 1, 2, 2)
        assert not torch.allclose(output[0, 0, 0, 0], torch.Tensor(0 + 1 + 4 + 5))
