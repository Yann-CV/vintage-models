import pytest
import torch

from vintage_models.components.pooling import SumPool2D, SumAddPool2D


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture()
def input():
    return torch.arange(start=0, end=2 * 4 * 4, dtype=torch.float32).reshape(
        (2, 1, 4, 4)
    )


class TestAddPool2D:
    pooler = SumPool2D(
        kernel_size=2,
    )

    def test_simple_usage(self, input):
        output = self.pooler(input)
        assert output.shape == (2, 1, 2, 2)
        assert output[0, 0, 0, 0] == 0 + 1 + 4 + 5

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.pooler.to("cuda")
        self.pooler(input.to("cuda"))


class TestTrainableAddPool2D:
    pooler = SumAddPool2D(
        kernel_size=2, in_channels=1, activation=torch.nn.Sigmoid()
    )

    def test_simple_usage(self, input):
        output = self.pooler(input)
        assert output.shape == (2, 1, 2, 2)
        assert not torch.allclose(output[0, 0, 0, 0], torch.Tensor(0 + 1 + 4 + 5))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.pooler.to("cuda")
        self.pooler(input.to("cuda"))
