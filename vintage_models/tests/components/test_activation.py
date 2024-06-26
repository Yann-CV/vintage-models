import pytest
import torch

from vintage_models.components.activation import ScaledTanh


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture()
def input():
    return torch.Tensor([0.5])


class TestScaledTanh:
    scaled_tanh = ScaledTanh(scale=2)

    def test_simple(self, input):
        assert torch.allclose(self.scaled_tanh(input), torch.tanh(input) * 2)

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.scaled_tanh.to("cuda")
        self.scaled_tanh(input.to("cuda"))
