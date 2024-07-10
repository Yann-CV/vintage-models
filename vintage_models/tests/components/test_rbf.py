import pytest
import torch

from vintage_models.components.rbf import EuclideanDistanceRBF


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture
def input():
    return torch.arange(start=0, end=1 * 3, dtype=torch.float32).reshape((1, 3))


class TestEuclideanDistanceRBF:
    rbf = EuclideanDistanceRBF(3, 10)

    def test_simple(self, input):
        assert self.rbf(input).shape == (
            1,
            10,
        )

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.rbf.to("cuda")
        self.rbf(input.to("cuda"))
