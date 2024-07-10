import pytest

import torch
from torch.nn import Sigmoid

from vintage_models.components.multilayer_perceptron import (
    TwoLayerGeluMLP,
    TwoLayerMLP,
    LinearWithActivation,
    MaxOut,
)


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture
def input():
    return torch.tensor(data=[[0, 0.5, 0], [0.5, 0.5, 0]], dtype=torch.float32)


class TestTwoLayerGeluMLP:
    mlp = TwoLayerGeluMLP(3, 3, 2)

    def test_simple(self, input):
        output = self.mlp(input)
        assert output.shape == (2, 3)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.mlp(torch.zeros(1, 2))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.mlp.to("cuda")
        self.mlp(input.to("cuda"))


class TestTwoLayerMLP:
    mlp = TwoLayerMLP(3, 3, 2)

    def test_simple(self, input):
        output = self.mlp(input)
        assert output.shape == (2, 3)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.mlp(torch.zeros(1, 2))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.mlp.to("cuda")
        self.mlp(input.to("cuda"))


class TestLinearWithActivation:
    linear_with_activation = LinearWithActivation(3, 2, Sigmoid())

    def test_simple(self, input):
        output = self.linear_with_activation(input)
        assert output.shape == (2, 2)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.linear_with_activation(torch.zeros(1, 2))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.linear_with_activation.to("cuda")
        self.linear_with_activation(input.to("cuda"))


class TestMaxOut:
    linear_with_max_out = MaxOut(3, 2, 4)

    def test_simple(self, input):
        output = self.linear_with_max_out(input)
        assert output.shape == (2, 2)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.linear_with_max_out(torch.zeros(1, 2))
