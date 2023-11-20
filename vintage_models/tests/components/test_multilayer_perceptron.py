import pytest

import torch

from vintage_models.components.multilayer_perceptron import TwoLayerGeluMLP, TwoLayerMLP


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


class TestTwoLayerMLP:
    mlp = TwoLayerMLP(3, 3, 2)

    def test_simple(self, input):
        output = self.mlp(input)
        assert output.shape == (2, 3)

    def test_wrong_input_size(self, input):
        with pytest.raises(RuntimeError):
            self.mlp(torch.zeros(1, 2))
