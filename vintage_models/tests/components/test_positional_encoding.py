import pytest

import torch

from vintage_models.components.positional_encoding import (
    PositionalEncoding1D,
    LearnablePositionalEncoding1D,
)


@pytest.fixture
def input():
    return torch.tensor(data=[[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)


class TestPositionalEncoding1D:
    encoding = PositionalEncoding1D(sequence_len=2, n=100, embedding_len=4)

    def test_simple(self, input):
        output = self.encoding(input)
        assert output.shape == (2, 4)
        assert not torch.allclose(output, input)


class TestLearnablePositionalEncoding1D:
    encoding = LearnablePositionalEncoding1D(sequence_len=2, embedding_len=4)

    def test_simple(self, input):
        output = self.encoding(input)
        assert output.shape == (2, 4)
        assert not torch.allclose(output, input)

    def test_wrong_sequence_length(self):
        with pytest.raises(ValueError):
            self.encoding(torch.zeros(3, 4))
