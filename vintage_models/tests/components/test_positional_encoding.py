import pytest

import torch

from vintage_models.components.positional_encoding import (
    PositionalEncoding1D,
    LearnablePositionalEncoding1D,
)


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture
def input():
    return torch.tensor(
        data=[[[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32
    )


class TestPositionalEncoding1D:
    encoding = PositionalEncoding1D(sequence_len=2, n=100, embedding_len=4)

    def test_simple(self, input):
        output = self.encoding(input)
        assert output.shape == (1, 2, 4)
        assert not torch.allclose(output, input)

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.encoding.to("cuda")
        self.encoding(input.to("cuda"))


class TestLearnablePositionalEncoding1D:
    encoding = LearnablePositionalEncoding1D(sequence_len=2, embedding_len=4)

    def test_simple(self, input):
        output = self.encoding(input)
        assert output.shape == (1, 2, 4)
        assert not torch.allclose(output, input)

    def test_wrong_sequence_length(self):
        with pytest.raises(ValueError):
            self.encoding(torch.zeros(1, 3, 4))

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.encoding.to("cuda")
        self.encoding(input.to("cuda"))
