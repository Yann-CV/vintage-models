import pytest
import torch

from vintage_models.components.attention import (
    ScaledDotProductAttention,
    HeadAttention,
    MultiHeadAttention,
)


@pytest.fixture
def queries():
    return torch.tensor([[0, 0.5, 0], [0, 0, 0]], dtype=torch.float32)


@pytest.fixture
def keys():
    return torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float32)


@pytest.fixture
def values():
    return torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.float32)


class TestScaledDotProductAttention:
    attention = ScaledDotProductAttention(dk=3)

    def test_simple(self, keys, values, queries):
        # dot product: [[0.5, 0.0], [0.0, 0.0]]
        # scaled: [[0.16, 0.0], [0.0, 0.0]]
        # softmax: [[1.0, 0.0], [0.5, 0.5]]
        out1 = torch.softmax(torch.tensor([0.5 / 3, 0.0]), dim=-1)[0].item()
        out2 = 0.5
        assert torch.allclose(
            self.attention(keys, values, queries),
            torch.tensor([[out1, out1, out1, out1], [out2, out2, out2, out2]]),
        )

    def test_wrong_key_size(self, queries, values):
        with pytest.raises(ValueError):
            self.attention(torch.zeros(1, 2), values, queries)

    def test_wrong_key_count(self, queries, values):
        with pytest.raises(ValueError):
            self.attention(torch.zeros(1, 3), values, queries)

    def test_wrong_query_size(self, keys, values):
        with pytest.raises(ValueError):
            self.attention(keys, values, torch.zeros(1, 2))

    def test_wrong_value_count(self, queries, keys):
        with pytest.raises(ValueError):
            self.attention(keys, torch.zeros(1, 4), queries)


class TestHeadAttention:
    head_attention = HeadAttention(dk=3, dv=4)

    def test_simple(self, keys, values, queries):
        output = self.head_attention(keys, values, queries)
        assert output.shape == (2, 4)

    def test_wrong_key_size(self, queries, values):
        with pytest.raises(RuntimeError):
            self.head_attention(torch.zeros(1, 2), values, queries)

    def test_wrong_query_size(self, keys, values):
        with pytest.raises(RuntimeError):
            self.head_attention(keys, values, torch.zeros(1, 2))

    def test_wrong_value_size(self, queries, keys):
        with pytest.raises(RuntimeError):
            self.head_attention(keys, torch.zeros(2, 3), queries)


class TestMultiHeadAttention:
    multi_head_attention = MultiHeadAttention(dk=3, dv=4, h=2)

    def test_simple(self, keys, values, queries):
        output = self.multi_head_attention(keys, values, queries)
        assert output.shape == (2, 4)
