import pytest
import torch

from vintage_models.components.convolution import FilteredConv2d


@pytest.fixture()
def input():
    return torch.arange(start=0, end=4 * 4, dtype=torch.float32).reshape((1, 4, 2, 2))


class TestFilteredConv2d:
    def test_simple(self, input):
        conv = FilteredConv2d(
            in_channel_indices=[[0, 1], [2, 3]],
            out_channels=1,
            kernel_size=2,
        )
        output = conv(input)
        assert output.shape == (1, 2, 1, 1)
