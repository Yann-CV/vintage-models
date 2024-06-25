import pytest
import torch

from vintage_models.components.convolution import FilteredConv2d


GPU_NOT_AVAILABLE = not torch.cuda.is_available()


@pytest.fixture()
def input():
    return torch.arange(start=0, end=4 * 4, dtype=torch.float32).reshape((1, 4, 2, 2))


class TestFilteredConv2d:
    filtered_conv = FilteredConv2d(
        in_channel_indices=[[0, 1], [2, 3]],
        out_channels=1,
        kernel_size=2,
    )

    def test_simple(self, input):
        output = self.filtered_conv(input)
        assert output.shape == (1, 2, 1, 1)

    @pytest.mark.skipif(GPU_NOT_AVAILABLE, reason="No gpu available")
    def test_gpu_usage(self, input):
        self.filtered_conv.to("cuda")
        self.filtered_conv(input.to("cuda"))
