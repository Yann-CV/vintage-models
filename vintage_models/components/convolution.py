from torch import Tensor, cat
from torch.nn import Conv2d, ModuleList, Module
from torch.nn.common_types import _size_2_t


class FilteredConv2d(Module):
    """Apply conv2d only to some specified channels.

    This module is useful when you want to apply different conv2d to different channels.
    For example, you have a tensor of shape (batch, channel, width, height), and you want to
    apply conv2d to some channels and not to others. You can do this by using this module.

    Args:
        in_channel_indices: A list of list of integers. Each list of integers represents the
        indices of channels that will be used as input for the corresponding conv2d.
        out_channels: The out channels for every conv2d built from in_channel_indices.
        Others: See `torch.nn.Conv2d`.
    """

    def __init__(
        self,
        in_channel_indices: list[list[int]],
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.conv2ds = ModuleList(
            [
                Conv2d(
                    len(channel_list),
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                    padding_mode,
                    device,
                    dtype,
                )
                for channel_list in in_channel_indices
            ]
        )
        self.in_channel_indices = in_channel_indices

    def forward(self, x: Tensor) -> Tensor:
        conv_result_list = []
        for conv2d, in_channels in zip(
            self.conv2ds, self.in_channel_indices, strict=True
        ):
            conv_result_list.append(conv2d(x[:, in_channels]))

        return cat(conv_result_list, dim=-3)
