from enum import Enum

from torch.nn import Module
from torch import Tensor


class PaddingMode(Enum):
    """
    See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


def padding_values_to_be_multiple(real: int, multiple_of: int) -> tuple[int, int]:
    if multiple_of == real or real % multiple_of == 0:
        return 0, 0

    to_add = (
        multiple_of - real
        if multiple_of > real
        else (real // multiple_of + 1) * multiple_of - real
    )
    padding_start = to_add // 2
    return padding_start, to_add - padding_start


class MaybeToColor(Module):
    def forward(self, img: Tensor) -> Tensor:
        channel_count = img.shape[0]

        if channel_count not in (1, 3):
            raise ValueError(
                f"The image must have either 1 or 3 channels, but has {channel_count}."
            )

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img
