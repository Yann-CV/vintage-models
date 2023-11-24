from enum import Enum


class PaddingMode(Enum):
    """Possible padding modes.

    See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


def padding_values_to_be_multiple(real: int, multiple_of: int) -> tuple[int, int]:
    """Compute the padding values to add in order to make real a multiple of multiple_of."""
    if multiple_of == real or real % multiple_of == 0:
        return 0, 0

    to_add = (
        multiple_of - real
        if multiple_of > real
        else (real // multiple_of + 1) * multiple_of - real
    )
    padding_start = to_add // 2
    return padding_start, to_add - padding_start
