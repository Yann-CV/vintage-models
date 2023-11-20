from functools import partial

from torch import Tensor
from torch.nn import Module, Unfold
from torch.nn.functional import pad


from vintage_models.utility.image import PaddingMode, padding_values_to_be_multiple


class PatchConverter(Module):
    def __init__(
        self,
        patch_size: int,
        image_width: int,
        image_height: int,
        padding_mode: PaddingMode = PaddingMode.CONSTANT,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.image_width = image_width
        self.image_height = image_height
        self.padding_mode = padding_mode
        self.pad = partial(
            pad,
            pad=(
                padding_values_to_be_multiple(self.image_width, self.patch_size)
                + padding_values_to_be_multiple(self.image_height, self.patch_size)
            ),
            mode=self.padding_mode.value,
            value=0,
        )
        self.unfold = Unfold(
            kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(input=x)
        return self.unfold(x).transpose(-2, -1)
