from functools import partial

from torch import Tensor
from torch.nn import Module, Unfold
from torch.nn.functional import pad


from vintage_models.utility.transform import PaddingMode, padding_values_to_be_multiple


class PatchConverter(Module):
    """Convert an image to unfolded patches.

    Attributes:
        patch_size: The size of the patches.
        image_width: The width of the image.
        image_height: The height of the image.
        padding_mode: Padding mode if padding is required.
        padding: Padding values to be applied.
        pad: function to pad the image if needed.
        unfold: function to unfold the image area as patch.
    """

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
        self.padding = padding_values_to_be_multiple(
            self.image_width, self.patch_size
        ) + padding_values_to_be_multiple(self.image_height, self.patch_size)
        self.pad = partial(
            pad,
            pad=self.padding,
            mode=self.padding_mode.value,
            value=0,
        )
        self.unfold = Unfold(
            kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )

    @property
    def final_width(self) -> int:
        return self.image_width + self.padding[0] + self.padding[1]

    @property
    def final_height(self) -> int:
        return self.image_width + self.padding[0] + self.padding[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(input=x)
        return self.unfold(x).transpose(-2, -1)
