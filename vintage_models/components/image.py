from torch import Tensor
from torch.nn import Module
from torchvision.transforms.v2 import Resize, Pad, Compose, Grayscale

from vintage_models.utility.transform import PaddingMode, padding_values_to_be_multiple


class ImageTargeter(Module):
    """Prepare an image for a model (obtain the right size).

    If the input image is less than the target size, it will be padded.
    Otherwise, If the input image is greater than the target size, it will be resized.
    Finally, the image will be converted to grayscale or color if needed.

    Attributes:
        in_width: Width of the input image.
        in_height: Height of the input image.
        out_width: Width of the output image.
        out_height: Height of the output image.
        padding_mode: Padding mode if padding is required.
        color: Whether the output image should be color or grayscale.
    """

    def __init__(
        self,
        in_width: int,
        in_height: int,
        out_width: int,
        out_height: int,
        padding_mode: PaddingMode = PaddingMode.CONSTANT,
        color: bool = True,
    ) -> None:
        super().__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height
        self.padding_mode = padding_mode
        self.color = color

        transforms = (
            (
                [Resize(size=(self.out_height, self.out_width))]
                if self.in_width > self.out_width or self.in_height > self.out_height
                else []
            )
            + (
                [
                    Pad(
                        padding=padding_values_to_be_multiple(
                            self.in_width, self.out_width
                        )
                        + padding_values_to_be_multiple(
                            self.in_height, self.out_height
                        ),
                        padding_mode=self.padding_mode.value,
                    )
                ]
                if self.in_width < self.out_width or self.in_height < self.out_height
                else []
            )
            + ([MaybeToColor()] if color else [Grayscale(1)])
        )
        self.transforms: None | Compose = (
            None if not transforms else Compose(transforms)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_width or x.shape[-2] != self.in_height:
            raise ValueError(
                f"Input image shape {x.shape} does not match expected shape "
                f"({self.in_width}, {self.in_height})"
            )

        if self.transforms is not None:
            x = self.transforms(x)
        return x


class MaybeToColor(Module):
    """Convert a grayscale image to color if needed."""

    def forward(self, img: Tensor) -> Tensor:
        channel_count = img.shape[-3]

        if channel_count not in (1, 3):
            raise ValueError(
                f"The image must have either 1 or 3 channels, but has {channel_count}."
            )

        if img.shape[-3] == 1:
            img = img.repeat(1, 3, 1, 1) if len(img.shape) == 4 else img.repeat(3, 1, 1)

        return img
