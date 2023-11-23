from torch import Tensor
from torch.nn import Module
from torchvision.transforms.v2 import Resize, Pad, Compose, Grayscale

from vintage_models.utility.transform import PaddingMode, padding_values_to_be_multiple


class ImageTargeter(Module):
    def __init__(
        self,
        width_in: int,
        height_in: int,
        width_out: int,
        height_out: int,
        padding_mode: PaddingMode = PaddingMode.CONSTANT,
        color: bool = True,
    ) -> None:
        super().__init__()

        self._width_in = width_in
        self._height_in = height_in
        self._width_out = width_out
        self._height_out = height_out
        self._padding_mode = padding_mode

        transforms = (
            (
                [Resize(size=(self.final_width, self.final_height))]
                if self._width_in > self.final_height
                or self._height_in > self.final_height
                else []
            )
            + (
                [
                    Pad(
                        padding=padding_values_to_be_multiple(
                            self._width_in, self._width_out
                        )
                        + padding_values_to_be_multiple(
                            self._height_in, self._height_out
                        ),
                        padding_mode=self._padding_mode.value,
                    )
                ]
                if self._width_in < self.final_height
                or self._height_in < self.final_height
                else []
            )
            + ([MaybeToColor()] if color else [Grayscale(1)])
        )
        self.transforms: None | Compose = (
            None if not transforms else Compose(transforms)
        )

    @property
    def final_width(self) -> int:
        return self._width_out

    @property
    def final_height(self) -> int:
        return self._height_out

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self._width_in or x.shape[-2] != self._height_in:
            raise ValueError(
                f"Input image shape {x.shape} does not match expected shape "
                f"({self._width_in}, {self._height_in})"
            )

        if self.transforms is not None:
            x = self.transforms(x)
        return x


class MaybeToColor(Module):
    def forward(self, img: Tensor) -> Tensor:
        channel_count = img.shape[-3]

        if channel_count not in (1, 3):
            raise ValueError(
                f"The image must have either 1 or 3 channels, but has {channel_count}."
            )

        if img.shape[-3] == 1:
            img = img.repeat(1, 3, 1, 1) if len(img.shape) == 4 else img.repeat(3, 1, 1)

        return img
