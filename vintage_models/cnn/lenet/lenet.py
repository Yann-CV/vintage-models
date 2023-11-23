from torch.nn import Module
from torch import Tensor

from vintage_models.components.image import ImageTargeter
from vintage_models.utility.transform import PaddingMode


class LeNet5(Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        class_count: int,
        padding_mode: PaddingMode = PaddingMode.CONSTANT,
    ) -> None:
        super().__init__()
        self.image_targeter = ImageTargeter(
            width_in=image_width,
            height_in=image_height,
            width_out=32,
            height_out=32,
            padding_mode=padding_mode,
            color=False,
        )
        self.class_count = class_count

    def forward(self, x: Tensor) -> Tensor:
        image = self.image_targeter(x)
        return image

    def __str__(self):
        return f"lenet5" f"-classcount{self.class_count}-"
