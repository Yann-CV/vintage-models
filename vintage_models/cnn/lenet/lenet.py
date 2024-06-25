from torch.nn import Module, Conv2d, Sigmoid, Sequential, Linear, Softmax
from torch import Tensor

from vintage_models.components.activation import ScaledTanh
from vintage_models.components.convolution import FilteredConv2d
from vintage_models.components.image import ImageTargeter
from vintage_models.components.pooling import SumAddPool2D
from vintage_models.components.rbf import EuclideanDistanceRBF
from vintage_models.utility.transform import PaddingMode


class LeNet5(Module):
    """Vintage implementation of the LeNet-5 model.

    See the paper_review.md file for more information.

    If the input image is not 32x32 then it will either be padded or resized to 32x32.

    The class vector is computed from the softmax applied on the RBF layer.

    Attributes:
        class_count: Number of classes in the output.
        image_targeter: ImageTargeter used to prepare the input image.
        model_from_image: Model that takes an image and outputs a feature vector.
        model_from_feature: Model that takes a feature vector and outputs a class vector.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        class_count: int,
        padding_mode: PaddingMode = PaddingMode.CONSTANT,
    ) -> None:
        """
        Args:
            image_width: Width of the input image.
            image_height: Height of the input image.
            class_count: Number of classes in the output.
            padding_mode: Padding mode for the image targeter.
        """
        super().__init__()
        self.class_count = class_count

        self.image_targeter = ImageTargeter(
            in_width=image_width,
            in_height=image_height,
            out_width=32,
            out_height=32,
            padding_mode=padding_mode,
            color=False,
        )
        conv_1 = Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        add_pool_1 = SumAddPool2D(
            kernel_size=2,
            in_channels=6,
            activation=Sigmoid(),
        )
        conv_2 = FilteredConv2d(
            in_channel_indices=[
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [0, 4, 5],
                [5, 0, 1],
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 0],
                [0, 1, 4, 5],
                [0, 1, 2, 5],
                [0, 1, 3, 4],
                [1, 2, 4, 5],
                [0, 2, 3, 5],
                [0, 1, 2, 3, 4, 5],
            ],
            out_channels=1,
            kernel_size=5,
        )
        add_pool_2 = SumAddPool2D(
            kernel_size=2,
            in_channels=16,
            activation=Sigmoid(),
        )
        conv_3 = Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
        )
        self.model_from_image = Sequential(
            conv_1,
            add_pool_1,
            conv_2,
            add_pool_2,
            conv_3,
        )
        self.model_from_feature = Sequential(
            Linear(
                in_features=120,
                out_features=84,
            ),
            ScaledTanh(1.7159),
            EuclideanDistanceRBF(84, class_count),
            Softmax(dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        image = self.image_targeter(x)
        features = self.model_from_image(image).squeeze()
        return self.model_from_feature(features)

    def __str__(self):
        return (
            f"lenet5-classcount{self.class_count}-"
            f"paddingmode{self.image_targeter.padding_mode.value}-"
            f"width_in{self.image_targeter.in_width}-"
            f"height_in{self.image_targeter.in_height}"
        )
