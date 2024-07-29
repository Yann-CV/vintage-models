from collections import OrderedDict
from dataclasses import dataclass

from torch import Tensor
from torch.nn import Module, Sequential, ReLU, Sigmoid, Dropout, Tanh

from vintage_models.components.multilayer_perceptron import LinearWithActivation, MaxOut


class GanGenerator(Module):
    """Generator for the vintage generative adversarial network.

    Attributes:
        out_width: Width of the ouput image.
        out_height: Height of the ouput image.
        latent_size: Size of the latent space.
        input_size: Size of the input space.
    """

    def __init__(
        self,
        out_width: int,
        out_height: int,
        latent_size: int,
        input_size: int,
    ) -> None:
        super().__init__()
        self.out_width = out_width
        self.out_height = out_height
        self.latent_size = latent_size
        self.input_size = input_size

        self.model = Sequential(
            OrderedDict(
                [
                    (
                        "linear_with_relu_1",
                        LinearWithActivation(
                            in_size=input_size,
                            out_size=latent_size,
                            activation_layer=ReLU(),
                            normalize=True,
                        ),
                    ),
                    (
                        "linear_with_relu_2",
                        LinearWithActivation(
                            in_size=latent_size,
                            out_size=latent_size,
                            activation_layer=ReLU(),
                            normalize=True,
                        ),
                    ),
                    (
                        "linear_with_tanh",
                        LinearWithActivation(
                            in_size=latent_size,
                            out_size=self.out_width * self.out_height,
                            activation_layer=Tanh(),
                            normalize=False,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        generated = self.model(x)
        return generated.view(-1, 1, self.out_width, self.out_height)


class GanDiscriminator(Module):
    """Discriminator for the generative adversarial network.

    Attributes:
        in_width: Width of the input image.
        in_height: Height of the input image.
        hidden_size: Size of the hidden layer.
        maxout_depth: The depth of the maxout layers.
        drop_out_proba: Probability of dropout.
    """

    def __init__(
        self,
        in_width: int,
        in_height: int,
        hidden_size: int,
        maxout_depth: int,
        drop_out_proba: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.hidden_size = hidden_size
        self.maxout_depth = maxout_depth
        self.drop_out_proba = drop_out_proba

        in_size = in_width * in_height

        self.model = Sequential(
            OrderedDict(
                [
                    (
                        "dropout_1",
                        Dropout(drop_out_proba),
                    ),
                    (
                        "maxout_1",
                        MaxOut(in_size, hidden_size, maxout_depth),
                    ),
                    (
                        "dropout_2",
                        Dropout(drop_out_proba),
                    ),
                    (
                        "maxout_2",
                        MaxOut(hidden_size, hidden_size, maxout_depth),
                    ),
                    (
                        "linear_with_sigmoid",
                        LinearWithActivation(
                            hidden_size, 1, Sigmoid(), normalize=False
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.view(x.size(0), -1))


@dataclass
class GanLosses:
    """Losses for the generative adversarial network.

    Attributes:
        generator_loss: Loss for the generator.
        discriminator_loss: Loss for the discriminator.
    """

    generator_loss: Tensor
    discriminator_loss: Tensor

    @property
    def total_loss(self) -> Tensor:
        return self.generator_loss + self.discriminator_loss


class Gan(Module):
    """Vintage implementation of the generative adversarial network.

    See the paper_review.md file for more information.
    The forward method is about applying the discriminator on a batch of images.

    Attributes:
        generator: Generator for the generative adversarial network. Transform the noise in an image.
        discriminator: Discriminator for the generative adversarial network. Decide if an image
        is real or fake.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        generator_input_size: int,
        generator_latent_size: int,
        discriminator_hidden_size: int,
        discriminator_maxout_depth: int,
        discriminator_drop_out_proba: float = 0.2,
    ) -> None:
        """Initializes the GAN.

        Args:
            image_width: Width of the input and ouput images.
            image_height: Height of the input and ouput images.
            generator_input_size: Size of the input space for the generator.
            generator_latent_size: Size of the latent space for the generator.
            discriminator_hidden_size: Size of the hidden layer for the discriminator.
            discriminator_maxout_depth: The depth of the maxout layers for the discriminator.
            discriminator_drop_out_proba: Probability of dropout for the discriminator.
        """
        super().__init__()

        self.generator = GanGenerator(
            out_width=image_width,
            out_height=image_height,
            input_size=generator_input_size,
            latent_size=generator_latent_size,
        )

        self.discriminator = GanDiscriminator(
            in_width=image_width,
            in_height=image_height,
            hidden_size=discriminator_hidden_size,
            maxout_depth=discriminator_maxout_depth,
            drop_out_proba=discriminator_drop_out_proba,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.discriminator(x)

    def __str__(self) -> str:
        return (
            f"GAN_image_width_{self.generator.out_width}_image_height_{self.generator.out_height}_"
            f"generator_input_size_{self.generator.input_size}_"
            f"generator_latent_size_{self.generator.latent_size}_"
            f"discriminator_hidden_size_{self.discriminator.hidden_size}_"
            f"discriminator_maxout_depth_{self.discriminator.maxout_depth}_"
            f"discriminator_drop_out_proba_{self.discriminator.drop_out_proba}"
        )
