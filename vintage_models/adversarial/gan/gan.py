from collections import OrderedDict
from dataclasses import dataclass
from functools import partial

from torch import Tensor, device as torch_device, randn, reshape, ones_like, zeros_like
from torch.nn import Module, Sequential, ReLU, Sigmoid, BCELoss, BatchNorm1d

from vintage_models.components.multilayer_perceptron import LinearWithActivation, MaxOut


class GanGenerator(Module):
    """Generator for the vintage generative adversarial network.

    Attributes:
        out_width: Width of the ouput image.
        out_height: Height of the ouput image.
        hidden_size: Size of the hidden layer (after application of the linear and activation layer).
        latent_size: Size of the latent space.
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
                        ),
                    ),
                    (
                        "linear_with_relu_2",
                        LinearWithActivation(
                            in_size=latent_size,
                            out_size=latent_size,
                            activation_layer=ReLU(),
                        ),
                    ),
                    (
                        "linear_with_sigmoid",
                        LinearWithActivation(
                            latent_size, self.out_width * self.out_height, Sigmoid()
                        ),
                    ),
                ]
            )
        )

        self.to_image = partial(reshape, shape=(-1, 1, self.out_width, self.out_height))

    def forward(self, x: Tensor) -> Tensor:
        generated = self.model(x)

        return generated.reshape(-1, 1, self.out_width, self.out_height)


class GanDiscriminator(Module):
    """Discriminator for the vintage variational autoencoder.

    Attributes:
        in_width: Width of the input image.
        in_height: Height of the input image.
        hidden_size: Size of the hidden layer.
        maxout_depth: The depth of the maxout layers.
    """

    def __init__(
        self,
        in_width: int,
        in_height: int,
        hidden_size: int,
        maxout_depth: int,
        device: str | torch_device | int = "cpu",
    ) -> None:
        super().__init__()

        self.in_width = in_width
        self.in_height = in_height
        self.hidden_size = hidden_size
        self.maxout_depth = maxout_depth

        in_size = in_width * in_height
        self.to_vector = partial(reshape, shape=(-1, in_size))

        self.model = Sequential(
            OrderedDict(
                [
                    (
                        "maxout_hidden_1",
                        MaxOut(in_size, hidden_size, maxout_depth, device),
                    ),
                    ("batch_norm_1", BatchNorm1d(hidden_size, 0.8)),
                    (
                        "maxout_hidden_2",
                        MaxOut(hidden_size, hidden_size, maxout_depth, device),
                    ),
                    ("batch_norm_2", BatchNorm1d(hidden_size, 0.8)),
                    (
                        "linear_with_sigmoid",
                        LinearWithActivation(hidden_size, 1, Sigmoid()),
                    ),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_vector(x)
        return self.model(x)


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

    Attributes:
        generator: Generator for the generative adversarial network. Tranform the noise in an image.
        discriminator: Discriminator for the generative adversarial network. Decide if the image
        is real or fake.
        device: Device to use for the model running.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        generator_input_size: int,
        generator_latent_size: int,
        discriminator_hidden_size: int,
        discriminator_maxout_depth: int,
        device: str | torch_device | int = "cpu",
    ) -> None:
        """Initializes the Vae.

        Args:
            image_width: Width of the input and ouput images.
            image_height: Height of the input and ouput images.
            hidden_size: Size of the hidden layer for both the generator and discriminator.
            latent_size: Size of the latent layer of the generator. The size of noise used to generate
            a new image.
            maxout_depth: the depth of the maxout layers in the discriminator.
            device: Device to use for the model running.
        """
        super().__init__()
        self.device = torch_device(device)

        self.generator = GanGenerator(
            out_width=image_width,
            out_height=image_height,
            input_size=generator_input_size,
            latent_size=generator_latent_size,
        ).to(self.device)

        self.discriminator = GanDiscriminator(
            in_width=image_width,
            in_height=image_height,
            hidden_size=discriminator_hidden_size,
            maxout_depth=discriminator_maxout_depth,
            device=device,
        ).to(self.device)

        self.bce_loss = BCELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.discriminator(x)

    def generate(self, n: int) -> Tensor:
        size = (n, self.generator.input_size)
        return self.generator(randn(size=size, device=self.device))

    def generator_loss(self, x: Tensor) -> Tensor:
        """compute the loss from generated data.

        x has been created by calling the generate method.
        """
        discriminated = self.discriminator(x)
        label = ones_like(discriminated, requires_grad=False)
        return self.bce_loss(discriminated, label)

    def discriminator_loss(self, x: Tensor, fake: Tensor) -> Tensor:
        real_discriminated = self.discriminator(x)
        real_label = ones_like(real_discriminated, requires_grad=False)
        real_loss = self.bce_loss(real_discriminated, real_label)

        fake_discriminated = self.discriminator(fake.detach())
        fake_label = zeros_like(fake_discriminated, requires_grad=False)
        fake_loss = self.bce_loss(fake_discriminated, fake_label)

        return (real_loss + fake_loss) / 2

    def __str__(self) -> str:
        return (
            f"GAN_image_width_{self.generator.out_width}_image_height_{self.generator.out_height}"
            f"_generator_input_size_{self.generator.input_size}_"
            f"generator_latent_size_{self.generator.latent_size}_"
            f"discriminator_hidden_size_{self.discriminator.hidden_size}_"
            f"discriminator_maxout_depth_{self.discriminator.maxout_depth}"
        )
