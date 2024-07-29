from abc import abstractmethod, ABC
from pathlib import Path

import numpy as np
import torch
from lightning import LightningModule

from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor, ones_like, randn, zeros_like
from torch.nn import Linear, BCELoss, Module
from torch.optim import Adam, SGD
from torchvision.utils import make_grid

from vintage_models.adversarial.gan.gan import Gan, GanLosses
from vintage_models.autoencoder.vae.vae import Vae


class ImageGenerator(LightningModule, ABC):
    """Lightning module for image generation experiments.

    It specifies the structure of the training and testing steps for image generation.
    It is also responsible for logging the generated images in order to follow up
    the training evolution.
    """

    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

        # log generated images
        self.grid_side = 5
        self.generation_counter = 0
        self.epoch_counter = 0

    @abstractmethod
    def generate_data(self, batch_size: int) -> Tensor:
        ...

    @abstractmethod
    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        ...

    def log_image(self):
        with torch.no_grad():
            generated = self.generate_data(self.grid_side**2)

        grid = make_grid(
            generated,
            nrow=self.grid_side,
            normalize=True,
        )

        grid = np.moveaxis(grid.cpu().numpy(), 0, 2)  # from (C, H, W) to (H, W, C)

        assert isinstance(self.logger, MLFlowLogger)
        saving_path = (
            Path("/storage/ml") / str(self.logger.experiment_id) / "training_evolution"
        )
        if not saving_path.exists():
            saving_path.mkdir(parents=True, exist_ok=True)

        self.logger.experiment.log_image(
            self.logger.run_id,
            grid,
            f"training_evolution/generated_{self.epoch_counter}.png",
        )

        self.epoch_counter += 1

    def test_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch

        with torch.no_grad():
            generated = self.generate_data(self.grid_side**2)

        grid = make_grid(
            generated,
            nrow=self.grid_side,
            normalize=True,
        )

        grid = np.moveaxis(grid.cpu().numpy(), 0, 2)  # from (C, H, W) to (H, W, C)

        assert isinstance(self.logger, MLFlowLogger)
        saving_path = Path("/storage/ml") / str(self.logger.experiment_id) / "generated"
        if not saving_path.exists():
            saving_path.mkdir(parents=True, exist_ok=True)

        self.logger.experiment.log_image(
            self.logger.run_id,
            grid,
            f"generated_{self.generation_counter}.png",
        )

        self.generation_counter += 1

        return generated

    @abstractmethod
    def configure_optimizers(self):
        ...


class ImageAutoEncoderGenerator(ImageGenerator):
    """Lightning module for autoencoder image generation experiments."""

    def __init__(self, model: Vae) -> None:
        super().__init__(model)

        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.bce_loss = BCELoss(reduction="none")
        self.vector_size = (
            self.model.encoder.image_width * self.model.encoder.image_height
        )

    def generate_data(self, batch_size: int) -> Tensor:
        size = (batch_size, self.model.encoder.latent_size)
        return self.model.decoder(randn(size=size, device=self.device))

    def loss(self, x: Tensor) -> Tensor:
        reconstructed = self.model.forward(x)

        reconstruction_loss = (
            self.bce_loss(reconstructed, x).reshape(-1, self.vector_size).sum(dim=1)
        )

        mean, log_var = self.model.encoder.compute_mean_and_log_var(x)
        kl_div = -0.5 * (1 + log_var - log_var.exp() - mean.pow(2)).sum(dim=1)

        return kl_div.mean() + reconstruction_loss.mean()

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
        loss = self.loss(data)
        self.log(
            "training_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
        with torch.no_grad():
            loss = self.loss(data)
        self.log(
            "validation_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return self.optimizer


class ImageAdversarialGenerator(ImageGenerator):
    """Lightning module for image generation experiments."""

    def __init__(self, model: Gan) -> None:
        super().__init__(model)

        self.discriminator_optimiser = SGD(
            self.model.discriminator.parameters(), lr=0.01
        )
        self.generator_optimiser = SGD(self.model.generator.parameters(), lr=0.01)
        self.bce_loss = BCELoss()

        self.last_train_loss: GanLosses

    def configure_optimizers(self):
        # with multiple optimizers, we should set self.automatic_optimization = False
        # However this is somehow deactivating the model checkpointing
        dummy_optimizer = SGD(Linear(1, 1).parameters(), lr=0.1)
        return dummy_optimizer

    def generate_data(self, batch_size: int) -> Tensor:
        size = (batch_size, self.model.generator.input_size)
        return self.model.generator(randn(size=size, device=self.device))

    def generator_loss(self, X: Tensor) -> Tensor:
        fake = self.generate_data(X.size(0))
        discriminated = self.model.discriminator(fake)
        return self.bce_loss(discriminated, ones_like(discriminated))

    def discriminator_loss(self, X: Tensor) -> Tensor:
        real_discriminated = self.model.discriminator(X)
        real_label = ones_like(real_discriminated)
        real_loss = self.bce_loss(real_discriminated, real_label)

        fake = self.generate_data(X.size(0))
        fake_discriminated = self.model.discriminator(fake.detach())
        fake_label = zeros_like(fake_discriminated)
        fake_loss = self.bce_loss(fake_discriminated, fake_label)

        return real_loss + fake_loss

    def training_step(self, batch: tuple[Tensor, Tensor]) -> torch.Tensor:
        X, _ = batch

        self.discriminator_optimiser.zero_grad()
        discriminator_loss = self.discriminator_loss(X)
        discriminator_loss.backward()
        self.discriminator_optimiser.step()

        self.generator_optimiser.zero_grad()
        generator_loss = self.generator_loss(X)
        generator_loss.backward()
        self.generator_optimiser.step()

        self.last_train_loss = GanLosses(
            generator_loss=generator_loss, discriminator_loss=discriminator_loss
        )

        return torch.tensor(0.0, requires_grad=True)

    def on_train_epoch_end(self) -> None:
        self.log(
            "training_generator_loss",
            self.last_train_loss.generator_loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "training_discriminator_loss",
            self.last_train_loss.discriminator_loss,
            prog_bar=True,
            logger=True,
        )
        self.log_image()
