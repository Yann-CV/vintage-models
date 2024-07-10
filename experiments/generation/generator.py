from pathlib import Path

import torch
from lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor
from torch.nn import Linear
from torch.optim import Adam

from vintage_models.adversarial.gan.gan import Gan, GanLosses
from vintage_models.autoencoder.vae.vae import Vae


class ImageAutoEncoderGenerator(LightningModule):
    """Lightning module for image generation experiments."""

    def __init__(self, model: Vae) -> None:
        super().__init__()
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        # log generated images
        self.generation_counter = 0

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
        loss = self.model.loss(data)
        self.log(
            "training_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
        loss = self.model.loss(data)
        self.log(
            "validation_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
        generated = self.model.generate(1)

        assert isinstance(self.logger, MLFlowLogger)
        saving_path = Path("/storage/ml") / str(self.logger.experiment_id) / "generated"
        if not saving_path.exists():
            saving_path.mkdir(parents=True, exist_ok=True)

        self.logger.experiment.log_image(
            self.logger.run_id,
            generated.squeeze().cpu().numpy(),
            f"generated_{self.generation_counter}.png",
        )

        self.generation_counter += 1
        return generated

    def configure_optimizers(self):
        return self.optimizer


class ImageAdversarialGenerator(LightningModule):
    """Lightning module for image generation experiments."""

    def __init__(
        self,
        model: Gan,
    ) -> None:
        super().__init__()
        self.model = model

        self.training_step_outputs: list[Tensor] = []
        self.validation_step_outputs: list[Tensor] = []
        self.test_step_outputs: list[Tensor] = []
        b1 = 0.5
        b2 = 0.999
        lr = 0.0002
        self.discriminator_optimiser = Adam(
            self.model.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        self.generator_optimiser = Adam(
            self.model.generator.parameters(), lr=lr, betas=(b1, b2)
        )

    def configure_optimizers(self):
        # with multiple optimizers, we should set self.automatic_optimization = False
        # However this is somehow deactivating the model checkpointing
        dummy_optimizer = Adam(Linear(1, 1).parameters())
        return dummy_optimizer

    def training_step(self, batch: tuple[Tensor, Tensor]) -> torch.Tensor:
        X, _ = batch
        fake = self.model.generate(X.size(0))

        self.generator_optimiser.zero_grad()
        generator_loss = self.model.generator_loss(fake)
        generator_loss.backward()
        self.generator_optimiser.step()

        self.discriminator_optimiser.zero_grad()
        discriminator_loss = self.model.discriminator_loss(X, fake)
        discriminator_loss.backward()
        self.discriminator_optimiser.step()

        self.training_step_outputs.append(
            GanLosses(
                generator_loss=generator_loss, discriminator_loss=discriminator_loss
            )
        )

        return torch.tensor(0.0, requires_grad=True)

    def on_train_epoch_end(self) -> None:
        generator_loss = self.training_step_outputs[-1].generator_loss.item()
        discriminator_loss = self.training_step_outputs[-1].discriminator_loss.item()

        loss = generator_loss + discriminator_loss
        self.log(
            "training_generator_loss",
            generator_loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "training_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "training_discriminator_loss",
            discriminator_loss,
            prog_bar=True,
            logger=True,
        )

        self.training_step_outputs.clear()

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch

        with torch.no_grad():
            fake = self.model.generate(data.size(0))
            generator_loss = self.model.generator_loss(fake)
            discriminator_loss = self.model.discriminator_loss(data, fake)

        self.validation_step_outputs.append(
            GanLosses(
                generator_loss=generator_loss, discriminator_loss=discriminator_loss
            )
        )
        return generator_loss + discriminator_loss

    def on_validation_epoch_end(self) -> None:
        generator_loss = self.validation_step_outputs[-1].generator_loss.item()
        discriminator_loss = self.validation_step_outputs[-1].discriminator_loss.item()
        self.log(
            "val_loss",
            generator_loss + discriminator_loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_discriminator_loss",
            discriminator_loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_generator_loss",
            generator_loss,
            prog_bar=True,
            logger=True,
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        self.model.eval()
        data, _ = batch
        with torch.no_grad():
            generated = self.model.generate(1)
        self.test_step_outputs.append(generated)
        return generated

    def on_test_epoch_end(self) -> None:
        assert isinstance(self.logger, MLFlowLogger)
        saving_path = Path("/storage/ml") / str(self.logger.experiment_id) / "generated"
        if not saving_path.exists():
            saving_path.mkdir(parents=True, exist_ok=True)

        for idx, image in enumerate(self.test_step_outputs):
            # warning: only tested with mlflow logger
            self.logger.experiment.log_image(
                self.logger.run_id,
                image.squeeze().cpu().numpy(),
                f"generated_{idx}.png",
            )

        self.test_step_outputs.clear()
