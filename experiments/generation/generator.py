from pathlib import Path

from lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger
from torch import Tensor
from torch.optim import Adam

from vintage_models.autoencoder.vae.vae import Vae


class ImageGenerator(LightningModule):
    """Lightning module for image generation experiments."""

    def __init__(self, model: Vae) -> None:
        super().__init__()
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.training_step_outputs: list[Tensor] = []
        self.validation_step_outputs: list[Tensor] = []
        self.test_step_outputs: list[Tensor] = []

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
        loss = self.model.loss(data)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "training_loss",
            self.training_step_outputs[-1].item(),
            prog_bar=True,
            logger=True,
        )
        self.training_step_outputs.clear()

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
        loss = self.model.loss(data)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "validation_loss",
            self.validation_step_outputs[-1].item(),
            prog_bar=True,
            logger=True,
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, _ = batch
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

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return self.optimizer
