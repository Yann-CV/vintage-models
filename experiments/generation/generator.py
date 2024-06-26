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
