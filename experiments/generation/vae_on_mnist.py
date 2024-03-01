from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from experiments.data.mnist import MNISTDataModule
from experiments.generation.generator import ImageGenerator
from vintage_models.autoencoder.vae.vae import Vae


EPOCH_COUNT = 100
MODEL = Vae(
    image_width=28,
    image_height=28,
    hidden_size=28,
    latent_size=2,
)

GENERATOR = ImageGenerator(MODEL)

LOGGER = MLFlowLogger(
    experiment_name="VAE on MNIST",
    tracking_uri="/storage/ml/mlruns",
    run_name=str(MODEL),
    log_model=True,
)

CHECKPOINT_CALLBACK = ModelCheckpoint(
    save_top_k=1,
    monitor="training_loss",
    mode="min",
    dirpath="/storage/ml/models",
    filename="vae-mnist-{epoch:02d}-{accuracy:.2f}",
)

DATAMODULE = MNISTDataModule(
    Path("/storage/ml"),
    train_batch_size=500,
    color=False,
    between_0_and_1=True,
)

DATAMODULE.prepare_data()
DATAMODULE.setup("fit")
TRAINER = Trainer(
    accelerator="cuda",
    callbacks=[CHECKPOINT_CALLBACK],
    logger=LOGGER,
    max_epochs=EPOCH_COUNT,
)
TRAINER.fit(
    model=GENERATOR,
    train_dataloaders=DATAMODULE.train_dataloader(),
    val_dataloaders=DATAMODULE.val_dataloader(),
)
DATAMODULE.setup("test")
TRAINER.test(dataloaders=DATAMODULE.test_dataloader())
