from pathlib import Path
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from experiments.data.mnist import MNISTDataModule
from experiments.generation.generator import ImageAutoEncoderGenerator
from vintage_models.autoencoder.vae.vae import Vae
from torchvision.transforms.v2 import Compose, ToImage, ToDtype


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCH_COUNT = 3

MODEL = Vae(
    image_width=28,
    image_height=28,
    hidden_size=500,
    latent_size=200,
)

GENERATOR = ImageAutoEncoderGenerator(MODEL)

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
    num_workers=11,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ]
    ),
)

DATAMODULE.prepare_data()
DATAMODULE.setup("fit")
TRAINER = Trainer(
    accelerator=DEVICE,
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
