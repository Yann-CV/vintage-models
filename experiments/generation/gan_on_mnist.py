from pathlib import Path

import torch
from lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from experiments.data.mnist import MNISTDataModule
from experiments.generation.generator import ImageAdversarialGenerator
from vintage_models.adversarial.gan.gan import Gan
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCH_COUNT = 50
MODEL = Gan(
    image_width=28,
    image_height=28,
    generator_input_size=100,
    generator_latent_size=1200,
    discriminator_hidden_size=240,
    discriminator_maxout_depth=5,
)

GENERATOR = ImageAdversarialGenerator(MODEL)

LOGGER = MLFlowLogger(
    experiment_name="GAN on MNIST",
    tracking_uri="/storage/ml/mlruns",
    run_name=str(MODEL),
    log_model=True,
)

DATAMODULE = MNISTDataModule(
    Path("/storage/ml"),
    train_batch_size=128,
    test_batch_size=128,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize((0.5,), (0.5,)),
        ]
    ),
    num_workers=11,
)

DATAMODULE.prepare_data()
DATAMODULE.setup("fit")
TRAINER = Trainer(
    accelerator=DEVICE,
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
