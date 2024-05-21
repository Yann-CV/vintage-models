from pathlib import Path

import torch.cuda
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from experiments.data.mnist import MNISTDataModule
from experiments.generation.generator import ImageAdversarialGenerator
from vintage_models.adversarial.gan.gan import Gan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCH_COUNT = 100
MODEL = Gan(
    image_width=28,
    image_height=28,
    hidden_size=500,
    latent_size=200,
    maxout_depth=3,
    device=DEVICE,
)

GENERATOR = ImageAdversarialGenerator(MODEL)

LOGGER = MLFlowLogger(
    experiment_name="GAN on MNIST",
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
    num_workers=11,
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
