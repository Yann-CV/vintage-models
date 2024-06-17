from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from experiments.classification.classifier import ImageClassifier
from experiments.data.mnist import MNISTDataModule
from vintage_models.cnn.lenet.lenet import LeNet5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCH_COUNT = 100
MODEL = LeNet5(
    image_width=28,
    image_height=28,
    class_count=10,
)

CLASSIFIER = ImageClassifier(MODEL)

LOGGER = MLFlowLogger(
    experiment_name="LeNet5 on MNIST",
    tracking_uri="/storage/ml/mlruns",
    run_name=str(MODEL),
    log_model=True,
)

CHECKPOINT_CALLBACK = ModelCheckpoint(
    save_top_k=1,
    monitor="training_loss",
    mode="min",
    dirpath="/storage/ml/models",
    filename="lenet5-mnist-{epoch:02d}-{accuracy:.2f}",
)

DATAMODULE = MNISTDataModule(Path("/storage/ml"), train_batch_size=500)

DATAMODULE.prepare_data()
DATAMODULE.setup("fit")
TRAINER = Trainer(
    accelerator=DEVICE,
    callbacks=[CHECKPOINT_CALLBACK],
    logger=LOGGER,
    max_epochs=EPOCH_COUNT,
)
TRAINER.fit(
    model=CLASSIFIER,
    train_dataloaders=DATAMODULE.train_dataloader(),
    val_dataloaders=DATAMODULE.val_dataloader(),
)
DATAMODULE.setup("test")
TRAINER.test(dataloaders=DATAMODULE.test_dataloader())
