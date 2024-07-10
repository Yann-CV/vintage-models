from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from experiments.classification.classifier import ImageClassifier
from experiments.data.mnist import MNISTDataModule
from vintage_models.vision_transformers.vit.vit import ViT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCH_COUNT = 100

MODEL = ViT(
    patch_size=8,
    image_width=28,
    image_height=28,
    embedding_len=64,
    mlp_hidden_size=32,
    head_count=2,
    layer_count=4,
    class_count=10,
)
CLASSIFIER = ImageClassifier(MODEL)

LOGGER = MLFlowLogger(
    experiment_name="ViT on MNIST",
    tracking_uri="/storage/ml/mlruns",
    run_name=str(MODEL),
    log_model=True,
)

CHECKPOINT_CALLBACK = ModelCheckpoint(
    save_top_k=1,
    monitor="training_loss",
    mode="min",
    dirpath="/storage/ml/models",
    filename="vit-mnist-{epoch:02d}-{accuracy:.2f}",
)

DATAMODULE = MNISTDataModule(Path("/storage/ml"), train_batch_size=2000)


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
