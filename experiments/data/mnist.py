from pathlib import Path

from lightning import LightningDataModule

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToImage, Normalize, ToDtype

from vintage_models.components.image import MaybeToColor


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        train_batch_size: int = 64,
        test_batch_size: int = 1,
        color: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_dims = (1, 28, 28)
        self.class_count = 10
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        transform = [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize((0.1307,), (0.3081,)),
        ]
        if color:
            transform.append(MaybeToColor())

        self.transform = Compose(transform)

        self.mnist_val: MNIST
        self.mnist_train: MNIST
        self.mnist_test: MNIST

    def prepare_data(self):
        # download
        MNIST(str(self.data_dir), train=True, download=True)
        MNIST(str(self.data_dir), train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(str(self.data_dir), train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                str(self.data_dir), train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.train_batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.test_batch_size)
