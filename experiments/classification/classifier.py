from typing import Mapping

from lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import nll_loss
from torch.optim import SGD
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
    MulticlassAveragePrecision,
)


class ImageClassifier(LightningModule):
    """Lightning module for image classification experiments."""

    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model
        self.metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=10),
                "precision": MulticlassPrecision(num_classes=10),
                "recall": MulticlassRecall(num_classes=10),
                "auroc": MulticlassAUROC(num_classes=10),
                "ap": MulticlassAveragePrecision(num_classes=10),
            }
        )
        self.optimizer = SGD(self.model.parameters(), lr=0.5)
        self.training_step_outputs: list[Tensor] = []
        self.validation_step_outputs: list[tuple[Tensor, Tensor]] = []
        self.test_step_outputs: list[tuple[Tensor, Tensor]] = []

    def training_step(self, batch: Tensor) -> Tensor:
        data, target = batch
        preds = self.model(data)
        loss = nll_loss(preds, target)
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

    def validation_step(self, batch: Tensor) -> Mapping[str, Tensor]:
        data, target = batch
        preds = self.model(data)
        self.validation_step_outputs.append((preds, target))
        return {"preds": preds, "target": target}

    def on_validation_epoch_end(self) -> None:
        for preds, target in self.validation_step_outputs:
            self.metrics.update(preds=preds, target=target)
        metric_results = self.metrics.compute()
        self.metrics.reset()
        self.log_dict(metric_results, prog_bar=True, logger=True)

    def test_step(self, batch: Tensor) -> Mapping[str, Tensor]:
        data, target = batch
        preds = self.model(data)
        self.test_step_outputs.append((preds, target))

        return {"preds": preds, "target": target}

    def on_test_epoch_end(self) -> None:
        for preds, target in self.test_step_outputs:
            self.metrics.update(preds=preds, target=target)
        metric_results = self.metrics.compute()
        test_results = {
            name + "_test": result for name, result in metric_results.items()
        }
        self.metrics.reset()
        self.log_dict(test_results, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.optimizer
