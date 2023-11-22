from pathlib import Path

from lightning import Fabric
from pytorch_lightning.loggers import MLFlowLogger
from torch.nn.functional import nll_loss
from torch.optim import SGD

from vintage_models.experiments.data.mnist import MNISTDataModule
from vintage_models.vision_transformers.vit.vit import ViT

EPOCH_COUNT = 100
MODEL = ViT(
    patch_size=16,
    image_width=28,
    image_height=28,
    embedding_len=64,
    mlp_hidden_size=32,
    head_count=2,
    layer_count=4,
    class_count=10,
)
OPTIMIZER = SGD(MODEL.parameters(), lr=0.1)
LOGGER = MLFlowLogger(
    experiment_name="ViT on MNIST",
    tracking_uri="/storage/ml/mlruns",
    run_name=str(MODEL),
    log_model=True,
)

DATAMODULE = MNISTDataModule(Path("/storage/ml"), train_batch_size=4000)
DATAMODULE.prepare_data()
DATAMODULE.setup("fit")

FABRIC = Fabric(accelerator="cuda", loggers=[LOGGER])
FABRIC.launch()

model, optimizer = FABRIC.setup(MODEL, OPTIMIZER)

train_loader = FABRIC.setup_dataloaders(DATAMODULE.train_dataloader())

for epoch in range(EPOCH_COUNT):
    print(f"starting epoch {epoch}")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        FABRIC.backward(loss)
        optimizer.step()
    FABRIC.log("loss", loss.item(), epoch)
    print(f"Loss epoch {epoch}: {loss.item()}")
