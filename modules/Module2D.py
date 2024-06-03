import logging
import torch
import pytorch_lightning as pl
from torch import Tensor, nn
from metrics import IoUMetricOneClass, DiceMetric

log = logging.getLogger(__name__)


class SegmentationModule(pl.LightningModule):

    def __init__( 
        self,
        model: nn.Module,
        criterion: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> pl.LightningModule:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "criterion",
            ]
        )

        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.metrics = [IoUMetricOneClass(0.5), DiceMetric(0.5)]

    def forward(self, x: Tensor):
        return self.model(x)

    def training_step(
        self, batch: list[Tensor], batch_idx: int
    ) -> Tensor:
        data, label = batch
        output = self.forward(data)
        if isinstance(output, tuple):
            output, loss_vae = output
            loss = self.criterion(output, label) + loss_vae
        else:
            loss = self.criterion(output, label)
        iou = self.metrics[0](output, label)
        dice = self.metrics[1](output, label)
        
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )

        self.log(
            "train_iou", iou, on_step=False, on_epoch=True, prog_bar=True
        )

        self.log(
            "train_dice", dice, on_step=False, on_epoch=True, prog_bar=True
        )
        
        return loss

    def validation_step(  
        self, batch: list[Tensor], batch_idx: int
    ) -> Tensor:
        data, label = batch
        output = self.model(data)
        loss = self.criterion(output, label)
        iou = self.metrics[0](output, label)
        dice = self.metrics[1](output, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_iou", iou, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_dice", dice, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def test_step(
        self, batch: list[Tensor], batch_idx: int
    ) -> None:
        data, label = batch
        output = self.model(data)
        loss = self.criterion(output, label)
        iou = self.metrics[0](output, label)
        dice = self.metrics[1](output, label)

        self.log(
            "test_iou", iou, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_dice", dice, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers( 
        self,
    ):
            opt = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
            return [opt], []

