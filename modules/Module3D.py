import logging
import torch
import pytorch_lightning as pl
from torch import Tensor, nn
from metrics import IoUMetricOneClass, DiceMetric, IoUMetricOneClass3D, DiceMetric3D
from monai.inferers import SlidingWindowInferer

log = logging.getLogger(__name__)

class SegmentationModule3D(pl.LightningModule):

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
        self.metrics = [IoUMetricOneClass3D(0.5), DiceMetric3D(0.5), IoUMetricOneClass(0.5), DiceMetric(0.5)]

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
        inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=2, overlap=0.7)
        mean_loss = 0
        mean_iou_3d = 0
        mean_dice_3d = 0
        mean_iou_2d = 0
        mean_dice_2d = 0
        for i, sample in enumerate(data):
            output = inferer(sample[None], self.model)
            mean_loss += self.criterion(output, label[i][None])
            mean_iou_3d += self.metrics[0](output, label[i][None])
            mean_dice_3d += self.metrics[1](output, label[i][None])
            
            mean_iou_2d += self.metrics[2](output[0].permute(3, 0, 1, 2), label[i].permute(3, 0, 1, 2))
            mean_dice_2d += self.metrics[3](output[0].permute(3, 0, 1, 2), label[i].permute(3, 0, 1, 2))
            
        self.log("val_loss", mean_loss / len(data), on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_iou_3d", mean_iou_3d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_dice_3d", mean_dice_3d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_iou_2d", mean_iou_2d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_dice_2d", mean_dice_2d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
        return mean_loss

    def test_step(
        self, batch: list[Tensor], batch_idx: int
    ) -> None:  # noqa: D102
        data, label = batch
        inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=2, overlap=0.7)
        mean_loss = 0
        mean_iou_3d = 0
        mean_dice_3d = 0
        mean_iou_2d = 0
        mean_dice_2d = 0
        for i, sample in enumerate(data):
            output = inferer(sample[None], self.model)
            mean_loss += self.criterion(output, label[i][None])
            mean_iou_3d += self.metrics[0](output, label[i][None])
            mean_dice_3d += self.metrics[1](output, label[i][None])
            
            mean_iou_2d += self.metrics[2](output[0].permute(3, 0, 1, 2), label[i].permute(3, 0, 1, 2))
            mean_dice_2d += self.metrics[3](output[0].permute(3, 0, 1, 2), label[i].permute(3, 0, 1, 2))
        self.log("test_loss", mean_loss / len(data), on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_iou_3d", mean_iou_3d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_dice_3d", mean_dice_3d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_iou_2d", mean_iou_2d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test_dice_2d", mean_dice_2d / len(data), on_step=False, on_epoch=True, prog_bar=True
        )
    def configure_optimizers(  # noqa: D102
        self,
    ):
            opt = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
            return [opt], []