import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassF1Score,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision, MulticlassRecall)


class Sen12LsLitModule(LightningModule):
    """
    A PyTorch Lightning module for remote sensing image segmentation tasks using the Sen12Landslides dataset.
    This module handles the training, validation, testing, and prediction steps for a semantic segmentation model.
    It includes metrics for evaluating segmentation performance such as F1 Score, Precision, Recall, and Jaccard Index.
    Attributes:
        net (torch.nn.Module): The neural network model for segmentation.
        criterion (torch.nn.Module): Loss function (CrossEntropyLoss) for training.
        train_metrics (torchmetrics.MetricCollection): Collection of metrics for training.
        val_metrics (torchmetrics.MetricCollection): Collection of metrics for validation.
        test_metrics (torchmetrics.MetricCollection): Collection of metrics for testing.
    Args:
        name (str): Name of the model.
        net (torch.nn.Module): The neural network architecture.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        compile (bool): Whether to compile the model using torch.compile.
    """

    def __init__(
        self,
        name: str,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        super().__init__()

        # Save hyperparameters (except the model)
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        num_classes = self.net.num_classes
        metrics = MetricCollection(
            {
                "F1Score": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "Precision": MulticlassPrecision(
                    num_classes=num_classes, average="macro"
                ),
                "Recall": MulticlassRecall(num_classes=num_classes, average="macro"),
                "Jaccard": MulticlassJaccardIndex(
                    num_classes=num_classes, average="macro"
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch["img"], batch["msk"]
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1).long()
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, masks.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["img"], batch["msk"]
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1).long()
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        self.val_metrics.update(preds, masks.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch["img"], batch["msk"]
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1).long()
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, masks.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch["img"], batch["msk"]
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1).long()
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "imgs": images, "masks": masks}

    @rank_zero_only
    def on_train_epoch_end(self):
        val_metrics = self.train_metrics.compute()
        self.log_dict(val_metrics, sync_dist=True)
        self.val_metrics.reset()

    @rank_zero_only
    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, sync_dist=True)
        self.val_metrics.reset()

    @rank_zero_only
    def on_test_epoch_end(self):
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, sync_dist=True)
        self.test_metrics.reset()

    # def setup(self, stage):
    #     if self.hparams.compile and stage == "fit":
    #         self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
