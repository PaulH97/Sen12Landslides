import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex
)
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ClassificationTask(pl.LightningModule):
    """
    PyTorch Lightning module for binary segmentation with either:
      - CrossEntropyLoss (2-channel output, class indices 0 or 1)
      - BCEWithLogitsLoss (1-channel output, float mask 0.0 or 1.0)
    """

    def __init__(self, model: nn.Module, config: dict):
        super().__init__()

        # Unpack config
        train_cfg = config["TRAINING"]
        self.model = model
        self.threshold = train_cfg.get("threshold", 0.5)
        self.loss_type = train_cfg.get("loss_type", "ce").lower()
        self.lr = train_cfg["lr"]
        self.optimizer_cls = getattr(optim, train_cfg["optimizer"])

        # We'll only apply pos_weight for BCE
        self.register_buffer("pos_weight_bce", torch.tensor([20.0]))

        # Create the loss function
        if self.loss_type == "ce":
            # 2-channel output -> integer mask
            self.lossFN = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(self.device))
        elif self.loss_type == "bce":
            # 1-channel output -> float mask
            self.lossFN = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_bce)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Define metrics
        metrics = MetricCollection([
            BinaryF1Score(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryJaccardIndex()
        ])
        self.trainMetricsFN = metrics.clone(prefix="train_")
        self.valMetricsFN   = metrics.clone(prefix="val_")
        self.testMetricsFN  = metrics.clone(prefix="test_")

        # Save hyperparams (except the model)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sub-model."""
        return self.model(x)

    def get_binary_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert raw logits to binary predictions (0 or 1).
        - CE: 2-channel, use argmax over dim=1
        - BCE: 1-channel, apply sigmoid + threshold
        """
        if self.loss_type == "ce":
            # shape (B, 2, H, W)
            preds = torch.argmax(logits, dim=1).float()
        else:
            # shape (B, 1, H, W)
            preds = (torch.sigmoid(logits) > self.threshold).float()
        return preds

    def training_step(self, batch, batch_idx):
        images, masks = batch["img"], batch["msk"]

        # Convert masks based on the loss function
        if self.loss_type == "ce":
            masks = masks.squeeze(1).long()
        else:
            masks = masks.float()

        # Forward + loss
        logits = self(images)
        loss = self.lossFN(logits, masks)

        # Update metrics
        preds = self.get_binary_predictions(logits)
        self.trainMetricsFN.update(preds, masks.int())

        # Log step-wise training loss (and let Lightning do epoch avg)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["img"], batch["msk"]

        if self.loss_type == "ce":
            masks = masks.squeeze(1).long()
        else:
            masks = masks.float()

        logits = self(images)
        loss = self.lossFN(logits, masks)
        preds = self.get_binary_predictions(logits)

        # Update val metrics
        self.valMetricsFN.update(preds, masks.int())

        # Log val loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch["img"], batch["msk"]

        if self.loss_type == "ce":
            masks = masks.squeeze(1).long()
        else:
            masks = masks.float()

        logits = self(images)
        loss = self.lossFN(logits, masks)
        preds = self.get_binary_predictions(logits)
        self.testMetricsFN.update(preds, masks.int())

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            self.testMetricsFN.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch["img"], batch["msk"]
        logits = self(images)
        preds = self.get_binary_predictions(logits)
        preds = preds.unsqueeze(1)
        return {"preds": preds, "images": images, "masks": masks}

    def on_train_epoch_end(self):
        # Compute & log training metrics
        train_metrics = self.trainMetricsFN.compute()
        self.log_dict(
            train_metrics,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.trainMetricsFN.reset()

    def on_validation_epoch_end(self):
        # Compute & log validation metrics
        val_metrics = self.valMetricsFN.compute()
        self.log_dict(
            val_metrics,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.valMetricsFN.reset()

    def on_test_epoch_end(self):
        # Compute & log test metrics
        test_metrics = self.testMetricsFN.compute()
        self.log_dict(
            test_metrics,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self.testMetricsFN.reset()

    def configure_optimizers(self):
        """
        Configure optimizer and LR scheduler.
        Uses ReduceLROnPlateau by default, monitoring 'val_loss'.
        """
        optimizer = self.optimizer_cls(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
