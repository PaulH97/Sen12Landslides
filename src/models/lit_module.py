import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hydra.utils import instantiate
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply softmax directly along the channel dimension
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot encoding if needed
        if targets.shape[1] == 1:
            targets = torch.cat([1 - targets, targets], dim=1)
        
        # Flatten the tensors to [B, 2, H*W]
        probs_flat = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.contiguous().view(targets.size(0), targets.size(1), -1)
        
        intersection = (probs_flat * targets_flat).sum(-1)
        cardinality = probs_flat.sum(-1) + targets_flat.sum(-1)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score.mean()
        return dice_loss

class DC_and_CE_loss(nn.Module):
    def __init__(self):
        super(DC_and_CE_loss, self).__init__()
        # Standard cross-entropy loss for pixel-wise classification
        self.ce = nn.CrossEntropyLoss()
        # Simplified soft Dice loss
        self.dc = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Parameters:
          logits: logits of shape [B, 2, H, W]
          target: ground truth tensor of shape [B, 1, H, W] with values 0 (no anomaly) or 1 (anomaly)
        """
        # CrossEntropyLoss expects targets of shape [B, H, W] with class indices
        ce_loss = self.ce(logits, target[:, 0].long())
        dice_loss = self.dc(logits, target)
        return ce_loss + dice_loss

class Sen12LsLitModule(LightningModule):
    """
    PyTorch Lightning module for binary segmentation with either:
      - CrossEntropyLoss (2-channel output, class indices 0 or 1)
      - BCEWithLogitsLoss (1-channel output, float mask 0.0 or 1.0)
    """

    def __init__(
        self,
        name: str,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool
        ):
        super().__init__()

        # Save hyperparameters (except the model)
        self.save_hyperparameters(ignore=['net'], logger=False)

        self.net = net

        # loss function
        self.criterion  = torch.nn.CrossEntropyLoss()

        num_classes = self.net.num_classes
        metrics = MetricCollection({
            "F1Score": MulticlassF1Score(num_classes=num_classes, average='macro'),
            "Precision": MulticlassPrecision(num_classes=num_classes, average='macro'),
            "Recall": MulticlassRecall(num_classes=num_classes, average='macro'),
            "Jaccard": MulticlassJaccardIndex(num_classes=num_classes, average='macro')
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics  = metrics.clone(prefix="val_")
        self.test_metrics  = metrics.clone(prefix="test_")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sub-model."""
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
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)
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
