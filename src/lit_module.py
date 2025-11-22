import torch
from lightning.pytorch import LightningModule
from hydra.utils import instantiate
from torchmetrics import MetricCollection
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, dice_w: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice_w = dice_w
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight.to(logits.device)
        )
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.flatten()
        targets_flat = targets.flatten()
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return (1.0 - self.dice_w) * bce + self.dice_w * dice_loss


class LitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        task: str,
        loss: dict,
        metrics: dict,
        optimizer: dict = None,
        scheduler: dict = None,
        compile: bool = False,
        monitor_metric: str = "val/loss",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        self.net = torch.compile(net) if compile else net
        self.task = task
        self.threshold = threshold

        self.loss_fn = instantiate(loss, _convert_="object")

        base_metrics = MetricCollection({name: instantiate(m) for name, m in metrics.items()})
        self.train_metrics = base_metrics.clone(prefix="train/")
        self.val_metrics = base_metrics.clone(prefix="val/")
        self.test_metrics = base_metrics.clone(prefix="test/")

        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.monitor_metric = monitor_metric

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch, stage: str):
        inputs = batch["img"]
        targets = batch["msk"]
        
        out = self(inputs)
        logits = out.get("segmentation", None)
        if logits is None:
            raise RuntimeError("Model output missing key 'segmentation'")

        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits detected in {stage}")

        if self.task == "multiclass":
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets[:, 0]
            targets = targets.long()
            loss = self.loss_fn(logits, targets)
            preds = torch.argmax(logits, dim=1)
            metric_inputs = preds
            metric_targets = targets

        elif self.task == "binary":
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            loss = self.loss_fn(logits, targets)
            probs = torch.sigmoid(logits).squeeze(1)
            metric_inputs = probs.flatten()
            metric_targets = (targets > 0.5).long().squeeze(1).flatten()

        else:
            raise ValueError(f"Unknown task type: {self.task}")

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected in {stage}")

        metrics = getattr(self, f"{stage}_metrics")
        metrics.update(metric_inputs, metric_targets)

        self.log(f"{stage}/loss", loss,
                batch_size=targets.shape[0],
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(metrics,
                batch_size=targets.shape[0],
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        inputs = batch["img"]
        targets = batch["msk"]
        
        out = self(inputs)
        logits = out["segmentation"]
        
        if self.task == "binary":
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).long()
            return {"preds": preds, "targets": targets, "probs": probs}
        else:
            preds = torch.argmax(logits, dim=1)
            return {"preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.net.parameters())
        if self.scheduler_cfg:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}