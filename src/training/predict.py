import torch
import wandb
from lightning import Trainer
from lightning.pytorch.loggers import Logger
import gc
import traceback
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
from pathlib import Path
from omegaconf import DictConfig
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import random

from src.data.data_loading import get_dataloaders

def adjust_channels(channels, variant):
    channels = int(channels)
    # If the variant is 'no_dem', subtract one channel (to drop the DEM band)
    if variant == "no_dem":
        return channels - 1
    return channels

def select_best_ckpt(ckpt_dir):
    best_ckpt = None
    best_val_loss = float('inf')
    for ckpt_path in ckpt_dir.glob("*.ckpt"):
        ckpt_name = ckpt_path.name
        try:
            val_loss_str = ckpt_name.split("val_loss=")[-1].split(".ckpt")[0]
            val_loss = float(val_loss_str)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = ckpt_path
        except (IndexError, ValueError) as e:
            logging.warning(f"Could not parse val_loss from checkpoint name {ckpt_name}: {e}")
    return best_ckpt

def save_batch_predictions(batch, batch_idx, output_folder):
    """
    For each sample in the batch, create and save a figure that shows:
      - Three images from selected timesteps (first, middle, and last),
      - The prediction image,
      - The ground truth mask.
    
    Each sample is saved as its own figure.
    
    Args:
        batch: Dictionary containing:
            - "imgs": Tensor of shape [B, T, C, H, W].
            - "preds": Tensor of shape [B, 1, H, W].
            - "masks": Tensor of shape [B, 1, H, W].
        output_folder: Path (string or Path) where the figures will be saved.
    """
    # Convert tensors to numpy arrays.
    preds_np = batch["preds"].detach().cpu().numpy()   # shape: [B, H, W]
    masks_np = batch["masks"].detach().cpu().numpy()     # shape: [B, H, W]
    imgs_np  = batch["imgs"].detach().cpu().numpy()       # shape: [B, T, C, H, W]
        
    B, T, C, H, W = imgs_np.shape

    # Choose three timesteps: if T >= 15 use [0, 10, 14], else use first, middle, last.
    if T >= 15:
        time_indices = [0, 10, 14]
    else:
        time_indices = [0, T//2, T-1]

    # Helper: convert a 2D grayscale image to RGB.
    def ensure_rgb(image):
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            return np.repeat(image, 3, axis=-1)
        return image

    # Process each sample in the batch.
    for b in range(B):
        n_cols = len(time_indices) + 2  # three timesteps, one prediction, one mask.
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        
        # Plot selected timesteps.
        for i, t in enumerate(time_indices):
            img = imgs_np[b, t]  # shape: [C, H, W]
            C_current = img.shape[0]
            if C_current > 3:
                # Sentinel-2: use first 3 channels for RGB.
                rgb_img = img[:3, :, :]
                rgb_img = np.transpose(rgb_img, (1, 2, 0))  # shape: [H, W, 3]
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
                axes[i].imshow(rgb_img)
            elif C_current <= 3:
                # Sentinel-1: combine the first two channels (e.g. average) to form a grayscale image.
                gray_img = np.mean(img[:2, :, :], axis=0)  # shape: [H, W]
                gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min() + 1e-8)
                axes[i].imshow(gray_img, cmap='gray')
            else:
                # If only one channel or other: convert to RGB.
                single = np.transpose(img, (1, 2, 0))  # shape: [H, W, C]
                single = (single - single.min()) / (single.max() - single.min() + 1e-8)
                single = ensure_rgb(single)
                axes[i].imshow(single)
            axes[i].set_title(f"Time {t}")
            axes[i].axis('off')
        
        # Plot prediction.
        pred_img = preds_np[b]  # shape: [H, W]
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
        pred_img = ensure_rgb(pred_img)
        axes[len(time_indices)].imshow(pred_img, cmap='gray')
        axes[len(time_indices)].set_title("Prediction")
        axes[len(time_indices)].axis('off')
        
        # Plot ground truth mask.
        mask_img = masks_np[b]  # shape: [H, W]
        mask_img = (mask_img - mask_img.min()) / (mask_img.max() - mask_img.min() + 1e-8)
        mask_img = ensure_rgb(mask_img)
        axes[len(time_indices) + 1].imshow(mask_img, cmap='gray')
        axes[len(time_indices) + 1].set_title("Mask")
        axes[len(time_indices) + 1].axis('off')
        
        out_path = Path(output_folder) / "images" / f"batch{batch_idx}_sample_{b}.png"
        out_path.parent.mkdir(exist_ok=True, parents=True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out_path)
        plt.close(fig)
        logging.info(f"Saved sample {b} to {out_path}")

OmegaConf.register_new_resolver("adjust_channels", adjust_channels)

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    logging.info(OmegaConf.to_yaml(cfg))
    try:
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
        logging.info(f"Train dataset size: {len(train_loader.dataset)}, Val dataset size: {len(val_loader.dataset)}, Test dataset size: {len(test_loader.dataset)}")
     
        callback = instantiate(cfg.callback)
        ckpt_dir = Path(callback.dirpath)
        ckpt_path = select_best_ckpt(ckpt_dir)

        trainer= Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            default_root_dir=cfg.log_dir,
            callbacks=[callback]
        )

        model = instantiate(cfg.model)
        predictions = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)

        preds_dir = Path(cfg.output_dir) / "predictions"
        preds_dir.mkdir(exist_ok=True, parents=True)
        
        random_batches = random.sample(predictions, 2) 
        for idx, batch in enumerate(random_batches):
            save_batch_predictions(batch, idx, preds_dir)
        
        preds = torch.cat([batch["preds"] for batch in predictions], dim=0)
        masks = torch.cat([batch["masks"] for batch in predictions], dim=0)
        pred_file = preds_dir / f"predictions.pt"
        torch.save({"preds": preds, "masks": masks}, pred_file)
        
        logging.info(f"Predictions saved to {pred_file}")

    except Exception as e:
        traceback.print_exc()
        print(f"Training crashed on iteration with error: {e}")
    finally:
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()

# python /dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/src/training/test.py --multirun dataset=s2,s1asc,s1dsc