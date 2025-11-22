import hydra
import torch
import gc
import logging
import traceback
import os
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from lightning import seed_everything

from src.utils.helpers import normalize_percentile

def plot_predictions(preds, targets, s2_data, out_path):
    """Plot S2 RGB timesteps with target and prediction."""
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds.squeeze(0)
    if targets.ndim == 3 and targets.shape[0] == 1:
        targets = targets.squeeze(0)
    
    T = s2_data.shape[0]
    n_cols = T + 3  # timesteps + target + prediction + overlay
    
    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 2.5))
    
    # Plot S2 RGB timesteps
    for t in range(T):
        frame = s2_data[t]  # [C, H, W]
        rgb = frame[:3].transpose(1, 2, 0)  # [H, W, 3]
        rgb = normalize_percentile(rgb)
        axes[t].imshow(rgb)
        axes[t].set_title(f"T={t}", fontsize=8)
        axes[t].axis("off")
    
    # Target
    axes[T].imshow(targets, cmap="gray")
    axes[T].set_title("Target", fontsize=8)
    axes[T].axis("off")
    
    # Prediction
    axes[T + 1].imshow(preds, cmap="gray")
    axes[T + 1].set_title("Prediction", fontsize=8)
    axes[T + 1].axis("off")
    
    # Overlay (last timestep + prediction)
    last_rgb = s2_data[-1][:3].transpose(1, 2, 0)
    last_rgb = normalize_percentile(last_rgb)
    axes[T + 2].imshow(last_rgb)
    overlay = np.zeros((*preds.shape, 4))
    overlay[preds == 1] = [1, 0, 0, 0.5]
    axes[T + 2].imshow(overlay)
    axes[T + 2].set_title("Overlay", fontsize=8)
    axes[T + 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg):
    try:
        torch.set_float32_matmul_precision('high')
        seed_everything(cfg.seed, workers=True)

        datamodule = instantiate(cfg.datamodule)
        datamodule.setup(stage="test")
        
        model = instantiate(cfg.model.instance, _convert_="all")
        lit_module = instantiate(cfg.module, net=model, _convert_="all", _recursive_=False)
        trainer = instantiate(cfg.trainer, logger=False, _convert_="all")

        ckpt_path = cfg.get("ckpt_path", None)
        
        test_loader = datamodule.test_dataloader()
        preds = trainer.predict(lit_module, dataloaders=test_loader, ckpt_path=ckpt_path)

        out_dir = cfg.get("preds_out_dir", "predictions")
        os.makedirs(out_dir, exist_ok=True)
        
        for i, (batch_out, batch) in enumerate(zip(preds, test_loader)):
            if i >= 10:
                break
            
            p = batch_out["preds"].cpu().numpy()
            t = batch_out["targets"].cpu().numpy()
            s2_data = batch["img"]["S2"]["data"].cpu().numpy()  # [B, T, C, H, W]
            
            for j in range(len(p)):
                plot_predictions(p[j], t[j], s2_data[j], os.path.join(out_dir, f"pred_{i}_{j}.png"))

        logging.info(f"Saved predictions in {out_dir}")

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Prediction crashed: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

        
if __name__ == "__main__":
    main()