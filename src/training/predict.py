import torch
from pathlib import Path 
import yaml
import pytorch_lightning as pl
import wandb
import gc
import argparse
import random 
from matplotlib import pyplot as plt
import numpy as np

from benchmarking.models.modelTask import ClassificationTask
from benchmarking.data_loading import get_dataloaders
from benchmarking.models import get_model
import traceback

def save_batch_predictions(preds, masks, imgs, output_folder, batch_idx):
    """
    Save one figure per sample showing three timesteps (first, middle, last) from imgs,
    along with static prediction and mask (each with shape [B, 1, 128, 128]).
    
    Args:
        preds: Tensor with shape [B, 1, 128, 128].
        masks: Tensor with shape [B, 1, 128, 128].
        imgs: Tensor with shape [B, 15, 11, 128, 128].
        output_folder: Folder path where images will be saved.
        batch_idx: Batch index used for naming files.
    """
    # Convert tensors to numpy arrays.
    preds_np = preds.detach().cpu().numpy()   # shape: [B, 1, 128, 128]
    masks_np = masks.detach().cpu().numpy()     # shape: [B, 1, 128, 128]
    imgs_np  = imgs.detach().cpu().numpy()       # shape: [B, 15, 11, 128, 128]
    
    # Squeeze the singleton channel in preds and masks.
    preds_np = np.squeeze(preds_np, axis=1)  # shape: [B, 128, 128]
    masks_np = np.squeeze(masks_np, axis=1)  # shape: [B, 128, 128]
    
    B, T, C, H, W = imgs_np.shape  # e.g. (64, 15, 11, 128, 128)
    # Choose three timesteps: first, middle, and last.
    time_indices = [0, 10, 14]
    
    for b in range(B):
        fig, axes = plt.subplots(3, len(time_indices), figsize=(12, 12))
        for col, t in enumerate(time_indices):
            # Process image from imgs[b, t] with shape (11, 128, 128).
            img = imgs_np[b, t]
            # If the channel count is not 1 or 3, select the first 3 channels.
            if img.shape[0] not in [1, 3]:
                img = img[:3]
            # Convert from (C, H, W) to (H, W, C).
            img = np.transpose(img, (1, 2, 0))
            # If the result is (128, 128, 1), squeeze to (128, 128).
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            # Apply min-max normalization to the image.
            img_min = img.min()
            img_max = img.max()
            img = (img - img_min) / (img_max - img_min)
            
            # For preds and masks, use the static version.
            pred = preds_np[b]   # shape: (128, 128)
            mask = masks_np[b]   # shape: (128, 128)
            
            # Plot RGB image.
            axes[0, col].imshow(img)
            axes[0, col].set_title(f'RGB (t={t})')
            axes[0, col].axis('off')
            
            # Plot prediction.
            axes[1, col].imshow(pred, cmap='gray')
            axes[1, col].set_title('Prediction')
            axes[1, col].axis('off')
            
            # Plot ground truth.
            axes[2, col].imshow(mask, cmap='gray')
            axes[2, col].set_title('Ground Truth')
            axes[2, col].axis('off')
        
        plt.tight_layout()
        out_path = Path(output_folder, f"batch{batch_idx}_sample{b}.png")
        plt.savefig(out_path)
        plt.close(fig)

def predict(model_config, iteration_idx):
    try:
        model_config = Path(model_config)
        with open(model_config, 'r') as file:
            config = yaml.safe_load(file)
        model_name = model_config.stem

        # Prepare data paths and directories
        base_dir = Path(config["DATA"]["base_dir"])
        exp_data_dir = Path(config["DATA"]["exp_data_dir"])
        satellite = config["DATA"]["satellite"]
        data_dir = exp_data_dir / satellite / f"i{iteration_idx}"
        data_paths_file = data_dir / "data_paths.json"
        norm_file = data_dir / "norm_data.json"
        
        model_dir = data_dir / model_name 
        train_dir = model_dir / "training"
        test_dir = model_dir / "test"

        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(f"{train_dir} or {test_dir} does not exist. Make sure they are created during the pretraining setup.")

        # Get dataloaders
        train_dl, val_dl, test_dl = get_dataloaders(config, base_dir, data_paths_file, norm_file)

        print("Length of train_dl: ", len(train_dl))
        print("Length of val_dl: ", len(val_dl))
        print("Length of test_dl: ", len(test_dl))

        ckpt_path = train_dir / "checkpoints" / f"{model_name}-best.ckpt"
                    
        model = get_model(config)
        model = ClassificationTask.load_from_checkpoint(model=model, checkpoint_path=ckpt_path, config=config)

        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

        # Save predictions (only on rank 0)
        if trainer.is_global_zero:
            predictions = trainer.predict(model, test_dl)
            predict_dir = Path(test_dir, "predictions")
            predict_dir.mkdir(parents=True, exist_ok=True)

            # Select specific batches to save
            specific_batches = random.sample(range(len(predictions)), 2)  
            for batch_idx in specific_batches:
                if batch_idx < len(predictions):
                    batch = predictions[batch_idx]
                    preds = batch["preds"]
                    masks = batch["masks"]
                    imgs = batch["images"]

                    # Ensure tensors are on CPU
                    preds = preds.cpu()
                    masks = masks.cpu()
                    imgs = imgs.cpu()
                    
                    save_batch_predictions(preds, masks, imgs, predict_dir, batch_idx)
                else:
                    print(f"Batch index {batch_idx} is out of range.")

    except Exception as e:
        print(f"Training crashed on iteration {data_dir.stem} with error: {e}")
        traceback.print_exc()
    finally:
        wandb.finish()
        torch.cuda.empty_cache()  # Clear GPU memory after each iteration
        gc.collect()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific iteration index.")
    parser.add_argument("-c", "--model_config", required=True, help="Model config file")
    parser.add_argument("-i", "--iteration", type=int, required=True, help="Iteration index")
    args = parser.parse_args()
    predict(args.model_config, args.iteration)

# salloc --account=hpda-c --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --partition=hpda2_compute_gpu --time=02:00:00
# salloc --account=hpda-c --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=2 --partition=hpda2_testgpu --time=01:00:00
