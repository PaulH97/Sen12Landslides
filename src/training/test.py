import torch
import wandb
from lightning import Trainer
import gc
import traceback
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
from pathlib import Path
import json
from lightning.pytorch import seed_everything

from src.data.data_loading import get_dataloaders
import random
from torch.utils.data import Subset
from tqdm import tqdm

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

def filter_dataloader(test_loader, seed=42, non_annotated_percentage=0.2):
    random.seed(seed)
    filtered_indices = []
    for idx, sample in enumerate(tqdm(test_loader.dataset, desc="Filtering test dataset")):
        # Check if the mask tensor has any annotation (a value of 1).
        has_annotation = sample["msk"].eq(1).any().item()
        if has_annotation:
            filtered_indices.append(idx)
        else:
            if random.random() < non_annotated_percentage:
                filtered_indices.append(idx)

    original_test_size = len(test_loader.dataset)
    filtered_test_size = len(filtered_indices)
    logging.info(f"Test dataset filtered: from {original_test_size} to {filtered_test_size} samples")

    # Re-create the test DataLoader with the filtered dataset
    return torch.utils.data.DataLoader(
        Subset(test_loader.dataset, filtered_indices),
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory
    )

OmegaConf.register_new_resolver("adjust_channels", adjust_channels)

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    # logging.info(OmegaConf.to_yaml(cfg))
    try:
        seed_everything(cfg.seed)
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
        logging.info(f"Train dataset size: {len(train_loader.dataset)}, Val dataset size: {len(val_loader.dataset)}, Test dataset size: {len(test_loader.dataset)}")
        
        if cfg.experiment.name == "exp3":
            # Filter the test dataset to include only 20% of non-annotated samples
            test_loader = filter_dataloader(test_loader, seed=cfg.seed, non_annotated_percentage=0.2)

        # Check the input shape of the data
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch["img"].shape
        logging.info(f"Input shape: {input_shape}")

        callback = instantiate(cfg.callback)        
        ckpt_dir = Path(callback.dirpath)
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))

        for idx, ckpt_path in enumerate(ckpt_files):
            
            trainer= Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                default_root_dir=cfg.output_dir,
                devices=1,
                callbacks=[callback]
            )

            model = instantiate(cfg.model)
            metrics = trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)
            
            metrics_file = Path(cfg.output_dir) / f"metrics_{idx}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Test metrics saved to {metrics_file}")

            del model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()

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