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

OmegaConf.register_new_resolver("adjust_channels", adjust_channels)

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    logging.info(OmegaConf.to_yaml(cfg))
    try:
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
        logging.info(f"Train dataset size: {len(train_loader.dataset)}, Val dataset size: {len(val_loader.dataset)}, Test dataset size: {len(test_loader.dataset)}")
        
        logger = instantiate(cfg.logger)
        callback = instantiate(cfg.callback)

        ckpt_dir = Path(callback.dirpath)
        ckpt_path = select_best_ckpt(ckpt_dir)

        trainer= Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            default_root_dir=cfg.log_dir,
            logger=logger,
            devices=1,
            callbacks=[callback]
        )

        model = instantiate(cfg.model)
        
        metrics = trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)
        metrics_file = Path(cfg.output_dir) / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Test metrics saved to {metrics_file}")

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