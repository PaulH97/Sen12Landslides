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
from lightning.pytorch import seed_everything
from datetime import datetime

from src.data.data_loading import get_dataloaders
from src.utils.helpers import run_sanity_check

def adjust_channels(channels, variant):
    channels = int(channels)
    # If the variant is 'no_dem', subtract one channel (to drop the DEM band)
    if variant == "no_dem":
        return channels - 1
    return channels

OmegaConf.register_new_resolver("adjust_channels", adjust_channels)

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3.2")
def main(cfg):
    # logging.info(OmegaConf.to_yaml(cfg))
    try:
        # seed_everything(cfg.seed)
        logging.info(f"Seed set to {cfg.seed}")

        train_loader, val_loader, test_loader = get_dataloaders(cfg)
        logging.info(f"Train dataset size: {len(train_loader.dataset)}, Val dataset size: {len(val_loader.dataset)}, Test dataset size: {len(test_loader.dataset)}")

        # Check the input shape of the data
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch["img"].shape
        logging.info(f"Input shape: {input_shape}")

        # sanity check
        if cfg.sanity_check:
            for name, loader in {"train": train_loader, "val": val_loader, "test": test_loader}.items():
                sanity_check_dir = Path("sanity_check") / name
                sanity_check_dir.mkdir(parents=True, exist_ok=True)
                run_sanity_check(loader, sanity_check_dir, num_samples=5)
        
        cfg_to_log = {
            "dataset": cfg.dataset.name,
            "model": cfg.model.name,
            "exp": cfg.experiment.name,
            "exp_variant": cfg.experiment.variant,
            "seed": cfg.seed,
        }
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.callback.filename = "{epoch}-{val_loss:.4f}" + f"-{timestamp}" 
        callback = instantiate(cfg.callback)
        logger = instantiate(cfg.logger)
        Path(logger.save_dir).mkdir(parents=True, exist_ok=True)
        logger.experiment.config.update(cfg_to_log)

        trainer= Trainer(
            logger=logger,
            callbacks=[callback],
            **cfg.trainer
        )

        model = instantiate(cfg.model)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
     
    except Exception as e:
        traceback.print_exc()
        print(f"Training crashed on iteration with error: {e}")
    finally:
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()

# python /dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/src/training/train.py --multirun dataset=s2,s1asc,s1dsc