import torch
from pathlib import Path 
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import gc
import argparse
import traceback

from benchmarking.models.modelTask import ClassificationTask
from benchmarking.data_loading import get_dataloaders
from benchmarking.models import get_model
import os

def safe_rmdir(path):
    if path and Path(path).exists():
        print(f"Removing temporary directory: {path}")
        for item in Path(path).iterdir():
            if item.is_dir():
                safe_rmdir(item)  # Recursively delete subdirectories
            else:
                item.unlink()    # Delete file
        Path(path).rmdir() 

def main(model_config, iteration_idx):
    """Main training function that handles training, testing, and predictions."""
    try:
        model_config = Path(model_config)
        with open(model_config, 'r') as file:
            config = yaml.safe_load(file)
        model_name = model_config.stem

        # Get the iteration data
        base_dir = Path(config["DATA"]["base_dir"])
        exp_data_dir = Path(config["DATA"]["exp_data_dir"])
        satellite = config["DATA"]["satellite"]
        data_dir = exp_data_dir / satellite / f"i{iteration_idx}"
        data_paths_file = data_dir / "data_paths.json"
        norm_file = data_dir / "norm_data.json"

        # Load pre-existing output directories created by pretrain_setup.py
        model_dir = data_dir / model_name 
        train_dir = model_dir / "training"
        test_dir = model_dir / "test"

        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(f"{train_dir} or {test_dir} does not exist. Make sure they are created during the pretraining setup.")

        # Initialize dataloaders for training, validation, and testing datasets
        train_dl, val_dl, test_dl = get_dataloaders(config, base_dir, data_paths_file, norm_file)

        print("Length of train_dl: ", len(train_dl))
        print("Length of val_dl: ", len(val_dl))
        print("Length of test_dl: ", len(test_dl))

        # sanity check 
                    
        model = get_model(config)
        task = ClassificationTask(model, config)
        
        ckpt_path = train_dir / "checkpoints" / f"{model_name}-best.ckpt"

        # Define trainer and start training
        num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", 1))

        # Create shortcuts for iteration and refined data
        experiment = next(part.replace("_", "") for part in exp_data_dir.parts if "exp" in part)
        dataset_type = "refined" if any("refined" in part for part in exp_data_dir.parts) else "original"

        wb_logger = WandbLogger(
            name=f'{config["MODEL"]["architecture"]}_{experiment}_{satellite}_{dataset_type}_i{iteration_idx}',
            save_dir=train_dir,
            project='Sen12Landslides',
            group=f"{satellite}_{dataset_type}",
            job_type=f"I-{iteration_idx}",
            tags=[
                config["MODEL"]["architecture"],
                satellite,
                f"I-{iteration_idx}", 
                dataset_type.capitalize(), 
                experiment,
                "train"
            ],
            mode="online"
        )

        # Dynamically set log_every_n_steps based on length of dataloader
        num_logs_per_epoch = 10
        log_every_n_steps = max(1, len(train_dl) // num_logs_per_epoch)

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            strategy="ddp",
            num_nodes=num_nodes, 
            log_every_n_steps=log_every_n_steps,
            max_epochs=config["TRAINING"]["epochs"],
            callbacks=[
            ModelCheckpoint(monitor="val_BinaryF1Score", dirpath=ckpt_path.parent, filename=ckpt_path.stem, save_top_k=1, mode="max", save_last=True)
            ],
            logger=wb_logger,
            default_root_dir=None,
            precision="16-mixed"
        )

        trainer.fit(model=task, train_dataloaders=train_dl, val_dataloaders=val_dl)

    except Exception as e:
        traceback.print_exc()
        print(f"Training crashed on iteration {iteration_idx} with error: {e}")
    finally:
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()

# salloc --account=pn76xa-c --cpus-per-task=8 --ntasks-per-node=4 --gres=gpu:4 --partition=hpda2_compute_gpu --time=03:00:00