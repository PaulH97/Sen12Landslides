from pathlib import Path
import json
import os
from torch.utils.data import DataLoader

from src.data.transforms import Compose, Normalize, RemoveChannel, RandomFlip
from src.data.datasets import SITS

def get_dataloaders(cfg):

    exp_dir = Path(cfg.exp_dir)
    rm_channel_idx = -1 if (cfg.experiment.variant == "no_dem") else None
    
    norm_file = exp_dir / "norm_data.json"
    with open(norm_file, "r") as f:
        norm_dict = json.load(f)
    mean, std = norm_dict["mean"], norm_dict["std"]

    dataloaders = {}
    for dataset_type in ["train", "val", "test"]:
        transforms_pipeline = Compose([
            RemoveChannel(channel_idx=rm_channel_idx),
            Normalize(mean=mean, std=std, rm_channel_idx=rm_channel_idx),
        ])

        if cfg.model.name == "stvit":
            transforms_pipeline = Compose([
            RemoveChannel(channel_idx=rm_channel_idx),
            Normalize(mean=mean, std=std, rm_channel_idx=rm_channel_idx),
        ])
        
        data_paths_file = exp_dir / f"data_paths.json"
        with open(data_paths_file, "r") as f:
            data_paths = json.load(f)

        data_paths = [Path(cfg.base_dir) / path for path in data_paths[dataset_type]] # make absolut paths
        # data_paths = [path for path in data_paths if "chimanimani" in path.name]

        data_loader = DataLoader(
            dataset=SITS(data_paths, transforms=transforms_pipeline),
            batch_size=cfg.dataset.batch_size,
            num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 8))-2, # 2 CPUs for other tasks
            shuffle=True if dataset_type == "train" else False,
            pin_memory=True
        )

        dataloaders[dataset_type] = data_loader

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]

# Example usage:
# train_loader, val_loader, test_loader = get_dataloaders(config, h5_folder, norm_file)
