from pathlib import Path
import json
import os
from torch.utils.data import DataLoader

from src.data.transforms import Compose, NoDataReplace, RandomFlip, Normalize
from src.data.datasets import SITS_HDF5, SITS

def get_dataloaders(cfg):

    exp_dir = Path(cfg.exp_dir)
    batch_size = cfg.train.batch_size   
    nodata_value = cfg.train.no_data_value
    h5_training = cfg.train.h5

    norm_file = exp_dir / "norm_data.json"
    with open(norm_file, "r") as f:
        norm_dict = json.load(f)
    mean, std = norm_dict["mean"], norm_dict["std"]

    dataloaders = {}
    for dataset_type in ["train", "val", "test"]:
        if dataset_type == "train":
            transforms_pipeline = Compose([
                NoDataReplace(global_means=mean, nodata_value=nodata_value),
                # RandomFlip(horizontal_flip_prob=0.5, vertical_flip_prob=0.5),
                Normalize(mean=mean, std=std)
            ])
        else:
            transforms_pipeline = Compose([
                NoDataReplace(global_means=mean, nodata_value=nodata_value),
                Normalize(mean=mean, std=std)
            ])

        if h5_training:
            h5_path = exp_dir / f"dataset_{dataset_type}.h5"
            data_loader = DataLoader(
                dataset=SITS_HDF5(h5_path, transforms=transforms_pipeline),
                batch_size=batch_size,
                num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 8)),
                shuffle=True if dataset_type == "train" else False,
                pin_memory=True
            )
        else:
            data_paths_file = exp_dir / f"data_paths.json"

            with open(data_paths_file, "r") as f:
                data_paths = json.load(f)

            data_paths = [Path(cfg.base_dir) / path for path in data_paths[dataset_type]] # make absolut paths
            data_loader = DataLoader(
                dataset=SITS(data_paths, transforms=transforms_pipeline),
                batch_size=batch_size,
                num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 8)),
                shuffle=True if dataset_type == "train" else False,
                pin_memory=True
            )

        dataloaders[dataset_type] = data_loader

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]

# Example usage:
# train_loader, val_loader, test_loader = get_dataloaders(config, h5_folder, norm_file)
