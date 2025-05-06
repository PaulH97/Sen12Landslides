import json
import os
from pathlib import Path

from src.data.datasets import SITS
from src.data.transforms import Compose, Normalize, RandomFlip, RemoveChannel
from torch.utils.data import DataLoader


def get_dataloaders(cfg):
    """
    Creates and returns data loaders for training, validation, and testing.
    This function sets up dataloaders with appropriate transformations based on
    the configuration. It handles normalization, optional DEM channel removal,
    and configures batch sizes and workers according to available resources.
    Parameters
    ----------
    cfg : Config
        Configuration object containing experiment settings:
        - exp_dir: Directory containing experiment files
        - base_dir: Base directory for data paths
        - experiment.variant: Experiment variant (e.g., "no_dem")
        - dataset.batch_size: Batch size for data loading
        - model.name: Name of the model being used
    Returns
    -------
    tuple
        A tuple of (train_loader, val_loader, test_loader), each being a PyTorch
        DataLoader instance configured with appropriate dataset and settings.
    Notes
    -----
    - Uses normalization parameters from a JSON file in the experiment directory
    - Removes DEM channel if experiment variant is "no_dem"
    - Sets number of workers based on SLURM environment or defaults to 8-2
    - Enables shuffling for training data only
    """

    exp_dir = Path(cfg.exp_dir)
    rm_channel_idx = -1 if (cfg.experiment.variant == "no_dem") else None

    norm_file = exp_dir / "norm_data.json"
    with open(norm_file, "r") as f:
        norm_dict = json.load(f)
    mean, std = norm_dict["mean"], norm_dict["std"]

    dataloaders = {}
    for dataset_type in ["train", "val", "test"]:
        transforms_pipeline = Compose(
            [
                RemoveChannel(channel_idx=rm_channel_idx),
                Normalize(mean=mean, std=std, rm_channel_idx=rm_channel_idx),
            ]
        )

        data_paths_file = exp_dir / f"data_paths.json"
        with open(data_paths_file, "r") as f:
            data_paths = json.load(f)

        data_paths = [
            Path(cfg.base_dir) / path for path in data_paths[dataset_type]
        ]  # make absolut paths

        data_loader = DataLoader(
            dataset=SITS(data_paths, transforms=transforms_pipeline),
            batch_size=cfg.dataset.batch_size,
            num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", 8))
            - 2,  # 2 CPUs for other tasks
            shuffle=True if dataset_type == "train" else False,
            pin_memory=True,
        )

        dataloaders[dataset_type] = data_loader

    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


# Example usage:
# train_loader, val_loader, test_loader = get_dataloaders(config, h5_folder, norm_file)
