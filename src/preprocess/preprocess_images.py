import argparse
import atexit
import logging
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml
from modules.dating_module import apply_dating
from modules.ndvi_module import extract_ndvi
from modules.patch_module import create_patches
from modules.stack_module import stack_s1_bands, stack_s2_bands

np.seterr(divide="ignore", invalid="ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def read_config(config_file):
    """
    Read and parse a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary containing the parsed configuration.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def cleanup_temp():
    """
    Cleans up temporary files by removing the temporary directory.

    This function retrieves the system's temporary directory path,
    logs a message indicating which directory is being cleaned up,
    and then removes the entire temporary directory and its contents.
    The removal operation ignores any errors that occur during deletion.

    Returns:
        None
    """
    temp_dir = tempfile.gettempdir()
    logging.info(f"Cleaning up temporary files in {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)


def main(inv_name, config_file):
    """
    Main function to process satellite imagery for landslide detection.
    This function orchestrates various preprocessing tasks based on the provided configuration,
    including stacking Sentinel-1 and Sentinel-2 bands, extracting NDVI values, applying dating
    algorithms, and creating image patches for machine learning.
    Parameters
    ----------
    inv_name : str
        Name of the inventory/area to process
    config_file : str
        Path to the configuration file containing processing parameters
    Notes
    -----
    The configuration file should include:
    - base_dir: Base directory for data
    - dataset_type: Type of dataset to process (default: "final") # raw, final
    - patch_size: Size of patches to create (default: 96)
    - overlap: Overlap fraction between patches (default: 0.25)
    - timeseries_length: Number of timesteps in the series (default: 50)
    - seed: Random seed for reproducibility (default: 42)
    - tasks: List of preprocessing tasks to perform
    """

    config = read_config(config_file)
    dataset_type = config.get("dataset_type", "final")
    inv_dir = Path(config.get("base_dir")) / inv_name
    patch_size = config.get("patch_size", 96)
    overlap = config.get("overlap", 0.25)
    overlap_pixel = int(patch_size * overlap)
    ts_length = config.get("timeseries_length", 50)
    seed = random.seed(config.get("seed", 42))

    for task in config["tasks"]:

        if task["name"] == "stack_s1_bands":
            logging.info(f"Stacking S1 bands of {inv_dir.name}")
            s1_asc_dir = inv_dir / "sentinel-1-new" / "asc"
            s1_dsc_dir = inv_dir / "sentinel-1-new" / "dsc"
            stack_s1_bands(s1_asc_dir)
            stack_s1_bands(s1_dsc_dir)

        elif task["name"] == "stack_s2_bands":
            logging.info(f"Stacking S2 bands of {inv_dir.name}")
            s2_dir = inv_dir / "sentinel-2" / "images"
            stack_s2_bands(s2_dir)

        elif task["name"] == "extract_ndvi":
            logging.info(f"Extract NDVI values for annotations of {inv_dir.name}")
            extract_ndvi(inv_dir, dataset_type)

        elif task["name"] == "apply_dating":
            logging.info(f"Apply dating with NDVI values of {inv_dir.name}")
            apply_dating(inv_dir, dataset_type)

        elif task["name"] == "create_patches":
            logging.info(
                f"Create patches of Sentinel-1-asc/dsc and Sentinel-2 data of {inv_dir.name}"
            )
            create_patches(
                inv_dir, dataset_type, patch_size, overlap_pixel, ts_length, seed=seed
            )


if __name__ == "__main__":
    atexit.register(cleanup_temp)
    parser = argparse.ArgumentParser(description="Process inventory name.")
    parser.add_argument(
        "-i",
        "--inventory",
        type=str,
        required=True,
        help="Name of the inventory to process",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    main(args.inventory, args.config)
