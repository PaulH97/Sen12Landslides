import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def run_sanity_check_xarray(xr_file, output_file):
    """
    Simplified sanity check display.

    Plots a single row:
      - The left portion contains time-series images (RGB composite for Sentinel-2,
        grayscale for Sentinel-1) with the actual date over each image.
      - The right portion shows single images for DEM and MASK (if available) in that order.
        DEM is plotted using the 'viridis' colormap.

    Args:
        xr_file (str): Path to the input netCDF file.
        output_file (str): Path to save the output image.
    """
    ds = xr.open_dataset(xr_file)
    T = ds.sizes["time"]

    # Get time labels as strings (assumes the time coordinate is ISO formatted)
    time_labels = [str(t)[:10] for t in ds["time"].values]

    # Determine satellite type (default to Sentinel-2)
    satellite = ds.attrs.get("satellite", "s2").lower()
    is_s1 = satellite.startswith("s1")

    # Get band variables for time-series; skip MASK, DEM, SCL, spatial_ref, etc.
    skip_vars = {"MASK", "DEM", "SCL", "spatial_ref"}
    bands = [var for var in ds.data_vars if var not in skip_vars]

    # Build time-series images.
    ts_images = []
    if is_s1:
        # Sentinel-1: use the first available band as grayscale.
        if not bands:
            raise ValueError("No valid band variables found for Sentinel-1.")
        for t in range(T):
            img = ds[bands[0]].values[t]
            # Normalize using percentiles
            vmin, vmax = np.percentile(img, (5, 95))
            img = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
            ts_images.append(img)
    else:
        # Sentinel-2: require at least three bands for RGB.
        if len(bands) < 3:
            raise ValueError(
                "Need at least 3 bands for an RGB composite for Sentinel-2."
            )
        for t in range(T):
            # Stack first three bands into an RGB image.
            rgb = np.stack([ds[b].values[t] for b in bands[:3]], axis=-1)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
            ts_images.append(rgb)

    # Collect additional images: DEM and MASK (using the first time slice)
    extra_images = []
    extra_vars = []
    if "DEM" in ds.data_vars:
        extra_images.append(ds["DEM"].values[0])
        extra_vars.append("DEM")
    if "MASK" in ds.data_vars:
        extra_images.append(ds["MASK"].values[0])
        extra_vars.append("MASK")

    # Total columns: time-series images + extra images.
    n_ts = len(ts_images)
    n_extra = len(extra_images)
    total_cols = n_ts + n_extra

    # Create one row of subplots.
    fig, axs = plt.subplots(1, total_cols, figsize=(3 * total_cols, 5))
    if total_cols == 1:
        axs = [axs]

    # Plot time-series images with their real date as title.
    for i in range(n_ts):
        ax = axs[i]
        if is_s1:
            ax.imshow(ts_images[i], cmap="gray")
        else:
            ax.imshow(ts_images[i])
        ax.set_title(time_labels[i])
        ax.axis("off")

    # Plot extra images.
    for j, img in enumerate(extra_images):
        ax = axs[n_ts + j]
        if extra_vars[j] == "DEM":
            ax.imshow(img, cmap="viridis")
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(extra_vars[j])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved sanity check to: {output_file}")


def main(folder, output_folder, n_files=5):
    """
    Run sanity checks on a specified number of randomly selected netCDF files in a folder.
    This function processes netCDF files by performing sanity checks and saving the results
    as PNG images in a 'sanity_checks' subfolder. It randomly samples a specified number
    of files from the given directory.
    Args:
        folder (str): Path to the directory containing netCDF files to check.
        n_files (int, optional): Maximum number of files to process. Defaults to 5.
    Returns:
        None
    Notes:
        - Creates a 'sanity_checks' subdirectory in the input folder if it doesn't exist
        - Skips processing if no netCDF files are found
        - Catches and reports errors for individual file processing
        - Calls the run_sanity_check_xarray function for each file
    """
    folder_path = Path(folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Find all netCDF files
    nc_files = list(folder_path.glob("*.nc"))

    if not nc_files:
        print(f"No netCDF files found in {folder}")
        return

    # Sample n_files randomly (or all if fewer)
    sample_files = random.sample(nc_files, min(n_files, len(nc_files)))

    print(f"Running sanity checks on {len(sample_files)} files from {folder}")

    for i, file_path in enumerate(sample_files):
        output_file = output_folder / f"sanity_check_{file_path.stem}.png"
        print(f"[{i+1}/{len(sample_files)}] Processing {file_path.name}")
        try:
            run_sanity_check_xarray(str(file_path), str(output_file))
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run sanity checks on netCDF files")
    parser.add_argument(
        "--folder", type=str, help="Path to folder containing netCDF files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="sanity_checks",
        help="Path to output folder for sanity check images (default: 'sanity_checks')",
    )
    parser.add_argument(
        "--n_files",
        type=int,
        default=5,
        help="Number of files to randomly sample (default: 5)",
    )

    args = parser.parse_args()
    main(args.folder, args.output_folder, args.n_files)

    # python sanity_check.py --folder "Sen12Landslides/data/final/s2"
