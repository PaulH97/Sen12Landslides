import random 
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shutil
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
            raise ValueError("Need at least 3 bands for an RGB composite for Sentinel-2.")
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

if __name__ == "__main__":

    out_dir = Path(f"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/data/sanity_check")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # file= Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Inventories/patches_refined/S1-asc/chimanimani_s1asc_1246.nc")
    # out_file = Path(f"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/data/sanity_check/{file.stem}.png")
    # run_sanity_check_xarray(file, out_file)

    s2_files = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/data/raw/s1dsc").glob("*.nc")
    s2_files = random.sample(list(s2_files), 20)
    for file in s2_files:
        out_file =  out_dir / f"{file.stem}.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        run_sanity_check_xarray(file, out_file)