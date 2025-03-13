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
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt

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
            ts_images.append(ds[bands[0]].values[t])
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

def run_sanity_check(data_loader, output_dir, num_samples=10):
    """
    Runs a simple sanity check by randomly sampling 'num_samples' patches
    from the 'train_dl' DataLoader and visualizing them.

    Logic:
      - If the sample has 2 channels, we assume it's S1 (VV & VH).
      - If it has more than 2 channels, we assume it's S2 and plot the first 3 as RGB.
    
    Args:
        train_dl: PyTorch DataLoader whose dataset returns dicts with {'img': tensor, 'msk': tensor}
        output_dir: Directory to save the generated sanity-check images
        num_samples: Number of random samples to visualize
    """

    # Remove old sanity-check folder if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick random indices from the dataset
    random_indices = random.sample(range(len(data_loader.dataset)), num_samples)

    for sample_idx in random_indices:
        sample = data_loader.dataset[sample_idx]
        img = sample['img'].numpy()  # [T, C, H, W]
        msk = sample['msk'].numpy()  # [H, W] or [1, H, W]

        # Flatten mask if needed
        mask_img = msk.squeeze()

        T, C, H, W = img.shape
        output_file = output_dir / f"sample_{sample_idx}.png"

        if C <= 3:
            # ------------------------
            # Sentinel-1 style VV & VH (+ DEM)
            # ------------------------
            fig, axs = plt.subplots(
                2, T + 1,
                figsize=(3 * (T + 1), 8),
                sharey=True
            )

            for t in range(T):
                vv = img[t, 0, :, :]
                vh = img[t, 1, :, :]

                def normalize(arr):
                    rng = arr.max() - arr.min()
                    return (arr - arr.min()) / rng if rng != 0 else np.zeros_like(arr)

                vv_norm = normalize(vv)
                vh_norm = normalize(vh)

                axs[0, t].imshow(vv_norm, cmap='gray')
                axs[0, t].axis('off')
                axs[0, t].set_title(f"Time {t} VV", fontsize=8)

                axs[1, t].imshow(vh_norm, cmap='gray')
                axs[1, t].axis('off')
                axs[1, t].set_title(f"Time {t} VH", fontsize=8)

            # Mask in the last column
            axs[0, -1].imshow(mask_img, cmap='gray')
            axs[0, -1].axis('off')
            axs[0, -1].set_title("Mask", fontsize=8)
            axs[1, -1].imshow(mask_img, cmap='gray')
            axs[1, -1].axis('off')

        else:
            # ------------------------
            # Sentinel-2 style (RGB)
            # ------------------------
            fig, axs = plt.subplots(
                1, T + 1,
                figsize=(3 * (T + 1), 6),
                sharey=True
            )

            for t in range(T):
                # Use first 3 channels as RGB
                # Adjust if your data uses a different ordering!
                red   = img[t, 2, :, :] if C >= 3 else img[t, 0, :, :]
                green = img[t, 1, :, :] if C >= 2 else img[t, 0, :, :]
                blue  = img[t, 0, :, :]

                rgb = np.stack([red, green, blue], axis=-1)
                rgb_min, rgb_max = rgb.min(), rgb.max()
                if rgb_max - rgb_min != 0:
                    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
                else:
                    rgb_norm = np.zeros_like(rgb)

                axs[t].imshow(rgb_norm)
                axs[t].axis('off')
                axs[t].set_title(f"Time {t}", fontsize=8)

            # Mask in the last column
            axs[-1].imshow(mask_img, cmap='gray')
            axs[-1].axis('off')
            axs[-1].set_title("Mask", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved sanity check to: {output_file}")
    
if __name__ == "__main__":

    # file= Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Inventories/patches_refined_test/S2/chimanimani_s2_1217.nc")
    # out_file = Path(f"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/data/sanity_check/{file.stem}.png")
    # run_sanity_check_xarray(file, out_file)

    s2_files = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Inventories/patches_refined_test/S2").glob("*.nc")
    s2_files = random.sample(list(s2_files), 50)
    for file in s2_files:
        out_file = Path(f"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/data/sanity_check/{file.stem}.png")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        run_sanity_check_xarray(file, out_file)