import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def postprocess_s2_dataset(ds):
    """
    Apply ESA radiometric offset correction to Sentinel-2 L2A data.

    - Looks for bands named B01…B12 in `ds`.
    - If acquisition date is after 2022-01-25, subtracts 1000 from those bands.
    - Returns a new Dataset with corrected band values and all other variables untouched.

    Args:
        ds (xarray.Dataset): must contain reflectance bands “B01”–“B12” and a time coord.

    Returns:
        xarray.Dataset: corrected Sentinel-2 dataset.
    """
    # identify reflectance bands
    s2_bands = [b for b in ds.data_vars if b.startswith("B") and b[1:].isdigit()]
    if not s2_bands:
        return ds

    # stack into (band, time, x, y)
    da = ds[s2_bands].to_array(dim="band")

    # get first timestamp
    t0 = np.array(da.time.values)[0]
    if np.datetime64("2022-01-25") < t0:
        # subtract offset
        mask = np.isin(da.band.values, s2_bands)
        da.values[mask] = np.where(
            np.isnan(da.values[mask]), np.nan, da.values[mask] - 1000
        )

    # unpack back into dataset and merge other vars
    ds_corr = da.to_dataset(dim="band")
    return xr.merge([ds_corr, ds.drop_vars(s2_bands)])


def postprocess_s1_dataset(ds, epsilon=1e-6):
    """
    Convert Sentinel-1 RTC backscatter to decibels (dB) by applying 10 * log10 transform,
    replacing any <=0 values with a small epsilon so that no NaNs are introduced.

    Args:
        ds (xarray.Dataset): must contain one or both of “VV”, “VH”.
        epsilon (float): small floor value for non-positive backscatter (default 1e-6).

    Returns:
        xarray.Dataset: corrected Sentinel-1 dataset without NaNs.
    """
    ds_out = ds.copy()
    for pol in ("VV", "VH"):
        if pol in ds_out.data_vars:
            da = ds_out[pol].values
            # floor to epsilon
            da_floored = np.where(da <= 0, epsilon, da)
            # convert to dB
            ds_out[pol] = 10 * np.log10(da_floored)
    return ds_out


def run_sanity_check(data_loader, output_dir, num_samples=10):
    """
    Generate visualization samples from a data loader for sanity checking.
    This function creates visualizations of random samples from the dataset,
    displaying both the input images and corresponding masks. For Sentinel-1 data
    (C<=3), it shows VV and VH polarizations separately. For Sentinel-2 or RGB data
    (C>3), it displays RGB composite images. All visualizations are saved to the
    specified output directory.
    Parameters:
    -----------
    data_loader : torch.utils.data.DataLoader
        The data loader containing the dataset to visualize
    output_dir : pathlib.Path
        Directory where visualization images will be saved.
        Will be created if it doesn't exist or cleared if it does.
    num_samples : int, default=10
        Number of random samples to visualize from the dataset
    Notes:
    ------
    - Expects data_loader.dataset to return dict with 'img' and 'msk' keys
    - 'img' should be a tensor of shape [T, C, H, W] where:
        T = number of time steps
        C = number of channels
        H = height
        W = width
    - 'msk' should be a tensor of shape [H, W] or [1, H, W]
    - For C<=3, assumes Sentinel-1 style data (VV, VH, optional DEM)
    - For C>3, assumes first 3 channels can be used as RGB
    """

    # Remove old sanity-check folder if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick random indices from the dataset
    random_indices = random.sample(range(len(data_loader.dataset)), num_samples)

    for sample_idx in random_indices:
        sample = data_loader.dataset[sample_idx]
        img = sample["img"].numpy()  # [T, C, H, W]
        msk = sample["msk"].numpy()  # [H, W] or [1, H, W]

        # Flatten mask if needed
        mask_img = msk.squeeze()

        T, C, H, W = img.shape
        output_file = output_dir / f"sample_{sample_idx}.png"

        if C <= 3:
            # ------------------------
            # Sentinel-1 style VV & VH (+ DEM)
            # ------------------------
            fig, axs = plt.subplots(2, T + 1, figsize=(3 * (T + 1), 8), sharey=True)

            for t in range(T):
                vv = img[t, 0, :, :]
                vh = img[t, 1, :, :]

                def normalize(arr):
                    rng = arr.max() - arr.min()
                    return (arr - arr.min()) / rng if rng != 0 else np.zeros_like(arr)

                vv_norm = normalize(vv)
                vh_norm = normalize(vh)

                axs[0, t].imshow(vv_norm, cmap="gray")
                axs[0, t].axis("off")
                axs[0, t].set_title(f"Time {t} VV", fontsize=8)

                axs[1, t].imshow(vh_norm, cmap="gray")
                axs[1, t].axis("off")
                axs[1, t].set_title(f"Time {t} VH", fontsize=8)

            # Mask in the last column
            axs[0, -1].imshow(mask_img, cmap="gray")
            axs[0, -1].axis("off")
            axs[0, -1].set_title("Mask", fontsize=8)
            axs[1, -1].imshow(mask_img, cmap="gray")
            axs[1, -1].axis("off")

        else:
            # ------------------------
            # Sentinel-2 style (RGB)
            # ------------------------
            fig, axs = plt.subplots(1, T + 1, figsize=(3 * (T + 1), 6), sharey=True)

            for t in range(T):
                # Use first 3 channels as RGB
                # Adjust if your data uses a different ordering!
                red = img[t, 2, :, :] if C >= 3 else img[t, 0, :, :]
                green = img[t, 1, :, :] if C >= 2 else img[t, 0, :, :]
                blue = img[t, 0, :, :]

                rgb = np.stack([red, green, blue], axis=-1)
                rgb_min, rgb_max = rgb.min(), rgb.max()
                if rgb_max - rgb_min != 0:
                    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
                else:
                    rgb_norm = np.zeros_like(rgb)

                axs[t].imshow(rgb_norm)
                axs[t].axis("off")
                axs[t].set_title(f"Time {t}", fontsize=8)

            # Mask in the last column
            axs[-1].imshow(mask_img, cmap="gray")
            axs[-1].axis("off")
            axs[-1].set_title("Mask", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved sanity check to: {output_file}")
