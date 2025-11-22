import random
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

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


def normalize_percentile(arr, p_low=2, p_high=98):
    """Normalize array using percentile stretching."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    p2, p98 = np.percentile(valid, (p_low, p_high))
    if p98 > p2:
        return np.clip((arr - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(arr)


def run_sanity_check(datamodule, output_dir, num_samples=5):
    """Visualize samples from datamodule for sanity checking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datamodule.setup()
    dataset = datamodule.train_ds
    
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        sample = dataset[idx]
        
        img = sample["img"].numpy()   # [T, C, H, W]
        msk = sample["msk"].numpy()   # [H, W]
        
        T, C, H, W = img.shape
        
        # Detect modality and DEM: S1 (2 or 3), S2 (10 or 11)
        has_dem = C in [3, 11]
        is_s1 = C <= 3
        
        n_cols = T + 2 + (1 if has_dem else 0)  # timesteps + mask + overlay + DEM
        fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 2.5))
        
        # Plot timesteps
        for t in range(T):
            frame = img[t]
            if is_s1:
                axes[t].imshow(normalize_percentile(frame[0]), cmap="gray")
            else:
                rgb = normalize_percentile(frame[:3].transpose(1, 2, 0))
                axes[t].imshow(rgb)
            axes[t].set_title(f"T={t}", fontsize=8)
            axes[t].axis("off")
        
        col = T
        
        # DEM (last channel if present)
        if has_dem:
            dem = img[0, -1]  # DEM is same for all timesteps
            axes[col].imshow(normalize_percentile(dem), cmap="terrain")
            axes[col].set_title("DEM", fontsize=8)
            axes[col].axis("off")
            col += 1
        
        # Mask
        axes[col].imshow(msk, cmap="gray")
        axes[col].set_title("Mask", fontsize=8)
        axes[col].axis("off")
        col += 1
        
        # Overlay
        last_frame = img[-1]
        if is_s1:
            axes[col].imshow(normalize_percentile(last_frame[0]), cmap="gray")
        else:
            axes[col].imshow(normalize_percentile(last_frame[:3].transpose(1, 2, 0)))
        
        overlay = np.zeros((*msk.shape, 4))
        overlay[msk == 1] = [1, 0, 0, 0.5]
        axes[col].imshow(overlay)
        axes[col].set_title("Overlay", fontsize=8)
        axes[col].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved: sample_{idx}.png")
    
    print(f"\nSaved {num_samples} samples to {output_dir}")