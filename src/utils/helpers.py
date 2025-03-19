import random 
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shutil
import xarray as xr

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