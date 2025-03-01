import random
import shutil
from matplotlib import pyplot as plt
import numpy as np

def run_sanity_check(train_dl, satellite,  output_dir):

    def s1_sanity_check(sample, sample_idx, output_dir):

        output_file = output_dir / f"sample_{sample_idx}"

        img = sample['img'].numpy()  # Convert the image to NumPy
        msk = sample['msk'].numpy()  # Convert the mask to NumPy


        # Plot each time step
        timesteps = img.shape[0]
        fig, axs = plt.subplots(2, timesteps + 1, figsize=(3 * (timesteps + 1), 8), sharey=True)

        # Plot each time step for the image data
        for i in range(timesteps):
            vh = img[i, 0, :, :]
            vv = img[i, 1, :, :]  

            # Normalize for visualization with safe division
            vh_range = vh.max() - vh.min()
            vv_range = vv.max() - vv.min()

            vh_norm = (vh - vh.min()) / vh_range if vh_range != 0 else np.zeros_like(vh)
            vv_norm = (vv - vv.min()) / vv_range if vv_range != 0 else np.zeros_like(vv)

            # Plot VV polarization for the given timestep
            axs[0, i].imshow(vv_norm, cmap='gray')
            axs[0, i].axis('off')
            axs[0, i].set_title(f'Time {i} VV', fontsize=8)

            # Plot VH polarization for the given timestep
            axs[1, i].imshow(vh_norm, cmap='gray')
            axs[1, i].axis('off')
            axs[1, i].set_title(f'Time {i} VH', fontsize=8)

        # Plot the mask in the last column
        mask_img = msk.squeeze()
        axs[0, -1].imshow(mask_img, cmap='gray')
        axs[0, -1].set_title("Mask", fontsize=8)
        axs[0, -1].axis('off')
        axs[1, -1].imshow(mask_img, cmap='gray')
        axs[1, -1].axis('off')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
        plt.close()

        print(f"Saved S1 sanity check to: {output_file}")

    def s2_sanity_check(sample, sample_idx, output_dir):

        output_file = output_dir / f"sample_{sample_idx}"

        img = sample['img'].numpy()  # Convert the image to NumPy
        msk = sample['msk'].numpy()  # Convert the mask to NumPy

        # Plot each time step
        timesteps = img.shape[0]
        fig, axs = plt.subplots(1, timesteps + 1, figsize=(3 * (timesteps + 1), 6), sharey=True)

        # Plot each time step for the image data
        for i in range(timesteps):
            rgb_img = np.stack([img[i, 2, :, :], img[i, 1, :, :], img[i, 0, :, :]], axis=-1)  # Assuming B02, B03, B04 order
            rgb_img_min, rgb_img_max = rgb_img.min(), rgb_img.max()
            rgb_img_norm = (rgb_img - rgb_img_min) / (rgb_img_max - rgb_img_min)

            # Plot RGB image
            axs[i].imshow(rgb_img_norm)
            axs[i].axis('off')
            axs[i].set_title(f'Time {i}', fontsize=8)

        # Plot the mask in the last column
        mask_img = msk.squeeze()
        axs[-1].imshow(mask_img, cmap='gray')
        axs[-1].axis('off')
        axs[-1].set_title("Mask", fontsize=8) 

        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

        print(f"Saved S2 sanity check to: {output_file}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_indices = random.sample(range(len(train_dl.dataset)), 10)
    
    for random_idx in random_indices:
        random_sample = train_dl.dataset[random_idx]  # Get the random sample from the dataset
        if satellite in ["S2", "S2_noDEM"]:
            s2_sanity_check(random_sample, random_idx, output_dir)
        elif satellite in ["S1-asc", "S1-dsc", "S1-asc_noDEM", "S1-dsc_noDEM"]:
            s1_sanity_check(random_sample, random_idx, output_dir)
        else:
            raise ValueError(f"Satellite {satellite} not recognized")