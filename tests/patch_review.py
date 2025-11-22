import json
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
matplotlib.use('Agg') 

def load_splits(splits_path):
    """Load splits.json file."""
    with open(splits_path, 'r') as f:
        return json.load(f)


def normalize_rgb(rgb):
    """
    Normalize RGB bands for display using percentile stretching.
    """
    rgb_normalized = np.zeros_like(rgb, dtype=np.float32)
    
    for i in range(3):
        band = rgb[:, :, i]
        p2, p98 = np.percentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        if p98 > p2:
            rgb_normalized[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            rgb_normalized[:, :, i] = 0
    
    return rgb_normalized


def compute_cloud_score(ds, t, bands):
    """Compute a simple cloud/quality score for a timestep."""
    if 'SCL' in ds.data_vars:
        scl = ds['SCL'].values[t]
        cloud_pixels = np.sum((scl == 8) | (scl == 9) | (scl == 3))
        return cloud_pixels
    
    scores = []
    for band in bands[:min(3, len(bands))]:
        data = ds[band].values[t]
        bright_pixels = np.sum(data > np.percentile(data, 95))
        scores.append(bright_pixels)
    
    return np.mean(scores)


def select_best_timesteps(ds, bands, n_timesteps, satellite_type):
    """Select the best quality timesteps based on event date coverage."""
    T = ds.sizes["time"]
    
    if T <= n_timesteps:
        return list(range(T))
    
    event_date = None
    if 'event_date' in ds.attrs:
        try:
            event_date = np.datetime64(ds.attrs['event_date'])
        except:
            pass
    
    timestamps = ds["time"].values
    
    if event_date is None:
        if satellite_type.startswith("s1"):
            return np.linspace(0, T-1, n_timesteps, dtype=int).tolist()
        else:
            quality_scores = []
            for t in range(T):
                score = compute_cloud_score(ds, t, bands)
                quality_scores.append((t, score))
            quality_scores.sort(key=lambda x: x[1])
            best_indices = [idx for idx, _ in quality_scores[:n_timesteps]]
            best_indices.sort()
            return best_indices
    
    time_diffs = np.abs(timestamps - event_date)
    event_idx = np.argmin(time_diffs)
    
    n_before = n_timesteps // 2
    n_after = n_timesteps - n_before - 1
    
    before_indices = list(range(max(0, event_idx - n_before), event_idx))
    after_indices = list(range(event_idx, min(T, event_idx + n_after + 1)))
    
    selected = before_indices + after_indices
    
    if len(selected) < n_timesteps:
        additional_before = max(0, event_idx - n_timesteps)
        selected = list(range(additional_before, event_idx)) + after_indices
    
    if len(selected) < n_timesteps:
        additional_after = min(T, event_idx + n_timesteps)
        selected = before_indices + list(range(event_idx, additional_after))
    
    selected = selected[:n_timesteps]
    selected.sort()
    
    return selected


def visualize_and_save_patch(data_folder, patch, output_path, n_timesteps=5):
    """Visualize a single patch and save to file.
    
    Args:
        data_folder: Path to data folder
        patch: Patch dictionary
        output_path: Path to save the image
        n_timesteps: Number of timesteps to display (default: 5)
    """
    patch_id = patch['id']
    
    if 's2' not in patch:
        print(f"❌ No S2 data for patch {patch_id}")
        return False
    
    nc_path = Path(data_folder) / patch['s2']
    if not nc_path.exists():
        print(f"❌ File not found: {nc_path}")
        return False
    
    try:
        ds = xr.open_dataset(nc_path)
        T = ds.sizes["time"]
        
        satellite_type = ds.attrs.get("satellite", "s2").lower()
        skip_vars = {"MASK", "DEM", "SCL", "spatial_ref"}
        bands = [var for var in ds.data_vars if var not in skip_vars]
        
        selected_timesteps = select_best_timesteps(ds, bands, n_timesteps, satellite_type)
        time_labels = [str(ds["time"].values[t])[:10] for t in selected_timesteps]
        
        event_date = None
        if 'event_date' in ds.attrs:
            try:
                event_date = np.datetime64(ds.attrs['event_date'])
            except:
                pass
        
        # Build RGB images
        rgb_images = []
        if 'B04' in bands and 'B03' in bands and 'B02' in bands:
            for t in selected_timesteps:
                rgb = np.stack([
                    ds['B04'].values[t],
                    ds['B03'].values[t],
                    ds['B02'].values[t]
                ], axis=-1)
                rgb = normalize_rgb(rgb)
                rgb_images.append(rgb)
        else:
            print(f"❌ Missing RGB bands. Available: {bands}")
            ds.close()
            return False
        
        # Create figure: n_timesteps + DEM + Mask overlay
        n_cols = n_timesteps + 2
        fig, axs = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 5))
        
        # Plot RGB images
        for i, t_idx in enumerate(selected_timesteps):
            ax = axs[i]
            ax.imshow(rgb_images[i])
            
            title_text = f"{time_labels[i]}"
            title_color = 'black'
            
            if event_date is not None:
                timestamp = ds["time"].values[t_idx]
                if timestamp < event_date:
                    title_text += "\n(Before)"
                    title_color = 'blue'
                elif timestamp == event_date or abs((timestamp - event_date) / np.timedelta64(1, 'D')) < 5:
                    title_text += "\n(Event)"
                    title_color = 'red'
                else:
                    title_text += "\n(After)"
                    title_color = 'green'
            
            ax.set_title(title_text, fontsize=10, fontweight='bold', color=title_color)
            ax.axis('off')
        
        # Plot DEM
        if "DEM" in ds.data_vars:
            ax = axs[n_timesteps]
            dem = ds["DEM"].values[0]
            im = ax.imshow(dem, cmap='terrain')
            ax.set_title('DEM', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation (m)')
        else:
            axs[n_timesteps].text(0.5, 0.5, 'No DEM', ha='center', va='center', fontsize=12)
            axs[n_timesteps].axis('off')
        
        # Plot RGB with Mask overlay
        if "MASK" in ds.data_vars and len(rgb_images) > 0:
            ax = axs[n_timesteps + 1]
            background_rgb = rgb_images[-1].copy()
            mask = ds["MASK"].values[0]
            
            mask_rgba = np.zeros((*mask.shape, 4))
            mask_rgba[mask == 1] = [1, 0, 0, 0.6]
            
            ax.imshow(background_rgb)
            ax.imshow(mask_rgba)
            ax.set_title('Annotated', fontsize=11, fontweight='bold', color='red')
            ax.axis('off')
        else:
            axs[n_timesteps + 1].text(0.5, 0.5, 'No Mask', ha='center', va='center', fontsize=12)
            axs[n_timesteps + 1].axis('off')
        
        # Main title
        pixel_annotated = patch.get('pixel_annotated', 'N/A')
        event_date_str = ds.attrs.get('event_date', 'N/A')
        fig.suptitle(
            f"Patch: {patch_id} | Satellite: S2 | Event Date: {event_date_str} | Annotated Pixels: {pixel_annotated}",
            fontsize=15,
            fontweight='bold',
            y=0.98
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        ds.close()
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Error processing patch {patch_id}:")
        print(traceback.format_exc())
        return False


def save_patches_to_remove(patches_to_remove, output_file):
    """Save the list of patches to remove to a file."""
    with open(output_file, 'w') as f:
        for patch_id in patches_to_remove:
            f.write(f"{patch_id}\n")
    
    print(f"\n✓ Saved {len(patches_to_remove)} patches to remove in: {output_file}")


def load_progress(progress_file):
    """Load review progress from file."""
    if not os.path.exists(progress_file):
        return 0, []
    
    with open(progress_file, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            return 0, []
        
        current_idx = int(lines[0].strip())
        patches_to_remove = [line.strip() for line in lines[1:] if line.strip()]
        return current_idx, patches_to_remove


def save_progress(progress_file, current_idx, patches_to_remove):
    """Save review progress to file."""
    with open(progress_file, 'w') as f:
        f.write(f"{current_idx}\n")
        for patch_id in patches_to_remove:
            f.write(f"{patch_id}\n")


def interactive_patch_review(data_folder, splits_path, output_file="patches_to_remove.txt", 
                            review_dir="patch_review_images", n_timesteps=5):
    """
    Interactively review patches by saving images to disk.
    Works on remote servers without display.
    
    Args:
        data_folder: Path to data folder
        splits_path: Path to splits.json
        output_file: File to save patches to remove
        review_dir: Directory to save patch images
        n_timesteps: Number of timesteps to display (default: 5)
    """
    splits = load_splits(splits_path)
    
    # Collect all patches
    all_patches = []
    for split in ['train', 'val', 'test']:
        if split in splits:
            all_patches.extend(splits[split])

    selected_patches = [p for p in all_patches]
    print(f"Total patches to review after filtering: {len(all_patches)} -> {len(selected_patches)}")
    all_patches = selected_patches if len(selected_patches) > 0 else all_patches

    # Create review directory
    review_path = Path(review_dir)
    review_path.mkdir(exist_ok=True)
    
    progress_file = review_path / "progress.txt"
    
    # Load progress if exists
    current_idx, patches_to_remove = load_progress(progress_file)
    
    print(f"{'='*80}")
    print(f"Starting interactive patch review - {len(all_patches)} patches total")
    if current_idx > 0:
        print(f"Resuming from patch {current_idx + 1}")
    print(f"{'='*80}\n")
    print("Instructions:")
    print(f"  1. Images are saved to: {review_path.absolute()}")
    print(f"  2. Open the image file to review the patch")
    print(f"  3. Enter your decision:")
    print(f"     - 'k' or 'keep' = Keep the patch (good quality)")
    print(f"     - 'r' or 'remove' = Remove the patch (bad quality)")
    print(f"     - 's' or 'skip' = Skip this patch")
    print(f"     - 'q' or 'quit' = Quit and save results")
    print(f"{'='*80}\n")
    
    while current_idx < len(all_patches):
        patch = all_patches[current_idx]
        patch_id = patch['id']
        
        print(f"\n[{current_idx+1}/{len(all_patches)}] Reviewing patch: {patch_id}")
        print("-" * 80)
        
        # Save image
        image_path = review_path / f"current_patch.png"
        
        success = visualize_and_save_patch(data_folder, patch, image_path, n_timesteps)
        
        if not success:
            print("Skipping to next patch due to error...")
            current_idx += 1
            save_progress(progress_file, current_idx, patches_to_remove)
            continue
        
        print(f"✓ Image saved: {image_path.absolute()}")
        print(f"Patches to remove so far: {len(patches_to_remove)}")
        
        # Get user input
        while True:
            print("\nYour decision? (k=keep, r=remove, s=skip, q=quit): ", end='')
            decision = input().strip().lower()
            
            if decision in ['k', 'keep']:
                print(f"✓ Keeping patch: {patch_id}")
                current_idx += 1
                save_progress(progress_file, current_idx, patches_to_remove)
                break
                
            elif decision in ['r', 'remove']:
                patches_to_remove.append(patch_id)
                print(f"✗ Marked for removal: {patch_id}")
                current_idx += 1
                save_progress(progress_file, current_idx, patches_to_remove)
                break
                
            elif decision in ['s', 'skip']:
                print(f"⊘ Skipped patch: {patch_id}")
                current_idx += 1
                save_progress(progress_file, current_idx, patches_to_remove)
                break
                
            elif decision in ['q', 'quit']:
                print(f"\n{'='*80}")
                print("Quitting review process...")
                print(f"Patches reviewed: {current_idx}/{len(all_patches)}")
                print(f"Patches marked for removal: {len(patches_to_remove)}")
                print(f"{'='*80}\n")
                save_patches_to_remove(patches_to_remove, output_file)
                return patches_to_remove
                
            else:
                print("Invalid input. Please enter 'k', 'r', 's', or 'q'")
    
    # Review complete
    print(f"\n{'='*80}")
    print("Review complete!")
    print(f"Patches reviewed: {current_idx}/{len(all_patches)}")
    print(f"Patches marked for removal: {len(patches_to_remove)}")
    print(f"{'='*80}\n")
    save_patches_to_remove(patches_to_remove, output_file)
    
    # Clean up
    if image_path.exists():
        image_path.unlink()
    if progress_file.exists():
        progress_file.unlink()
    
    return patches_to_remove


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        description='Interactive patch review tool - works without display',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example:
                python patch_review_no_display.py \\
                    --data /path/to/data \\
                    --splits /path/to/splits.json \\
                    --output patches_to_remove.txt \\
                    --review-dir ./review_images \\
                    --n-timesteps 5

            The script will save each patch image to the review directory.
            You can view it and then type your decision (k/r/s/q).
            Progress is automatically saved, so you can quit and resume later.
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data folder')
    parser.add_argument('--splits', type=str, required=True,
                       help='Path to splits.json file')
    parser.add_argument('--output', type=str, default='patches_to_remove.txt',
                       help='Output file for patches to remove')
    parser.add_argument('--review-dir', type=str, default='patch_review_images',
                       help='Directory to save review images')
    parser.add_argument('--n-timesteps', type=int, default=5,
                       help='Number of timesteps to display (default: 5)')
    
    args = parser.parse_args()
    
    patches_to_remove = interactive_patch_review(
        data_folder=args.data,
        splits_path=args.splits,
        output_file=args.output,
        review_dir=args.review_dir,
        n_timesteps=args.n_timesteps
    )