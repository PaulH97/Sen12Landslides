import json
import logging
import re
from pathlib import Path
import pandas as pd
import hydra
import numpy as np
import torch
import xarray as xr
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point


class NetCDFDataset(Dataset):
    """Dataset for reading and stacking variables from NetCDF files."""
    
    def __init__(self, files, exclude_vars=("MASK", "SCL", "spatial_ref"), clip_data=True):
        self.files = files
        self.exclude_vars = exclude_vars
        self.clip_ranges = {'s1': (-50, 10), 's2': (0, 10000), 'dem': (0, 8800)}
        self.clip_data = clip_data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with xr.open_dataset(self.files[idx]) as ds:
            # Determine satellite type from filename
            filename = self.files[idx].name.lower()
            if "s1asc" in filename or "s1dsc" in filename:
                sat_type = 's1'
            elif "s2" in filename:
                sat_type = 's2'
            else:
                sat_type = None
            
            # Stack data with per-band clipping
            data_arrays = []
            for var in ds.data_vars:
                if var not in self.exclude_vars:
                    data = ds[var].values
                    
                    # Apply clipping if enabled
                    if self.clip_data:
                        var_lower = var.lower()
                        if 'dem' in var_lower:
                            clip_min, clip_max = self.clip_ranges['dem']
                            data = np.clip(data, clip_min, clip_max)
                        elif sat_type in self.clip_ranges:
                            clip_min, clip_max = self.clip_ranges[sat_type]
                            data = np.clip(data, clip_min, clip_max)
                    
                    data_arrays.append(data)
        
        return np.stack(data_arrays, axis=0)


def collate_fn(batch):
    """Collate function for DataLoader."""
    return torch.concat([torch.Tensor(b) for b in batch], dim=0)


def compute_mean_std(files, n_workers=4, clip_data=True):
    """Compute per-band mean and std for NetCDF files."""
    if not files:
        return torch.zeros(0), torch.zeros(0)

    dataset = NetCDFDataset(files, clip_data=clip_data)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=n_workers, drop_last=False, shuffle=False, collate_fn=collate_fn)
    
    n_bands = dataset[0].shape[0]
    sum_data = torch.zeros([n_bands], dtype=torch.float64)
    count_data = torch.zeros([n_bands], dtype=torch.float64)

    for sample in tqdm(dataloader, desc="Computing mean"):
        sample = sample.reshape(n_bands, -1).double()
        sum_data += torch.nansum(sample, dim=-1)
        count_data += torch.sum(~sample.isnan(), dim=-1).double()

    mean = sum_data / count_data
    sum_sq = torch.zeros([n_bands], dtype=torch.float64)

    for sample in tqdm(dataloader, desc="Computing variance"):
        sample = sample.reshape(n_bands, -1).double()
        sum_sq += torch.nansum((sample - mean.unsqueeze(-1)) ** 2, dim=-1)

    std = torch.sqrt(sum_sq / count_data)
    return mean, std


def check_file_metadata(file_path, criteria):
    """Check if file meets filtering criteria."""
    filename = file_path.name.lower()

    # Check patterns
    if any(p.lower() in filename for p in criteria.exclude_patterns):
        return False, None
    if criteria.include_patterns and not any(p.lower() in filename for p in criteria.include_patterns):
        return False, None
    
    # Extract satellite type
    if "s1asc" in filename:
        sat_type = "s1asc"
    elif "s1dsc" in filename:
        sat_type = "s1dsc"
    elif "s2" in filename:
        sat_type = "s2"
    else:
        return False, None
    
    if sat_type not in criteria.satellites:
        return False, None

    try:
        with xr.open_dataset(file_path) as ds:
            metadata = {"satellite_type": sat_type}
            is_annotated = ds.attrs.get("annotated")
            metadata["annotated"] = is_annotated

            # Annotation filters
            if criteria.annotated_only and is_annotated != "True":
                return False, metadata
            if criteria.non_annotated_only and is_annotated == "True":
                return False, metadata

            # Confidence filter
            if is_annotated == "True":
                conf_str = ds.attrs.get("date_confidence", "")
                if conf_str:
                    conf_vals = [float(c) for c in conf_str.split(",") if c]
                    confidence = np.mean(conf_vals) if conf_vals else 0.0
                else:
                    confidence = 0.0
            else:
                confidence = 1.0
            
            metadata["confidence"] = confidence
            if criteria.min_confidence and confidence < criteria.min_confidence:
                return False, metadata
            if criteria.max_confidence and confidence > criteria.max_confidence:
                return False, metadata

            # Min annotated pixels filter
            if criteria.min_annotated_pixel is not None:
                if "MASK" in ds.data_vars:
                    mask = ds["MASK"].values
                    if mask.ndim == 3:
                        mask = mask[0]
                    pixel_count = int(np.sum(mask == 1))
                    metadata["annotated_pixel_count"] = pixel_count
                    if pixel_count < criteria.min_annotated_pixel:
                        return False, metadata
                else:
                    if criteria.min_annotated_pixel > 0:
                        return False, metadata

            return True, metadata
    except Exception as e:
        logging.warning(f"Error reading {file_path}: {e}")
        return False, None


def filter_files(files, criteria, n_workers=4):
    """Filter files based on criteria."""
    results = Parallel(n_jobs=n_workers)(
        delayed(check_file_metadata)(f, criteria) 
        for f in tqdm(files, desc="Filtering files")
    )
    
    filtered = [f for f, (ok, _) in zip(files, results) if ok]
    
    if criteria.min_annotated_pixel:
        logging.info(f"Applied min_annotated_pixel filter: {criteria.min_annotated_pixel}")
    
    return filtered


def extract_inventory(file_path):
    """Extract inventory name from file path or metadata."""
    try:
        with xr.open_dataset(file_path) as ds:
            for attr in ['inventory', 'region', 'location']:
                if attr in ds.attrs:
                    return str(ds.attrs[attr]).lower()
    except:
        pass
    
    # Parse from filename
    filename = re.sub(r"_(s1asc|s1dsc|s2)$", "", file_path.stem.lower())
    parts = filename.split('_')
    return parts[0] if parts else 'unknown'


def process_patch_ratio(patch_file):
    """Extract patch metadata including ratio, inventory, pixels, and coordinates."""
    with xr.open_dataset(patch_file) as ds:
        mask = ds["MASK"].isel(time=0).values
        foreground = np.sum(mask == 1)
        ratio = foreground / mask.size if mask.size > 0 else 0
        
        # Extract coordinates
        if 'center_lon' in ds.attrs and 'center_lat' in ds.attrs:
            lon, lat = float(ds.attrs['center_lon']), float(ds.attrs['center_lat'])
            crs = ds.attrs.get('crs', 'EPSG:4326')
        elif 'x' in ds.coords and 'y' in ds.coords:
            lon, lat = float(ds.x.values.mean()), float(ds.y.values.mean())
            crs = ds.spatial_ref.attrs.get('crs_wkt', 'EPSG:4326') if 'spatial_ref' in ds.variables else 'EPSG:4326'
        elif 'lon' in ds.coords and 'lat' in ds.coords:
            lon, lat = float(ds.lon.values.mean()), float(ds.lat.values.mean())
            crs = 'EPSG:4326'
        else:
            lon, lat, crs = 0.0, 0.0, 'EPSG:4326'

    return {
        "file": patch_file,
        "pixel_ratio": ratio,
        "inventory": extract_inventory(patch_file),
        "annotated_pixels": int(foreground),
        "lon": lon,
        "lat": lat,
        "crs": crs
    }


def compute_coverage_classes(files, n_workers=1):
    """Compute coverage classes and extract metadata."""
    if not files:
        return np.array([]), np.array([]), np.array([]), {}

    patch_data = Parallel(n_jobs=n_workers)(
        delayed(process_patch_ratio)(f) for f in tqdm(files, desc="Computing coverage")
    )
    
    ratios = [p["pixel_ratio"] for p in patch_data]
    inventories = [p["inventory"] for p in patch_data]
    
    # Build metadata dict
    metadata_dict = {}
    for p in patch_data:
        patch_id = file_key(p["file"])
        metadata_dict[patch_id] = {
            'inventory': p['inventory'],
            'annotated_pixels': p['annotated_pixels'],
            'lon': p['lon'],
            'lat': p['lat'],
            'crs': p['crs']
        }

    # Compute coverage classes
    if ratios:
        p25, p75 = np.percentile(ratios, [25, 75])
        classes = [0 if r <= p25 else 1 if r <= p75 else 2 for r in ratios]
    else:
        classes = []

    return (np.array([p["file"] for p in patch_data]), 
            np.array(classes), 
            np.array(inventories), 
            metadata_dict)


def stratified_split_with_inventory(patches, coverage, inventory, test_size=0.2, val_size=0.2, seed=42):
    """Split patches with inventory and coverage stratification."""
    if len(patches) == 0:
        return [], [], []

    # Log inventory distribution
    unique_inv, counts = np.unique(inventory, return_counts=True)
    logging.info("="*60)
    logging.info("INVENTORY DISTRIBUTION")
    for inv, count in zip(unique_inv, counts):
        logging.info(f"  {inv}: {count} ({count/len(inventory)*100:.1f}%)")
    logging.info("="*60)
    
    # Create combined strata
    combined = np.array([f"{c}_{i}" for c, i in zip(coverage, inventory)])
    unique_strata, strata_counts = np.unique(combined, return_counts=True)
    
    # Handle singletons
    rare_mask = np.isin(combined, unique_strata[strata_counts == 1])
    train_forced = np.where(rare_mask)[0]
    idx_split = np.where(~rare_mask)[0]
    stratify = combined[~rare_mask] if len(idx_split) > 0 else None
    
    if len(train_forced) > 0:
        logging.info(f"Forcing {len(train_forced)} singleton patches to train")
    
    # Split
    if len(idx_split) > 0:
        train_val_idx, test_idx = train_test_split(
            idx_split, test_size=test_size, stratify=stratify, random_state=seed
        )
        
        if len(train_val_idx) > 1:
            stratify_tv = np.array([f"{coverage[i]}_{inventory[i]}" for i in train_val_idx])
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_size, stratify=stratify_tv, random_state=seed
            )
        else:
            train_idx, val_idx = train_val_idx, np.array([], dtype=int)
    else:
        train_idx, val_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    
    train_idx = np.concatenate([train_idx, train_forced]) if len(train_forced) > 0 else train_idx
    
    # Log split distribution
    logging.info("SPLIT DISTRIBUTION BY INVENTORY")
    for inv in unique_inv:
        train_c = np.sum(inventory[train_idx] == inv)
        val_c = np.sum(inventory[val_idx] == inv)
        test_c = np.sum(inventory[test_idx] == inv)
        total = train_c + val_c + test_c
        logging.info(f"{inv}: Train={train_c}/{total} ({train_c/total*100:.1f}%), "
                    f"Val={val_c}/{total} ({val_c/total*100:.1f}%), "
                    f"Test={test_c}/{total} ({test_c/total*100:.1f}%)")
    logging.info("="*60)
    
    return patches[train_idx], patches[val_idx], patches[test_idx]


def file_key(file_path):
    """Extract patch key by removing satellite identifier."""
    return re.sub(r"_(s1asc|s1dsc|s2)", "", file_path.stem)


def to_rel(base_dir, paths):
    """Convert absolute paths to relative paths."""
    return [str(p.relative_to(base_dir)) for p in paths]


def align_files_strict(satellite_files):
    """Keep only locations present in ALL satellites (strict alignment)."""
    logging.info("="*60)
    logging.info("ALIGNING FILES: STRICT (ALL SATELLITES REQUIRED)")
    logging.info("="*60)
    
    # Get keys for each satellite
    sat_keys = {sat: {file_key(f) for f in files} for sat, files in satellite_files.items()}
    
    for sat, keys in sat_keys.items():
        logging.info(f"{sat.upper()}: {len(keys)} unique locations")
    
    # Intersection of all satellites
    common = set.intersection(*sat_keys.values())
    logging.info(f"Common locations: {len(common)}")
    
    # Filter files
    aligned = {}
    for sat, files in satellite_files.items():
        aligned[sat] = sorted([f for f in files if file_key(f) in common])
        removed = len(files) - len(aligned[sat])
        if removed > 0:
            logging.info(f"{sat.upper()}: Removed {removed} ({removed/len(files)*100:.1f}%)")
    
    logging.info("="*60)
    return aligned


def split_and_extract_metadata(files, test_size, val_size, seed, n_workers, force_reference='s2', strict=False):
    """Split files and extract metadata in one go.
    
    Args:
        files: Dict of satellite files
        test_size: Test split size
        val_size: Validation split size
        seed: Random seed
        n_workers: Number of workers
        force_reference: Force specific satellite as reference (default: 's2')
        strict: If True, raise error if forced reference is not available
    """
    # Determine reference satellite
    if force_reference and force_reference in files:
        ref_sat = force_reference
        logging.info(f"Using forced reference satellite: {ref_sat.upper()}")
    else:
        # Handle case where forced reference is not available
        if force_reference and force_reference not in files:
            if strict:
                raise ValueError(f"Forced reference satellite '{force_reference}' not found in available satellites: {list(files.keys())}")
            else:
                # Fallback to satellite with most files
                ref_sat = max(files.keys(), key=lambda k: len(files[k]))
                logging.warning(f"Forced reference '{force_reference}' not found, using {ref_sat.upper()} instead")
        else:
            # No force_reference specified, use satellite with most files
            ref_sat = max(files.keys(), key=lambda k: len(files[k]))
    
    ref_files = files[ref_sat]
    logging.info(f"Using {ref_sat.upper()} as reference: {len(ref_files)} files")
    
    # Compute coverage and metadata
    patches, coverage, inventory, metadata = compute_coverage_classes(ref_files, n_workers)
    
    # Split
    train, val, test = stratified_split_with_inventory(
        patches, coverage, inventory, test_size, val_size, seed
    )
    
    # Extract patch IDs
    return {
        'train': {file_key(f) for f in train},
        'val': {file_key(f) for f in val},
        'test': {file_key(f) for f in test}
    }, metadata


def map_splits_to_files(files, patch_splits):
    """Map patch ID splits to actual file paths for each satellite."""
    splits = {}
    
    for sat, file_list in files.items():
        sat_splits = {'train': [], 'val': [], 'test': []}
        
        for f in file_list:
            pid = file_key(f)
            for split_name in ['train', 'val', 'test']:
                if pid in patch_splits[split_name]:
                    sat_splits[split_name].append(f)
                    break
        
        # Sort for consistency
        splits[sat] = {k: sorted(v) for k, v in sat_splits.items()}
        
        logging.info(f"{sat.upper()}: Train={len(sat_splits['train'])}, "
                    f"Val={len(sat_splits['val'])}, Test={len(sat_splits['test'])}")
    
    return splits


def create_splits_geodataframe(splits, files, metadata):
    """Create GeoDataFrame from cached metadata."""
    ref_sat = list(splits.keys())[0]
    
    # Map patch_id to split
    patch_to_split = {}
    for split_name, file_list in splits[ref_sat].items():
        for f in file_list:
            patch_to_split[file_key(f)] = split_name
    
    # Build records
    records = []
    for f in files[ref_sat]:
        pid = file_key(f)
        meta = metadata.get(pid, {})
        records.append({
            'patch_id': pid,
            'split': patch_to_split.get(pid, 'unknown'),
            'inventory': meta.get('inventory', 'unknown'),
            'annotated_pixels': meta.get('annotated_pixels', 0),
            'lon': meta.get('lon', 0.0),
            'lat': meta.get('lat', 0.0),
            'crs_original': meta.get('crs', 'EPSG:4326')
        })
    
    df = pd.DataFrame(records)
    
    # Create geometries by CRS
    gdfs = []
    for crs, group in df.groupby('crs_original'):
        geometry = [Point(r['lon'], r['lat']) for _, r in group.iterrows()]
        gdf = gpd.GeoDataFrame(group, geometry=geometry, crs=crs)
        try:
            gdf = gdf.to_crs('EPSG:4326')
        except Exception as e:
            logging.warning(f"CRS transform failed for {crs}: {e}")
        gdfs.append(gdf)
    
    gdf = pd.concat(gdfs, ignore_index=True)
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    
    logging.info(f"GeoDataFrame: {len(gdf)} patches")
    return gdf


def save_geodataframe(gdf, output_dir, base_name='patch_locations'):
    """Save GeoDataFrame as GeoJSON."""
    if gdf is None or len(gdf) == 0:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file = output_dir / f"{base_name}.geojson"
    gdf.to_file(file, driver='GeoJSON')
    logging.info(f"Saved: {file}")


def create_splits_json(splits, all_files, metadata, input_dir, output_dir, filename="splits.json"):
    """
    Create a splits.json file with detailed patch info.
    This function is general and works for single or multiple satellites.

    Args:
        splits (dict): Dict mapping satellite to its data splits {'satellite': {'train': [...], 'val': [...], 'test': [...]}}.
        all_files (dict): Dict mapping satellite to its list of all files {'satellite': [file1, file2, ...]}.
        metadata (dict): Dict with metadata for each patch_id.
        input_dir (Path): Base directory of input files for making paths relative.
        output_dir (Path): Directory to save the output JSON file.
        filename (str): Name of the output file.
    """
    # Build patch_id -> files mapping from the 'all_files' dictionary
    all_patches = {}
    for sat, file_list in all_files.items():
        for file_path in file_list:
            patch_id = file_key(file_path)
            if patch_id not in all_patches:
                all_patches[patch_id] = {}
            all_patches[patch_id][sat] = file_path

    # Build patch_id -> split mapping from the 'splits' dictionary
    # We use the first satellite in the splits as the reference for split assignment
    ref_sat = list(splits.keys())[0]
    patch_to_split = {}
    for split_name, file_list in splits[ref_sat].items():
        for f in file_list:
            patch_to_split[file_key(f)] = split_name

    # Build the final dictionary entries
    result = {'train': [], 'val': [], 'test': []}
    for pid in sorted(patch_to_split.keys()):
        split_name = patch_to_split[pid]
        
        entry = {'id': pid}

        # Add all satellite files for this patch
        for sat in sorted(all_files.keys()):
            if pid in all_patches and sat in all_patches[pid]:
                entry[sat] = str(all_patches[pid][sat].relative_to(input_dir))

        # Add metadata if available
        if pid in metadata:
            entry['pixel_annotated'] = metadata[pid].get('annotated_pixels', 0)
            entry['inventory'] = metadata[pid].get('inventory', 'unknown')
        
        result[split_name].append(entry)

    # Save the JSON file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved: {output_file}")


def create_norm_json(splits, input_dir, output_dir, n_workers, clip_data=True, filename="norm.json"):
    """
    Create a norm.json file with normalization stats for the training set.
    This function is general and works for single or multiple satellites.

    Args:
        splits (dict): Dict with splits for each satellite {'satellite': {'train': [abs_paths], 'val': [...], 'test': [...]}}.
                      Files can be either absolute Path objects or relative string paths.
        input_dir (Path): Base directory to resolve relative paths if needed.
        output_dir (Path): Directory to save the output JSON file.
        n_workers (int): Number of workers for mean/std computation.
        filename (str): Name of the output file.
    """
    norm_data = {}

    for sat, split_dict in sorted(splits.items()):
        # Get training files
        train_files = split_dict.get('train', [])
        if not train_files:
            logging.warning(f"No training files for {sat.upper()}, skipping normalization.")
            continue

        # Convert to absolute paths if they're relative strings
        abs_train_files = []
        for f in train_files:
            if isinstance(f, str):
                abs_train_files.append(input_dir / Path(f))
            elif isinstance(f, Path):
                if f.is_absolute():
                    abs_train_files.append(f)
                else:
                    abs_train_files.append(input_dir / f)
            else:
                abs_train_files.append(Path(f))

        # Get band names from the first file
        try:
            with xr.open_dataset(abs_train_files[0]) as ds:
                bands = [v for v in ds.data_vars if v not in ("MASK", "SCL", "spatial_ref")]
        except Exception as e:
            logging.error(f"Could not read bands from {abs_train_files[0]}: {e}")
            continue

        logging.info(f"Computing normalization for {sat.upper()} using {len(abs_train_files)} training files...")
        mean_t, std_t = compute_mean_std(abs_train_files, n_workers, clip_data=clip_data)

        norm_data[sat] = {
            'mean': {b: float(mean_t[i]) for i, b in enumerate(bands)},
            'std': {b: float(std_t[i]) for i, b in enumerate(bands)}
        }
        logging.info(f"{sat.upper()}: Computed stats for {len(bands)} bands")

    # Save the JSON file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        json.dump(norm_data, f, indent=2)

    logging.info(f"Saved: {output_file}")


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3.2")
def main(cfg: DictConfig):
    cfg_split = cfg.split_settings
    input_dir = Path(cfg_split.input_dir)
    output_dir = Path(cfg_split.output_dir)
    clip_data = cfg_split.get('clip_data', True)
    criteria = cfg_split.filter_criteria
    n_workers = cfg_split.n_workers
    test_size = cfg_split.test_size
    val_size = cfg_split.val_size
    seed = cfg_split.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logging.info("="*80)
    logging.info("DATA SPLITTING PIPELINE")
    logging.info("="*80)

    # STEP 1: Filter files
    logging.info("[STEP 1] Filtering files...")
    satellite_files = {}
    for sat in criteria.satellites:
        files = sorted((input_dir / sat).glob("*.nc"))
        logging.info(f"{sat.upper()}: {len(files)} files found")
        satellite_files[sat] = filter_files(files, criteria, n_workers)
        logging.info(f"{sat.upper()}: {len(satellite_files[sat])} after filtering")

    # STEP 2: Unaligned splits (per-satellite)
    logging.info("\n[STEP 2] Creating UNALIGNED splits...")
    unaligned_splits = {}
    metadata_dicts = {}
    
    for sat, files in satellite_files.items():
        # Dictionaries for the current satellite
        sat_files_dict = {sat: files}
     
        patch_splits, metadata = split_and_extract_metadata(sat_files_dict, test_size, val_size, seed, n_workers, force_reference=None)  
        splits = map_splits_to_files(sat_files_dict, patch_splits)
        unaligned_splits[sat] = splits[sat]
        metadata_dicts[sat] = metadata
        
        # Save per-satellite outputs
        sat_dir = output_dir / sat       
        create_splits_json(splits=splits, all_files=sat_files_dict, metadata=metadata, input_dir=input_dir, output_dir=sat_dir, filename="splits.json")
        # Create norm json with absolute paths in splits
        create_norm_json(splits, input_dir, sat_dir, n_workers, clip_data=clip_data, filename="norm.json")
        
        # GeoDataFrame
        if cfg_split.get('export_geodataframe', True):
            try:
                gdf = create_splits_geodataframe(splits, {sat: files}, metadata)
                save_geodataframe(gdf, sat_dir)
            except Exception as e:
                logging.error(f"GeoDataFrame error: {e}")

    # STEP 3: Aligned splits (strict)
    if len(satellite_files) > 1:
        logging.info("\n[STEP 3] Creating ALIGNED splits (strict)...")
        
        aligned_files = align_files_strict(satellite_files)
        patch_splits, metadata = split_and_extract_metadata(aligned_files, test_size, val_size, seed, n_workers, force_reference='s2')
        aligned_splits = map_splits_to_files(aligned_files, patch_splits)
        
        # Create global files
        create_splits_json(aligned_splits,aligned_files, metadata, input_dir, output_dir, filename="splits_aligned.json")
        
        # Create norm json with absolute paths in splits  
        create_norm_json(aligned_splits, input_dir, output_dir, n_workers, clip_data=clip_data, filename="norm_aligned.json")
        
        # GeoDataFrame
        if cfg_split.get('export_geodataframe', True):
            try:
                gdf = create_splits_geodataframe(aligned_splits, aligned_files, metadata)
                save_geodataframe(gdf, output_dir, "patch_locations_aligned")
            except Exception as e:
                logging.error(f"GeoDataFrame error: {e}")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(OmegaConf.to_container(cfg_split, resolve=True), f, indent=2)
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("✓ COMPLETE")
    logging.info("="*80)
    logging.info("\nUNALIGNED (per-satellite):")
    for sat, splits in unaligned_splits.items():
        total = sum(len(splits[k]) for k in ['train', 'val', 'test'])
        logging.info(f"  {sat.upper()}: {total} patches")
    
    if len(satellite_files) > 1:
        logging.info("\nALIGNED (strict):")
        first = list(aligned_splits.keys())[0]
        total = sum(len(aligned_splits[first][k]) for k in ['train', 'val', 'test'])
        logging.info(f"  Total: {total} patches across {len(aligned_splits)} satellites")
    
    logging.info("="*80)


if __name__ == "__main__":
    main()