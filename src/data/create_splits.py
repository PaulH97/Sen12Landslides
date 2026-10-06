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


def inventory_name(attrs, file_path):
    """Inventory name from the file attributes, else from the file name."""
    for attr in ['inventory', 'region', 'location']:
        if attr in attrs:
            return str(attrs[attr]).lower()
    filename = re.sub(r"_(s1asc|s1dsc|s2)$", "", file_path.stem.lower())
    parts = filename.split('_')
    return parts[0] if parts else 'unknown'


def parse_event_dates(raw):
    """Event dates of one patch: 'a,b,c' -> {'a','b','c'}; empty / None -> set()."""
    return {p.strip() for p in str(raw or "").split(",") if p.strip() and p.strip() != "None"}


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
            metadata["inventory"] = inventory_name(ds.attrs, file_path)
            metadata["event_dates"] = sorted(parse_event_dates(ds.attrs.get("event_date")))
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
                    # 1 .. min_positive_pixel-1 px: too small to be positive, not hazard-free either
                    min_pos = criteria.get("min_positive_pixel")
                    if min_pos and 0 < pixel_count < min_pos:
                        return False, metadata
                else:
                    if criteria.min_annotated_pixel > 0:
                        return False, metadata

            return True, metadata
    except Exception as e:
        logging.warning(f"Error reading {file_path}: {e}")
        return False, None


def filter_files(files, criteria, n_workers=4):
    """Filter files based on criteria. Returns the kept files and {file: metadata} for them."""
    results = Parallel(n_jobs=n_workers)(
        delayed(check_file_metadata)(f, criteria) 
        for f in tqdm(files, desc="Filtering files")
    )
    
    filtered = [f for f, (ok, _) in zip(files, results) if ok]
    metas = {f: m for f, (ok, m) in zip(files, results) if ok}
    
    if criteria.min_annotated_pixel:
        logging.info(f"Applied min_annotated_pixel filter: {criteria.min_annotated_pixel}")
    
    return filtered, metas


def single_event_inventories(metas, min_positive_pixel):
    """{inventory: event date} for inventories whose positives all carry the same single event date."""
    dates, qualified = {}, {}
    for m in metas.values():
        if m.get("annotated_pixel_count", 0) >= max(min_positive_pixel, 1):
            dates.setdefault(m["inventory"], []).append(tuple(m["event_dates"]))
    for inv, per_patch in dates.items():
        unique = set(per_patch)
        if len(unique) == 1 and len(next(iter(unique))) == 1:
            qualified[inv] = next(iter(unique))[0]
    return qualified


def extract_inventory(file_path):
    """Extract inventory name from file path or metadata."""
    try:
        with xr.open_dataset(file_path) as ds:
            return inventory_name(ds.attrs, file_path)
    except Exception:
        return inventory_name({}, file_path)


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


def stratified_split_with_inventory(patches, coverage, inventory, test_size=0.2, val_size=0.2, seed=42,
                                     fixed_split=None):
    """Split patches into train/val/test, stratified by coverage class x inventory.

    Args:
        patches: array of patch file paths.
        coverage: array of coverage classes (0/1/2), parallel to `patches`.
        inventory: array of inventory names, parallel to `patches`.
        test_size: fraction held out for test.
        val_size: fraction of the remaining train pool held out for val.
        seed: random seed for the stratified draws.
        fixed_split: optional {patch_id: "train"|"val"|"test"}, typically loaded
            from another task's splits_aligned.json. A patch whose file_key
            appears here is assigned to that split directly and never enters
            train_test_split, so a patch present in both tasks always ends up
            in the same split. Only the remaining patches are stratified, and
            test_size/val_size then apply to that remainder -- so new patches
            still roughly follow the requested ratio instead of all landing in
            train. Pass None to draw every split from scratch.

    Returns:
        (train, val, test) arrays of patch paths.
    """
    if len(patches) == 0:
        return [], [], []

    # Log inventory distribution
    unique_inv, counts = np.unique(inventory, return_counts=True)
    logging.info("="*60)
    logging.info("INVENTORY DISTRIBUTION")
    for inv, count in zip(unique_inv, counts):
        logging.info(f"  {inv}: {count} ({count/len(inventory)*100:.1f}%)")
    logging.info("="*60)

    # Partition into patches pinned by fixed_split and the free remainder.
    # Indices here are positions into `patches`; the stratification below works
    # on positions into the free subset and is mapped back at the end.
    pinned_idx = {"train": [], "val": [], "test": []}
    free_idx = []
    if fixed_split:
        for i, p in enumerate(patches):
            split_name = fixed_split.get(file_key(p))
            if split_name in pinned_idx:
                pinned_idx[split_name].append(i)
            else:
                free_idx.append(i)
        logging.info(f"fixed_split: {sum(len(v) for v in pinned_idx.values())} patches pinned "
                     f"(train={len(pinned_idx['train'])}, val={len(pinned_idx['val'])}, "
                     f"test={len(pinned_idx['test'])}), {len(free_idx)} left to stratify")
    else:
        free_idx = list(range(len(patches)))

    free_idx_arr = np.array(free_idx, dtype=int)
    coverage_free = coverage[free_idx_arr] if len(free_idx_arr) else np.array([])
    inventory_free = inventory[free_idx_arr] if len(free_idx_arr) else np.array([])

    empty = np.array([], dtype=int)
    # Strata are built over the free patches only -- a stratum that is a
    # singleton here may well have more members among the pinned patches, but
    # those are already placed and must not influence this draw.
    combined = np.array([f"{c}_{i}" for c, i in zip(coverage_free, inventory_free)])

    if len(combined) == 0:
        train_idx_free, val_idx_free, test_idx_free = empty, empty, empty
    else:
        unique_strata, strata_counts = np.unique(combined, return_counts=True)

        # Singleton strata cannot be stratified across splits -> force to train
        rare_mask = np.isin(combined, unique_strata[strata_counts == 1])
        train_forced = np.where(rare_mask)[0]
        idx_split = np.where(~rare_mask)[0]
        stratify = combined[~rare_mask] if len(idx_split) > 0 else None

        if len(train_forced) > 0:
            logging.info(f"Forcing {len(train_forced)} singleton patches to train")

        if len(idx_split) > 0:
            try:
                train_val_idx, test_idx_free = train_test_split(
                    idx_split, test_size=test_size, stratify=stratify, random_state=seed
                )
            except ValueError as e:   # e.g. more strata than test patches
                logging.warning(f"stratified test split impossible ({e}); plain split instead")
                train_val_idx, test_idx_free = train_test_split(idx_split, test_size=test_size, random_state=seed)

            if len(train_val_idx) > 1:
                stratify_tv = np.array([f"{coverage_free[i]}_{inventory_free[i]}" for i in train_val_idx])
                # a stratum left with one member cannot be split again: train
                u_tv, c_tv = np.unique(stratify_tv, return_counts=True)
                rare_tv = np.isin(stratify_tv, u_tv[c_tv < 2])
                tv_forced, tv_idx = train_val_idx[rare_tv], train_val_idx[~rare_tv]
                if len(tv_forced) > 0:
                    logging.info(f"Forcing {len(tv_forced)} singleton patches to train (train/val split)")
                try:
                    train_idx_free, val_idx_free = train_test_split(
                        tv_idx, test_size=val_size, stratify=stratify_tv[~rare_tv], random_state=seed
                    )
                except ValueError as e:
                    logging.warning(f"stratified val split impossible ({e}); plain split instead")
                    train_idx_free, val_idx_free = (train_test_split(tv_idx, test_size=val_size, random_state=seed)
                                                    if len(tv_idx) > 1 else (tv_idx, empty))
                train_idx_free = np.concatenate([train_idx_free, tv_forced]).astype(int)
            else:
                train_idx_free, val_idx_free = train_val_idx, empty
        else:
            train_idx_free, val_idx_free, test_idx_free = empty, empty, empty

        if len(train_forced) > 0:
            train_idx_free = np.concatenate([train_idx_free, train_forced])

    # Map free-subset positions back onto `patches`, then add the pinned ones
    def _resolve(idx_free, pinned):
        mapped = free_idx_arr[idx_free.astype(int)] if len(idx_free) else empty
        return np.concatenate([mapped, np.array(pinned, dtype=int)]).astype(int)

    train_idx = _resolve(train_idx_free, pinned_idx["train"])
    val_idx = _resolve(val_idx_free, pinned_idx["val"])
    test_idx = _resolve(test_idx_free, pinned_idx["test"])

    # Log split distribution
    logging.info("SPLIT DISTRIBUTION BY INVENTORY")
    for inv in unique_inv:
        train_c = np.sum(inventory[train_idx] == inv)
        val_c = np.sum(inventory[val_idx] == inv)
        test_c = np.sum(inventory[test_idx] == inv)
        total = train_c + val_c + test_c
        if total == 0:
            logging.info(f"{inv}: no patches assigned")
            continue
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


def split_and_extract_metadata(files, test_size, val_size, seed, n_workers, force_reference='s2', strict=False,
                                fixed_split=None):
    """Split files and extract metadata in one go.

    Args:
        files: Dict of satellite files
        test_size: Test split size
        val_size: Validation split size
        seed: Random seed
        n_workers: Number of workers
        force_reference: Force specific satellite as reference (default: 's2')
        strict: If True, raise error if forced reference is not available
        fixed_split: optional {patch_id: split_name} to pin patches to an
            existing split assignment -- see stratified_split_with_inventory
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
        patches, coverage, inventory, test_size, val_size, seed, fixed_split=fixed_split
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


def create_splits_geodataframe(splits, files, metadata, event_dates=None):
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
        if event_dates is not None:
            records[-1]['event_date'] = event_dates.get(meta.get('inventory'))
    
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


def create_splits_json(splits, all_files, metadata, input_dir, output_dir, filename="splits.json", event_dates=None):
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
            if event_dates is not None:
                # hazard-free patches inherit the inventory's date
                entry['event_date'] = event_dates.get(entry['inventory'])
        
        result[split_name].append(entry)

    # Save the JSON file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved: {output_file}")


def create_norm_json(splits, input_dir, output_dir, n_workers, clip_data=True, filename="norm.json",
                     pool=None, metadata=None, min_positive_pixel=0):
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
        if pool is not None:
            # positive = >= min_positive_pixel annotated px, negative = none
            px = lambda f: metadata.get(file_key(f), {}).get('annotated_pixels', 0)
            keep = (lambda f: px(f) >= max(min_positive_pixel, 1)) if pool == "positive" else (lambda f: px(f) == 0)
            train_files = [f for f in train_files if keep(f)]
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


@hydra.main(config_path="../../configs/splits", config_name="default", version_base="1.3.2")
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

    fixed_split = None
    fixed_splits_json = cfg_split.get('fixed_splits_json', None)
    if fixed_splits_json:
        with open(fixed_splits_json) as f:
            reference = json.load(f)
        fixed_split = {entry['id']: split_name
                       for split_name, entries in reference.items()
                       for entry in entries}
        logging.info(f"Pinning splits from {fixed_splits_json} ({len(fixed_split)} patches)")

    logging.info("="*80)
    logging.info("DATA SPLITTING PIPELINE")
    logging.info("="*80)

    # STEP 1: Filter files
    logging.info("[STEP 1] Filtering files...")
    satellite_files, all_metas = {}, {}
    for sat in criteria.satellites:
        files = sorted((input_dir / sat).glob("*.nc"))
        logging.info(f"{sat.upper()}: {len(files)} files found")
        satellite_files[sat], sat_metas = filter_files(files, criteria, n_workers)
        all_metas.update(sat_metas)
        logging.info(f"{sat.upper()}: {len(satellite_files[sat])} after filtering")

    # STEP 1b: single-event inventories only
    event_dates = None
    if criteria.get("single_event_only"):
        event_dates = single_event_inventories(all_metas, criteria.get("min_positive_pixel") or 0)
        logging.info(f"Single-event inventories ({len(event_dates)}): {event_dates}")
        for sat in satellite_files:
            before = len(satellite_files[sat])
            satellite_files[sat] = [f for f in satellite_files[sat] if all_metas[f]["inventory"] in event_dates]
            logging.info(f"{sat.upper()}: {len(satellite_files[sat])} of {before} in single-event inventories")
    min_pos = criteria.get("min_positive_pixel") or 0
    norm_pools = cfg_split.get("norm_pools")   # e.g. [positive, negative]

    # STEP 2: Unaligned splits (per-satellite)
    logging.info("\n[STEP 2] Creating UNALIGNED splits...")
    unaligned_splits = {}
    metadata_dicts = {}
    
    for sat, files in satellite_files.items():
        # Dictionaries for the current satellite
        sat_files_dict = {sat: files}
     
        patch_splits, metadata = split_and_extract_metadata(sat_files_dict, test_size, val_size, seed, n_workers, force_reference=None, fixed_split=fixed_split)
        splits = map_splits_to_files(sat_files_dict, patch_splits)
        unaligned_splits[sat] = splits[sat]
        metadata_dicts[sat] = metadata
        
        # Save per-satellite outputs
        sat_dir = output_dir / sat       
        create_splits_json(splits=splits, all_files=sat_files_dict, metadata=metadata, input_dir=input_dir, output_dir=sat_dir, filename="splits.json", event_dates=event_dates)
        # Create norm json with absolute paths in splits
        for pool in (norm_pools or [None]):
            create_norm_json(splits, input_dir, sat_dir, n_workers, clip_data=clip_data,
                             filename="norm.json" if pool is None else f"norm_{pool}.json",
                             pool=pool, metadata=metadata, min_positive_pixel=min_pos)
        
        # GeoDataFrame
        if cfg_split.get('export_geodataframe', True):
            try:
                gdf = create_splits_geodataframe(splits, {sat: files}, metadata, event_dates)
                save_geodataframe(gdf, sat_dir)
            except Exception as e:
                logging.error(f"GeoDataFrame error: {e}")

    # STEP 3: Aligned splits (strict)
    if len(satellite_files) > 1:
        logging.info("\n[STEP 3] Creating ALIGNED splits (strict)...")
        
        aligned_files = align_files_strict(satellite_files)
        patch_splits, metadata = split_and_extract_metadata(aligned_files, test_size, val_size, seed, n_workers, force_reference='s2', fixed_split=fixed_split)
        aligned_splits = map_splits_to_files(aligned_files, patch_splits)
        
        # Create global files
        create_splits_json(aligned_splits,aligned_files, metadata, input_dir, output_dir, filename="splits_aligned.json", event_dates=event_dates)
        
        # Create norm json with absolute paths in splits  
        for pool in (norm_pools or [None]):
            create_norm_json(aligned_splits, input_dir, output_dir, n_workers, clip_data=clip_data,
                             filename="norm_aligned.json" if pool is None else f"norm_{pool}_aligned.json",
                             pool=pool, metadata=metadata, min_positive_pixel=min_pos)
        
        # GeoDataFrame
        if cfg_split.get('export_geodataframe', True):
            try:
                gdf = create_splits_geodataframe(aligned_splits, aligned_files, metadata, event_dates)
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