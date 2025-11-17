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
    """
    Dataset for reading and stacking variables from NetCDF files.

    This dataset loads NetCDF files using xarray and stacks their data variables
    into a single numpy array, excluding specified variables.

    Parameters
    ----------
    files : list
        List of file paths to NetCDF files.
    exclude_vars : tuple, optional
        Variables to exclude from the stacked output.
        Default is ("MASK", "SCL", "spatial_ref").

    Returns
    -------
    numpy.ndarray
        Stacked array of data variables with shape (C, ...) where C is the number
        of variables and ... represents the original dimensions of each variable.
        For example, (C, H, W) for 2D variables or (C, T, H, W) for 3D variables.
    """

    def __init__(self, files, exclude_vars=("MASK", "SCL", "spatial_ref")):
        self.files = files
        self.exclude_vars = exclude_vars

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with xr.open_dataset(file_path) as ds:
            data_arrays = []
            for var in ds.data_vars:
                if var not in self.exclude_vars:
                    arr = ds[var].values  # shape could be (H, W) or (T, H, W), etc.
                    data_arrays.append(arr)

        # Stack along a new band dimension => shape (C, H, W) or (C, T, H, W)
        data = np.stack(data_arrays, axis=0)
        return data


def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    
    Converts samples to tensors and concatenates along dimension 0.
    
    Parameters
    ----------
    batch : list
        List of samples to collate
        
    Returns
    -------
    torch.Tensor
        Concatenated batch tensor
    """
    batch = list(map(torch.Tensor, batch))
    batch = torch.concat(batch, dim=0)
    return batch


def compute_mean_std(files, n_workers=4):
    """
    Compute per-band mean and standard deviation for NetCDF files.
    
    Uses a two-pass algorithm to properly handle NaN values.
    
    Parameters
    ----------
    files : list
        List of NetCDF file paths
    n_workers : int, optional
        Number of worker processes, default 4
        
    Returns
    -------
    mean : torch.Tensor
        Mean for each band, shape [n_bands]
    std : torch.Tensor
        Standard deviation for each band, shape [n_bands]
    """
    if not files:
        logging.warning("No files provided to compute mean/std!")
        return torch.zeros(0), torch.zeros(0)

    dataset = NetCDFDataset(files)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=n_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Peek at first sample => shape: (C, T*H*W) or (C, H, W) ...
    first_sample = dataset[0]
    n_bands = first_sample.shape[0]

    # --- Pass 1: sum + count of non-NaN => mean
    sum_data = torch.zeros([n_bands], dtype=torch.float64)
    count_data = torch.zeros([n_bands], dtype=torch.float64)  # float for dividing

    for sample in tqdm(dataloader, desc="Pass 1 (Compute Mean)"):
        # Flatten all dims except band => (C, -1)
        sample = sample.reshape(n_bands, -1)
        sample = sample.to(torch.float64)

        sum_ = torch.nansum(sample, dim=-1)  # shape [C]
        count_ = torch.sum(~sample.isnan(), dim=-1).to(torch.float64)

        sum_data += sum_
        count_data += count_

    mean = sum_data / count_data

    # --- Pass 2: sum of squared diffs => var => std
    sum_sq = torch.zeros([n_bands], dtype=torch.float64)

    for sample in tqdm(dataloader, desc="Pass 2 (Compute Var)"):
        sample = sample.reshape(n_bands, -1).double()
        diff = sample - mean.unsqueeze(-1)
        # Square but skip NaN => we'll use nansum
        sq = diff * diff
        sq_ = torch.nansum(sq, dim=-1)  # sum of squares ignoring NaNs
        sum_sq += sq_

    variance = sum_sq / count_data
    std = torch.sqrt(variance)

    return mean, std


def check_file_metadata(file_path, criteria):
    """
    Check if a file meets specified metadata criteria.
    
    Evaluates filename patterns, satellite type, annotation status, and confidence scores.
    
    Parameters
    ----------
    file_path : Path
        Path to satellite image file
    criteria : object
        Filtering criteria with attributes: exclude_patterns, include_patterns,
        satellites, annotated_only, non_annotated_only, min_confidence, max_confidence,
        min_annotated_pixel
        
    Returns
    -------
    tuple
        (meets_criteria: bool, metadata: dict or None)
    """
    # Always check filename patterns first (these are fastest)
    filename = file_path.name.lower()

    # Check exclude patterns
    for pattern in criteria.exclude_patterns:
        if pattern.lower() in filename:
            return False, None

    # Check include patterns (if any specified, at least one must match)
    if criteria.include_patterns:
        if not any(
            pattern.lower() in filename for pattern in criteria.include_patterns
        ):
            return False, None

    # Extract satellite type from filename or path
    satellite_type = None
    if "s1asc" in filename or "s1-asc" in filename:
        satellite_type = "s1asc"
    elif "s1dsc" in filename or "s1-dsc" in filename:
        satellite_type = "s1dsc"
    elif "s2" in filename:
        satellite_type = "s2"

    if satellite_type not in criteria.satellites:
        return False, None

    try:
        with xr.open_dataset(file_path) as ds:
            metadata = {}
            metadata["satellite_type"] = satellite_type

            is_annotated = ds.attrs.get("annotated")
            metadata["annotated"] = is_annotated

            if criteria.annotated_only and is_annotated != "True":
                return False, metadata
            if criteria.non_annotated_only and is_annotated == "True":
                return False, metadata

            # Check confidence value if present
            if is_annotated == "True":
                confidence_str = ds.attrs.get("date_confidence", "")
                if confidence_str:
                    try:
                        confidence_values = (
                            confidence_str.split(",")
                            if "," in confidence_str
                            else [confidence_str]
                        )
                        confidence_values = [float(c) for c in confidence_values]
                        confidence = np.mean(confidence_values)
                    except ValueError:
                        confidence = 0.0
                else:
                    confidence = 0.0
            else:
                confidence = 1.0
            metadata["confidence"] = confidence

            if (
                criteria.min_confidence is not None
                and confidence < criteria.min_confidence
            ):
                return False, metadata
            if (
                criteria.max_confidence is not None
                and confidence > criteria.max_confidence
            ):
                return False, metadata

            # Check minimum annotated pixels
            if criteria.min_annotated_pixel is not None:
                if "MASK" in ds.data_vars:
                    mask = ds["MASK"].values
                    # Handle different time dimensions
                    if mask.ndim == 3:  # (time, H, W)
                        mask = mask[0]  # Take first timestep
                    annotated_pixel_count = int(np.sum(mask == 1))
                    metadata["annotated_pixel_count"] = annotated_pixel_count
                    
                    if annotated_pixel_count < criteria.min_annotated_pixel:
                        return False, metadata
                else:
                    # No MASK variable, treat as 0 annotated pixels
                    metadata["annotated_pixel_count"] = 0
                    if criteria.min_annotated_pixel > 0:
                        return False, metadata

            return True, metadata

    except Exception as e:
        logging.warning(f"Error reading file {file_path}: {e}")
        return False, None


def filter_files(files, criteria, n_workers=4, seed=42):
    """
    Filter files based on criteria and apply annotated/non-annotated ratio.
    
    Parameters
    ----------
    files : list
        List of file paths to filter
    criteria : object
        Filtering criteria object
    n_workers : int, optional
        Number of parallel workers, default 4
    seed : int, optional
        Random seed for reproducibility, default 42
        
    Returns
    -------
    list
        Filtered list of file paths
    """
    filtered_files = []
    metadata_dict = {}

    results = Parallel(n_jobs=n_workers)(
        delayed(check_file_metadata)(file_path, criteria)
        for file_path in tqdm(files, desc="Filtering files")
    )

    for file_path, (include, metadata) in zip(files, results):
        if include:
            filtered_files.append(file_path)
            metadata_dict[str(file_path)] = metadata

    # Log filtering summary
    if criteria.min_annotated_pixel is not None:
        logging.info(f"Applied min_annotated_pixel filter: {criteria.min_annotated_pixel} pixels")

    # Only apply ratio adjustment if we want a MIX of both types
    should_adjust_ratio = (
        criteria.annotated_ratio is not None and
        0.0 < criteria.annotated_ratio < 1.0 and
        not criteria.annotated_only and 
        not criteria.non_annotated_only
    )

    if should_adjust_ratio:
        annotated_files = [f for f in filtered_files if metadata_dict[str(f)]["annotated"] == "True"]
        non_annotated_files = [f for f in filtered_files if metadata_dict[str(f)]["annotated"] == "False"]

        logging.info(f"Before ratio adjustment: {len(annotated_files)} annotated, {len(non_annotated_files)} non-annotated")

        # Only proceed if we have BOTH types
        if len(annotated_files) > 0 and len(non_annotated_files) > 0:
            # Set seed for reproducibility
            np.random.seed(seed)
            
            total_desired = len(filtered_files)
            annotated_desired = int(total_desired * criteria.annotated_ratio)
            non_annotated_desired = total_desired - annotated_desired

            # Adjust the number of annotated and non-annotated files to match the desired ratio
            if len(annotated_files) > annotated_desired:
                annotated_files = np.random.choice(
                    annotated_files, annotated_desired, replace=False
                ).tolist()

            if len(non_annotated_files) > non_annotated_desired:
                non_annotated_files = np.random.choice(non_annotated_files, non_annotated_desired, replace=False).tolist()

            filtered_files = annotated_files + non_annotated_files
            logging.info(f"After ratio adjustment: {len(annotated_files)} annotated, {len(non_annotated_files)} non-annotated")
        else:
            logging.warning(
                f"Cannot apply annotated_ratio={criteria.annotated_ratio}: "
                f"need both annotated and non-annotated files. "
                f"Found {len(annotated_files)} annotated and {len(non_annotated_files)} non-annotated. "
                f"Keeping all {len(filtered_files)} filtered files."
            )
    elif criteria.annotated_ratio is not None:
        logging.info(
            f"Skipping ratio adjustment (annotated_only={criteria.annotated_only}, "
            f"non_annotated_only={criteria.non_annotated_only}, "
            f"ratio={criteria.annotated_ratio})"
        )
        
    return filtered_files


def process_patch_ratio(patch_file):
    """
    Calculate ratio of landslide pixels to total pixels in a patch.
    
    Parameters
    ----------
    patch_file : Path
        Path to NetCDF file containing MASK data
        
    Returns
    -------
    dict
        Keys: 'file', 'pixel_ratio'
    """
    with xr.open_dataset(patch_file) as ds:
        mask = ds["MASK"].isel(time=0).values  # shape: (H, W) or (1, H, W)
        foreground = np.sum(mask == 1)
        total_pixels = mask.size
        ratio = foreground / total_pixels if total_pixels > 0 else 0

    return {"file": patch_file, "pixel_ratio": ratio}


def compute_coverage_classes(files, n_workers=1):
    """
    Categorize files into coverage classes based on landslide pixel ratios.
    
    Classes based on 25th and 75th percentiles:
    - Class 0: ratio ≤ 25th percentile
    - Class 1: 25th < ratio ≤ 75th percentile  
    - Class 2: ratio > 75th percentile
    
    Parameters
    ----------
    files : list
        List of file paths to process
    n_workers : int, optional
        Number of parallel workers, default 1
        
    Returns
    -------
    tuple
        (patch_array: np.ndarray, coverage_array: np.ndarray)
    """
    if not files:
        return np.array([]), np.array([])

    patch_data_list = Parallel(n_jobs=n_workers)(
        delayed(process_patch_ratio)(file) for file in tqdm(files, desc="Compute coverage classes")
    )
    pixel_ratios = [pd["pixel_ratio"] for pd in patch_data_list]

    # Compute coverage classes
    if pixel_ratios:
        percentiles = np.percentile(pixel_ratios, [25, 75])
        coverage_classes = [
            0 if r <= percentiles[0] else 1 if r <= percentiles[1] else 2
            for r in pixel_ratios
        ]
    else:
        coverage_classes = []

    patch_array = np.array(files)
    coverage_array = np.array(coverage_classes)

    return patch_array, coverage_array


def stratified_split(patch_array, coverage_array, test_size=0.2, seed=42):
    """
    Split patches into train/test sets using stratified sampling.
    
    Parameters
    ----------
    patch_array : array-like
        Array of patch filenames or identifiers
    coverage_array : array-like
        Coverage values for stratification
    test_size : float, optional
        Proportion for test split, default 0.2
    seed : int, optional
        Random seed, default 42
        
    Returns
    -------
    tuple
        (train_files: array-like, test_files: array-like)
    """
    if len(patch_array) == 0:
        return [], []

    stratify = coverage_array if len(np.unique(coverage_array)) > 1 else None

    idx = np.arange(len(patch_array))

    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=stratify, random_state=seed
    )

    train_files = patch_array[train_idx]
    test_files = patch_array[test_idx]
    return train_files, test_files


def file_key(file_path):
    """
    Extract unique patch key by removing satellite identifiers.
    
    Parameters
    ----------
    file_path : Path
        Path to file
        
    Returns
    -------
    str
        Patch key without satellite suffix (e.g., 'sample_s1asc' → 'sample')
    """
    # Extract the base filename without extension
    base_name = file_path.stem
    # Remove the satellite identifier (e.g., '_s1asc', '_s1dsc', '_s2') from the base name
    key = re.sub(r"_(s1asc|s1dsc|s2)", "", base_name)
    return key


def to_rel(base_dir, paths):
    """
    Convert a list of absolute paths to paths relative to a given base directory.

    Parameters:
    -----------
    base_dir : str or Path
        The base directory to make paths relative to.
    paths : list
        A list of Path objects representing absolute paths.

    Returns:
    --------
    list of str
        A list of string paths, each relative to the base_dir.
    """
    return [str(p.relative_to(base_dir)) for p in paths]


def align_files_across_satellites(satellite_files):
    """
    Keep only file locations that exist across ALL satellites.
    
    This ensures that every patch in the dataset has data from all modalities,
    which is essential for multi-modal fusion strategies.
    
    Parameters
    ----------
    satellite_files : dict
        Dictionary mapping satellite name to list of file paths
        
    Returns
    -------
    dict
        Dictionary mapping satellite name to list of aligned file paths
    """
    if not satellite_files or len(satellite_files) == 0:
        return {}
    
    logging.info("" + "="*60)
    logging.info("ALIGNING FILES ACROSS SATELLITES")
    logging.info("="*60)
    
    # Extract file keys for each satellite
    satellite_keys = {}
    for sat, files in satellite_files.items():
        keys = {file_key(f) for f in files}
        satellite_keys[sat] = keys
        logging.info(f"{sat.upper()}: {len(keys)} unique locations")
    
    # Find intersection of keys across all satellites
    all_satellites = list(satellite_keys.keys())
    if len(all_satellites) == 0:
        return {}
    
    common_keys = satellite_keys[all_satellites[0]].copy()
    for sat in all_satellites[1:]:
        common_keys &= satellite_keys[sat]
    
    logging.info(f"Common locations across all satellites: {len(common_keys)}")
    
    # Show what was filtered out
    for sat in all_satellites:
        original = len(satellite_keys[sat])
        kept = len(common_keys)
        removed = original - kept
        if removed > 0:
            logging.warning(
                f"{sat.upper()}: Removed {removed} locations "
                f"({removed/original*100:.1f}%) not present in all satellites"
            )
    
    # Keep only files that have keys in common_keys
    aligned_files = {}
    for sat, files in satellite_files.items():
        aligned = [f for f in files if file_key(f) in common_keys]
        aligned_files[sat] = sorted(aligned)
        logging.info(f"{sat.upper()} aligned: {len(aligned)} files")
    
    logging.info("="*60 + "")
    
    return aligned_files


def verify_alignment(satellite_splits):
    """
    Verify that all satellites have the same patch locations in each split.
    
    Parameters
    ----------
    satellite_splits : dict
        Dictionary mapping satellite name to splits dict
        
    Returns
    -------
    bool
        True if alignment is correct, False otherwise
    """
    logging.info("" + "="*60)
    logging.info("VERIFYING MULTI-MODAL ALIGNMENT")
    logging.info("="*60)
    
    satellites = list(satellite_splits.keys())
    if len(satellites) < 2:
        logging.info("Only one satellite, skipping alignment check")
        return True
    
    all_aligned = True
    
    for split_name in ['train', 'val', 'test']:
        # Get keys for each satellite in this split
        split_keys = {}
        for sat in satellites:
            if split_name in satellite_splits[sat]:
                keys = {file_key(f) for f in satellite_splits[sat][split_name]}
                split_keys[sat] = keys
        
        # Check if all satellites have the same keys
        if split_keys:
            first_sat = satellites[0]
            reference_keys = split_keys[first_sat]
            
            for sat in satellites[1:]:
                if split_keys[sat] != reference_keys:
                    all_aligned = False
                    missing = reference_keys - split_keys[sat]
                    extra = split_keys[sat] - reference_keys
                    
                    logging.error(
                        f"MISALIGNMENT in {split_name.upper()}: "
                        f"{sat.upper()} vs {first_sat.upper()}"
                    )
                    if missing:
                        logging.error(f"   Missing {len(missing)} locations in {sat.upper()}")
                    if extra:
                        logging.error(f"   Extra {len(extra)} locations in {sat.upper()}")
            
            if all_aligned:
                logging.info(
                    f"✓ {split_name.upper()}: {len(reference_keys)} locations "
                    f"aligned across all {len(satellites)} satellites"
                )
    
    logging.info("="*60 + "")
    
    if all_aligned:
        logging.info("PERFECT ALIGNMENT: All splits have matching locations across satellites")
    else:
        logging.error("ALIGNMENT FAILED: Some splits have mismatched locations")
    
    return all_aligned


def extract_coordinates(file_path):
    """
    Extract center coordinates and CRS from a NetCDF file.
    
    Parameters
    ----------
    file_path : Path
        Path to NetCDF file
        
    Returns
    -------
    dict
        Keys: 'file', 'x', 'y', 'crs'
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # Try to get coordinates and CRS from attributes first (most reliable)
            if 'center_lon' in ds.attrs and 'center_lat' in ds.attrs:
                x = float(ds.attrs['center_lon'])
                y = float(ds.attrs['center_lat'])
                crs = ds.attrs.get('crs', 'EPSG:4326')
            # Fallback to coordinate arrays
            elif 'x' in ds.coords and 'y' in ds.coords:
                x = float(ds.x.values.mean())
                y = float(ds.y.values.mean())
                # Try to get CRS from spatial_ref variable
                if 'spatial_ref' in ds.variables:
                    crs = ds.spatial_ref.attrs.get('crs_wkt', 'EPSG:4326')
                else:
                    crs = 'EPSG:4326'
            elif 'lon' in ds.coords and 'lat' in ds.coords:
                x = float(ds.lon.values.mean())
                y = float(ds.lat.values.mean())
                crs = 'EPSG:4326'
            elif 'longitude' in ds.coords and 'latitude' in ds.coords:
                x = float(ds.longitude.values.mean())
                y = float(ds.latitude.values.mean())
                crs = 'EPSG:4326'
            else:
                x = 0.0
                y = 0.0
                crs = 'EPSG:4326'
            
            return {'file': file_path, 'x': x, 'y': y, 'crs': crs}
    except Exception as e:
        logging.warning(f"Could not extract coordinates from {file_path}: {e}")
        return {'file': file_path, 'x': 0.0, 'y': 0.0, 'crs': 'EPSG:4326'}


def create_splits_geodataframe(satellite_splits, satellite_files, n_workers=20):
    """
    Create a GeoDataFrame with patch locations and split assignments.
    
    Uses CRS information from NetCDF files and transforms to WGS84 for compatibility.
    Parallelized for faster processing.
    
    Parameters
    ----------
    satellite_splits : dict
        Dictionary mapping satellite name to splits dict
    satellite_files : dict
        Dictionary mapping satellite name to list of file paths
    n_workers : int, optional
        Number of parallel workers, default 20
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns: patch_id, split, x, y, crs_original, geometry (in WGS84)
    """
    logging.info("Creating GeoDataFrame for visualization...")
    
    # Use the first satellite as reference (all should be aligned)
    ref_sat = list(satellite_splits.keys())[0]
    ref_splits = satellite_splits[ref_sat]
    
    # Create mapping from path to split
    path_to_split = {}
    for split_name in ['train', 'val', 'test']:
        for file_path in ref_splits[split_name]:
            path_to_split[str(file_path)] = split_name
    
    # Get reference files
    ref_files = satellite_files[ref_sat]
    
    # Parallel coordinate extraction
    logging.info(f"Extracting coordinates from {len(ref_files)} files using {n_workers} workers...")
    coord_results = Parallel(n_jobs=n_workers)(
        delayed(extract_coordinates)(file_path) 
        for file_path in tqdm(ref_files, desc="Extracting coordinates")
    )
    
    # Build records with extracted coordinates
    records = []
    for coord_data in coord_results:
        file_path = coord_data['file']
        patch_id = file_key(file_path)
        split_name = path_to_split.get(str(file_path), 'unknown')
        
        records.append({
            'patch_id': patch_id,
            'split': split_name,
            'x': coord_data['x'],
            'y': coord_data['y'],
            'crs_original': coord_data['crs'],
        })
    
    if not records:
        logging.error("No records created for GeoDataFrame!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    logging.info("Transforming coordinates to WGS84...")
    
    # Group by CRS and create geometries
    crs_groups = df.groupby('crs_original')
    gdfs = []
    
    for crs_name, group in crs_groups:        
        # Create geometry with original CRS
        geometry = [Point(row['x'], row['y']) for _, row in group.iterrows()]
        gdf_group = gpd.GeoDataFrame(group, geometry=geometry, crs=crs_name)
        # Transform to WGS84 for consistency
        try:
            gdf_group = gdf_group.to_crs('EPSG:4326')
        except Exception as e:
            logging.warning(f"Could not transform {crs_name} to WGS84: {e}")
        
        gdfs.append(gdf_group)
    
    # Concatenate all groups
    gdf = pd.concat(gdfs, ignore_index=True)
    
    # Extract lon/lat from transformed geometry
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    
    # Log statistics
    logging.info(f"GeoDataFrame created with {len(gdf)} patches")
    logging.info(f"  Train: {len(gdf[gdf['split'] == 'train'])}")
    logging.info(f"  Val: {len(gdf[gdf['split'] == 'val'])}")
    logging.info(f"  Test: {len(gdf[gdf['split'] == 'test'])}")
    
    return gdf


def save_geodataframe(gdf, output_dir, base_name='patch_locations'):
    """
    Save GeoDataFrame in multiple formats for convenience.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to save
    output_dir : Path
        Output directory
    base_name : str
        Base name for output files
    """
    if gdf is None or len(gdf) == 0:
        logging.warning("Empty GeoDataFrame, skipping save")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as GeoJSON (best for web mapping)
    geojson_file = output_dir / f"{base_name}.geojson"
    gdf.to_file(geojson_file, driver='GeoJSON')
    logging.info(f"Saved GeoJSON: {geojson_file}")    


def split_aligned_patches(aligned_files, test_size=0.2, val_size=0.2, seed=42, n_workers=4):
    """
    Split aligned patches using geographical stratification.
    
    Since all satellites have the same patches after alignment, we can split
    by patch IDs directly without needing a reference satellite.
    
    Parameters
    ----------
    aligned_files : dict
        Dictionary mapping satellite name to list of aligned file paths
    test_size : float
        Fraction for test set
    val_size : float
        Fraction for validation set
    seed : int
        Random seed
    use_geographical : bool
        Whether to use geographical clustering
    n_geo_clusters : int
        Number of geographical clusters
    n_workers : int
        Number of parallel workers
        
    Returns
    -------
    dict
        Dictionary mapping 'train', 'val', 'test' to sets of patch IDs
    """
    # Use any satellite (they're all aligned, so all have same patches)
    ref_sat = list(aligned_files.keys())[0]
    ref_files = aligned_files[ref_sat]
    
    logging.info(f"{'='*60}")
    logging.info("SPLITTING ALIGNED PATCHES")
    logging.info(f"{'='*60}")
    logging.info(f"Total aligned patches: {len(ref_files)}")
    logging.info(f"Using {ref_sat.upper()} for split computation (all satellites are equivalent)")
    
    logging.info("Using COVERAGE-ONLY stratification")
    patch_array, strata = compute_coverage_classes(ref_files, n_workers=n_workers)
    
    # Split into train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(patch_array)),
        test_size=test_size,
        stratify=strata if len(np.unique(strata)) > 1 else None,
        random_state=seed
    )
    
    # Split train+val into train and val
    if len(np.unique(strata)) > 1:
        strata_train_val = strata[train_val_idx]
    else:
        strata_train_val = None
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=strata_train_val,
        random_state=seed
    )
    
    # Extract patch IDs (not file paths)
    train_patch_ids = {file_key(patch_array[i]) for i in train_idx}
    val_patch_ids = {file_key(patch_array[i]) for i in val_idx}
    test_patch_ids = {file_key(patch_array[i]) for i in test_idx}
    
    logging.info(f"Split by patch IDs:")
    logging.info(f"  Train: {len(train_patch_ids)} patches")
    logging.info(f"  Val:   {len(val_patch_ids)} patches")
    logging.info(f"  Test:  {len(test_patch_ids)} patches")
    logging.info(f"{'='*60}")
    
    return {
        'train': train_patch_ids,
        'val': val_patch_ids,
        'test': test_patch_ids
    }


def map_patch_ids_to_files(aligned_files, patch_id_splits):
    """
    Map patch ID splits to actual file paths for each satellite.
    
    Parameters
    ----------
    aligned_files : dict
        Dictionary mapping satellite name to list of file paths
    patch_id_splits : dict
        Dictionary mapping split name to set of patch IDs
        
    Returns
    -------
    dict
        Dictionary mapping satellite name to splits dict
    """
    satellite_splits = {}
    
    for sat, files in aligned_files.items():
        splits = {'train': [], 'val': [], 'test': []}
        
        for file_path in files:
            patch_id = file_key(file_path)
            
            if patch_id in patch_id_splits['train']:
                splits['train'].append(file_path)
            elif patch_id in patch_id_splits['val']:
                splits['val'].append(file_path)
            elif patch_id in patch_id_splits['test']:
                splits['test'].append(file_path)
        
        # Sort for consistency
        for split_name in splits:
            splits[split_name] = sorted(splits[split_name])
        
        satellite_splits[sat] = splits
        
        logging.info(
            f"{sat.upper()}: "
            f"Train={len(splits['train'])}, "
            f"Val={len(splits['val'])}, "
            f"Test={len(splits['test'])}"
        )
    
    return satellite_splits


def get_band_names(file_path, exclude_vars=("MASK", "SCL", "spatial_ref")):
    """
    Extract band/variable names from a NetCDF file.
    
    Parameters
    ----------
    file_path : Path
        Path to NetCDF file
    exclude_vars : tuple
        Variables to exclude
        
    Returns
    -------
    list
        List of variable names
    """
    try:
        with xr.open_dataset(file_path) as ds:
            return [var for var in ds.data_vars if var not in exclude_vars]
    except Exception as e:
        logging.warning(f"Error reading band names from {file_path}: {e}")
        return []


def compute_mean_std_with_bands(files, n_workers=4):
    """
    Compute per-band mean and standard deviation with band names.
    
    Parameters
    ----------
    files : list
        List of NetCDF file paths
    n_workers : int, optional
        Number of worker processes, default 4
        
    Returns
    -------
    tuple
        (mean_dict: dict, std_dict: dict, band_names: list)
    """
    if not files:
        logging.warning("No files provided to compute mean/std!")
        return {}, {}, []

    # Get band names from first file
    band_names = get_band_names(files[0])
    
    if not band_names:
        logging.warning("No bands found in files!")
        return {}, {}, []
    
    # Compute mean and std tensors
    mean_tensor, std_tensor = compute_mean_std(files, n_workers)
    
    # Convert to dictionaries with band names as keys
    mean_dict = {band: float(mean_tensor[i]) for i, band in enumerate(band_names)}
    std_dict = {band: float(std_tensor[i]) for i, band in enumerate(band_names)}
    
    return mean_dict, std_dict, band_names


def count_annotated_pixels(file_path):
    """
    Count the number of annotated (landslide) pixels in a patch.
    
    Parameters
    ----------
    file_path : Path
        Path to NetCDF file containing MASK data
        
    Returns
    -------
    int
        Number of pixels with value 1 in MASK
    """
    try:
        with xr.open_dataset(file_path) as ds:
            if "MASK" in ds.data_vars:
                mask = ds["MASK"].values
                # Handle different time dimensions
                if mask.ndim == 3:  # (time, H, W)
                    mask = mask[0]  # Take first timestep
                elif mask.ndim == 2:  # (H, W)
                    pass
                return int(np.sum(mask == 1))
            else:
                return 0
    except Exception as e:
        logging.warning(f"Error counting annotated pixels in {file_path}: {e}")
        return 0


def create_global_splits_json(satellite_splits, satellite_files, input_dir, output_dir):
    """
    Create a global splits.json with all patch information.
    
    Parameters
    ----------
    satellite_splits : dict
        Dictionary mapping satellite name to splits dict
    satellite_files : dict
        Dictionary mapping satellite name to list of file paths
    input_dir : Path
        Input directory base path
    output_dir : Path
        Output directory for saving JSON
    """
    logging.info("Creating global splits.json...")
    
    # Get all patch IDs from first satellite
    ref_sat = list(satellite_files.keys())[0]
    all_patches = {}
    
    # Build mapping from patch_id to files for each satellite
    for sat, files in satellite_files.items():
        for file_path in files:
            patch_id = file_key(file_path)
            if patch_id not in all_patches:
                all_patches[patch_id] = {}
            all_patches[patch_id][sat] = file_path
    
    # Determine split for each patch
    patch_to_split = {}
    for split_name in ['train', 'val', 'test']:
        for file_path in satellite_splits[ref_sat][split_name]:
            patch_id = file_key(file_path)
            patch_to_split[patch_id] = split_name
    
    # Create splits structure
    splits = {'train': [], 'val': [], 'test': []}
    
    for patch_id in sorted(all_patches.keys()):
        split_name = patch_to_split.get(patch_id, 'unknown')
        
        # Build entry for this patch
        entry = {'id': patch_id}
        
        # Add file paths for each satellite (as relative paths)
        for sat in sorted(satellite_files.keys()):
            if sat in all_patches[patch_id]:
                rel_path = str(all_patches[patch_id][sat].relative_to(input_dir))
                entry[sat] = rel_path
        
        # Count annotated pixels (use any satellite, they should all have same MASK)
        first_sat = list(all_patches[patch_id].keys())[0]
        pixel_count = count_annotated_pixels(all_patches[patch_id][first_sat])
        entry['pixel_annotated'] = pixel_count
        
        splits[split_name].append(entry)
    
    # Save to file
    output_file = output_dir / "splits.json"
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    logging.info(f"Global splits.json saved to {output_file}")
    logging.info(f"  Train: {len(splits['train'])} patches")
    logging.info(f"  Val: {len(splits['val'])} patches")
    logging.info(f"  Test: {len(splits['test'])} patches")


def create_global_norm_json(satellite_splits, input_dir, output_dir, n_workers=4):
    """
    Create a global norm.json with normalization statistics for all modalities.
    
    Parameters
    ----------
    satellite_splits : dict
        Dictionary mapping satellite name to splits dict
    input_dir : Path
        Input directory base path
    output_dir : Path
        Output directory for saving JSON
    n_workers : int
        Number of worker processes
    """
    logging.info("Creating global norm.json...")
    
    norm_data = {}
    
    for sat, splits_dict in sorted(satellite_splits.items()):
        logging.info(f"Computing normalization for {sat.upper()}...")
        
        # Get training files with full paths
        train_files_rel = splits_dict['train']
        train_files_full = [input_dir / Path(f) for f in train_files_rel]
        
        if not train_files_full:
            logging.warning(f"No training files for {sat}, skipping...")
            continue
        
        # Compute mean and std with band names
        mean_dict, std_dict, band_names = compute_mean_std_with_bands(
            train_files_full, n_workers
        )
        
        if not band_names:
            logging.warning(f"No bands found for {sat}, skipping...")
            continue
        
        # Store in nested structure with band names as keys
        norm_data[sat] = {
            'mean': mean_dict,
            'std': std_dict
        }
        
        logging.info(f"  {sat.upper()}: {len(band_names)} bands/channels")
        logging.info(f"    Bands: {', '.join(band_names)}")
    
    # Save to file
    output_file = output_dir / "norm.json"
    with open(output_file, 'w') as f:
        json.dump(norm_data, f, indent=2)
    
    logging.info(f"Global norm.json saved to {output_file}")
    
    # Log summary
    for sat, data in norm_data.items():
        logging.info(f"  {sat.upper()}: {len(data['mean'])} bands")


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3.2")
def main(cfg: DictConfig):

    cfg_split = cfg.split_settings
    input_dir = Path(cfg_split.input_dir)
    output_dir = Path(cfg_split.output_dir)
    criteria = cfg_split.filter_criteria
    n_workers = cfg_split.n_workers
    test_size = cfg_split.test_size
    val_size = cfg_split.val_size
    seed = cfg_split.seed
    
    # Multi-modal alignment
    align_modalities = cfg_split.get('align_modalities')

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("="*80)
    logging.info("STARTING DATA SPLITTING PIPELINE")
    logging.info("="*80)
    logging.info(f"Multi-modal alignment: {align_modalities}")
    logging.info("="*80)

    # =========================================================================
    # STEP 1: Filter files based on criteria
    # =========================================================================
    logging.info("[STEP 1] Filtering files based on criteria...")
    satellite_files = {}
    for sat in criteria.satellites:
        logging.info(f"Processing {sat.upper()}...")
        files = sorted(list((input_dir / sat).glob("*.nc")))
        logging.info(f"Found: {len(files)} files")
        
        satellite_files[sat] = filter_files(files, criteria, n_workers, seed)
        satellite_files[sat] = sorted(satellite_files[sat])
        logging.info(f"After filtering: {len(satellite_files[sat])} files")

    # =========================================================================
    # STEP 2: Align files across satellites
    # =========================================================================
    if align_modalities and len(satellite_files) > 1:
        logging.info("[STEP 2] Aligning files across satellites...")
        satellite_files = align_files_across_satellites(satellite_files)
    else:
        logging.info("[STEP 2] Skipping alignment (disabled or single satellite)")

    # =========================================================================
    # STEP 3: Split aligned patches by patch IDs
    # =========================================================================
    logging.info(f"[STEP 3] Splitting aligned patches...")
    
    if not satellite_files:
        logging.error("No files to split!")
        return
    
    patch_id_splits = split_aligned_patches(
        satellite_files,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        n_workers=n_workers
    )

    # =========================================================================
    # STEP 4: Map patch IDs to files for each satellite
    # =========================================================================
    logging.info(f"[STEP 4] Mapping patch IDs to files for each satellite...")
    satellite_splits = map_patch_ids_to_files(satellite_files, patch_id_splits)

    # =========================================================================
    # STEP 5: Verify alignment
    # =========================================================================
    if align_modalities and len(satellite_files) > 1:
        logging.info(f"[STEP 5] Verifying multi-modal alignment...")
        alignment_ok = verify_alignment(satellite_splits)
        
        if not alignment_ok:
            logging.error("WARNING: Alignment verification failed!")
            logging.error("This should not happen after proper alignment.")

    # =========================================================================
    # STEP 6: Save splits and compute statistics
    # =========================================================================
    logging.info(f"[STEP 6] Saving splits and computing normalization...")
    
    for satellite, splits_dict in satellite_splits.items():
        logging.info(f"Processing {satellite.upper()}...")
        logging.info(
            f"Final counts: "
            f"Train={len(splits_dict['train'])}, "
            f"Val={len(splits_dict['val'])}, "
            f"Test={len(splits_dict['test'])}"
        )

        sat_output_dir = output_dir / satellite
        sat_output_dir.mkdir(parents=True, exist_ok=True)

        # Create relative paths for saving
        splits_dict_rel = {k: to_rel(input_dir, v) for k, v in splits_dict.items()}

        # Save the splits
        data_paths_file = sat_output_dir / "data_paths.json"
        with open(data_paths_file, "w") as f:
            json.dump(splits_dict_rel, f, indent=2)
        logging.info(f"Splits saved to {data_paths_file}")

        # Compute normalization on training set with band names
        train_files_full = [input_dir / Path(f) for f in splits_dict_rel["train"]]
        if train_files_full:
            mean_dict, std_dict, band_names = compute_mean_std_with_bands(
                train_files_full, n_workers=n_workers
            )
            
            if band_names:
                logging.info(f"Computed normalization for {len(band_names)} bands:")
                for band in band_names:
                    logging.info(f"  {band}: mean={mean_dict[band]:.4f}, std={std_dict[band]:.4f}")

                mean_std_file = sat_output_dir / "norm_data.json"
                with open(mean_std_file, "w") as f:
                    json.dump({"mean": mean_dict, "std": std_dict}, f, indent=2)
                logging.info(f"Normalization saved to {mean_std_file}")
            else:
                logging.warning(f"No bands found for {satellite}, skipping normalization")

    if align_modalities and len(satellite_files) > 1:
        logging.info(f"[STEP 6.5] Creating global splits.json and norm.json...")
        
        # Create global splits.json
        create_global_splits_json(satellite_splits, satellite_files, input_dir, output_dir)
        
        satellite_splits_rel = {}
        for sat, splits_dict in satellite_splits.items():
            satellite_splits_rel[sat] = {k: to_rel(input_dir, v) for k, v in splits_dict.items()}
        
        create_global_norm_json(satellite_splits_rel, input_dir, output_dir, n_workers)

    # Save configuration
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(OmegaConf.to_container(cfg_split, resolve=True), f, indent=2)
    logging.info(f"  Configuration saved to {config_file}")

    # =========================================================================
    # STEP 7: Create and save GeoDataFrame
    # =========================================================================
    export_geodataframe = cfg_split.get('export_geodataframe', True)
    
    if export_geodataframe:
        try:
            logging.info(f"[STEP 7] Creating GeoDataFrame for visualization...")

            # Check if satellites have differing file counts
            sat_counts = {sat: len(files) for sat, files in satellite_files.items()}
            counts = list(sat_counts.values())
            different_counts = len(set(counts)) > 1

            if different_counts:
                logging.info("Different file counts detected across satellites. Exporting one GeoDataFrame per satellite.")
                for sat in satellite_files.keys():
                    try:
                        # Build per-satellite inputs for the helper (it uses the first sat as reference)
                        single_sat_splits = {sat: satellite_splits[sat]}
                        single_sat_files = {sat: satellite_files[sat]}

                        gdf = create_splits_geodataframe(single_sat_splits, single_sat_files, n_workers)
                        if gdf is not None:
                            base_name = f"patch_locations_{sat}"
                            save_geodataframe(gdf, output_dir, base_name=base_name)
                            logging.info(f"✓ GeoDataFrame export complete for {sat} (base_name={base_name})")
                    except Exception as e:
                        logging.error(f"Error creating/saving GeoDataFrame for {sat}: {e}")
            else:
                # All satellites have same counts -> single combined GeoDataFrame as before
                gdf = create_splits_geodataframe(satellite_splits, satellite_files)
                if gdf is not None:
                    save_geodataframe(gdf, output_dir)
                    logging.info("✓ GeoDataFrame export complete!")
                    logging.info(f"Files saved in: {output_dir}")
        except ImportError as e:
            logging.warning(f"Could not create GeoDataFrame: {e}")
        except Exception as e:
            logging.error(f"X Error creating GeoDataFrame: {e}")
    
    logging.info("" + "="*80)
    logging.info("✓ DATA SPLITTING COMPLETE")
    logging.info("="*80)
    
    # Print summary
    if satellite_splits:
        logging.info("FINAL SUMMARY:")
        first_sat = list(satellite_splits.keys())[0]
        n_train = len(satellite_splits[first_sat]['train'])
        n_val = len(satellite_splits[first_sat]['val'])
        n_test = len(satellite_splits[first_sat]['test'])
        n_total = n_train + n_val + n_test
        
        logging.info(f"Total unique locations: {n_total}")
        logging.info(f"Train: {n_train} ({n_train/n_total*100:.1f}%)")
        logging.info(f"Val:   {n_val} ({n_val/n_total*100:.1f}%)")
        logging.info(f"Test:  {n_test} ({n_test/n_total*100:.1f}%)")
        logging.info(f"Satellites: {len(satellite_splits)}")
        
        if align_modalities:
            logging.info(f"✓ All {n_total} locations have data from all {len(satellite_splits)} satellites")
            logging.info(f"✓ Multi-modal fusion is supported")

if __name__ == "__main__":
    main()