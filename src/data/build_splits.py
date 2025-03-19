import logging
import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import re
import xarray as xr
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NetCDFDataset(Dataset):
    """
    Loads each NetCDF file via xarray and returns the raw data as a NumPy array.
    We assume each band is an individual data variable and we want to exclude
    the MASK, SCL, and 'spatial_ref' variables.

    Additionally, we replace any -9999.0 fill values with np.nan.
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
    batch = list(map(torch.Tensor, batch))
    batch = torch.concat(batch, dim=0)
    return batch

def compute_mean_std(files, n_workers=4):
    """
    Two-pass global mean/std, ignoring NaNs in the data.
    1) In pass 1, we accumulate sum + count of non-NaN pixels => mean.
    2) In pass 2, we accumulate sum of squared diffs => variance => std.
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
        collate_fn=collate_fn
    )

    # Peek at first sample => shape: (C, T*H*W) or (C, H, W) ...
    first_sample = dataset[0]
    n_bands = first_sample.shape[0]

    # --- Pass 1: sum + count of non-NaN => mean
    sum_data = torch.zeros([n_bands], dtype=torch.float64)
    count_data = torch.zeros([n_bands], dtype=torch.float64)  # float for dividing

    for sample in tqdm(dataloader, desc='Pass 1 (Compute Mean)'):
        # Flatten all dims except band => (C, -1)
        sample = sample.reshape(n_bands, -1)
        # Convert to float64
        sample = sample.to(torch.float64)

        # sum of non-NaN
        sum_ = torch.nansum(sample, dim=-1)  # shape [C]
        # count of non-NaN
        count_ = torch.sum(~sample.isnan(), dim=-1).to(torch.float64)

        sum_data += sum_
        count_data += count_

    mean = sum_data / count_data

    # --- Pass 2: sum of squared diffs => var => std
    sum_sq = torch.zeros([n_bands], dtype=torch.float64)

    for sample in tqdm(dataloader, desc='Pass 2 (Compute Var)'):
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
    Check if a file meets the filtering criteria based on its metadata.
    Returns True if file should be included, False otherwise.
    """
    # Always check filename patterns first (these are fastest)
    filename = file_path.name.lower()
    
    # Check exclude patterns
    for pattern in criteria.exclude_patterns:
        if pattern.lower() in filename:
            return False, None
    
    # Check include patterns (if any specified, at least one must match)
    if criteria.include_patterns:
        if not any(pattern.lower() in filename for pattern in criteria.include_patterns):
            return False, None
    
    # Extract satellite type from filename or path
    satellite_type = None
    if "s1asc" in filename or "s1-asc" in filename:
        satellite_type = "s1asc"
    elif "s1dsc" in filename or "s1-dsc" in filename:
        satellite_type = "s1dsc"
    elif "s2" in filename:
        satellite_type = "s2"
    
    # Check if satellite type is in the allowed list
    if satellite_type not in criteria.satellites:
        return False, None
    
    # Now we need to open the file to check metadata
    try:
        with xr.open_dataset(file_path) as ds:
            metadata = {}
            
            # Add satellite type to metadata
            metadata["satellite_type"] = satellite_type
            
            # Check annotation status
            is_annotated = ds.attrs.get("annotated")
            metadata["annotated"] = is_annotated
            
            if criteria.annotated_only and not is_annotated:
                return False, metadata
            if criteria.non_annotated_only and is_annotated:
                return False, metadata
            
            # Check confidence value if present
            if is_annotated == "True":
                confidence_str = ds.attrs.get("date_confidence", "")
                if confidence_str:
                    try:
                        confidence_values = confidence_str.split(",") if "," in confidence_str else [confidence_str]
                        confidence_values = [float(c) for c in confidence_values]
                        confidence = np.mean(confidence_values)
                    except ValueError:
                        confidence = 0.0
                else:
                    confidence = 0.0
            else:
                confidence = 1.0
            metadata["confidence"] = confidence

            if criteria.min_confidence is not None and confidence < criteria.min_confidence:
                return False, metadata
            if criteria.max_confidence is not None and confidence > criteria.max_confidence:
                return False, metadata
            
            return True, metadata
            
    except Exception as e:
        logging.warning(f"Error reading file {file_path}: {e}")
        return False, None

def filter_files(files, criteria, n_workers=4):
    """
    Filter files based on the provided criteria
    Returns a tuple of (filtered_files, metadata_dict)
    """
    filtered_files = []
    metadata_dict = {}
    
    results = Parallel(n_jobs=n_workers)(
        delayed(check_file_metadata)(file_path, criteria) for file_path in tqdm(files, desc="Filtering files")
    )
    
    for file_path, (include, metadata) in zip(files, results):
        if include:
            filtered_files.append(file_path)
            metadata_dict[str(file_path)] = metadata
    
    # Handle annotated ratio if specified
    if criteria.annotated_ratio is not None and 0.0 <= criteria.annotated_ratio <= 1.0:

        annotated_files = [f for f in filtered_files if metadata_dict[str(f)]["annotated"] == "True"]
        non_annotated_files = [f for f in filtered_files if metadata_dict[str(f)]["annotated"] == "False"]
        
        logging.info(f"Before ratio adjustment: {len(annotated_files)} annotated, {len(non_annotated_files)} non-annotated")
        
        total_desired = len(filtered_files)
        ann_ratio = criteria.annotated_ratio
        non_ann_ratio = 1 - ann_ratio
        annotated_desired = int(total_desired * criteria.annotated_ratio)
        non_annotated_desired = int(total_desired * non_ann_ratio)
        
        # Adjust the number of annotated and non-annotated files to match the desired ratio
        if len(annotated_files) > annotated_desired:
            annotated_files = np.random.choice(annotated_files, annotated_desired, replace=False).tolist()
        
        if len(non_annotated_files) > non_annotated_desired:
            non_annotated_files = np.random.choice(non_annotated_files, non_annotated_desired, replace=False).tolist()
        
        filtered_files = annotated_files + non_annotated_files
        logging.info(f"After ratio adjustment: {len(annotated_files)} annotated, {len(non_annotated_files)} non-annotated")
    return filtered_files

def process_patch_ratio(patch_file):
    """
    Compute the foreground-to-background (landslide) ratio for a single patch file.
    Returns:
      {
        "file": patch_file,
        "pixel_ratio": float
      }
    """
    with xr.open_dataset(patch_file) as ds:
        mask = ds["MASK"].isel(time=0).values  # shape: (H, W) or (1, H, W)
        foreground = np.sum(mask == 1)
        total_pixels = mask.size
        ratio = foreground / total_pixels if total_pixels > 0 else 0

    return {"file": patch_file, "pixel_ratio": ratio}

def compute_coverage_classes(files, n_workers=1):
    if not files:
        return np.array([]), np.array([])

    patch_data_list = Parallel(n_jobs=n_workers)(
        delayed(process_patch_ratio)(file) for file in tqdm(files, desc="Process files")
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
    Given patch_array and coverage_array, do a single stratified train/test split.
    If coverage_array has too few samples per class, fallback to a normal split.
    
    Returns:
      train_files, test_files
    """
    if len(patch_array) == 0:
        return [], []

    stratify = coverage_array if len(np.unique(coverage_array)) > 1 else None

    idx = np.arange(len(patch_array))
    
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=stratify,
        random_state=seed
    )

    train_files = patch_array[train_idx]
    test_files = patch_array[test_idx]
    return train_files, test_files

def file_key(file_path):
    """
    Extract a common key from a file path to identify the same geographical location
    across different satellite data types.
    
    Args:
        file_path: Path object representing the file path
        
    Returns:
        A string key that identifies the geographical location
    """
    # Extract the base filename without extension
    base_name = file_path.stem
    # Remove the satellite identifier (e.g., '_s1asc', '_s1dsc', '_s2') from the base name
    key = re.sub(r'_(s1asc|s1dsc|s2)', '', base_name)
    return key

def to_rel(base_dir, paths):
    """Convert absolute paths to relative paths from base_dir"""
    return [str(p.relative_to(base_dir)) for p in paths]

def build_reference_test_split(files, n_workers=4, test_size=0.2, seed=42):
    patch_array, coverage_array = compute_coverage_classes(files, n_workers=n_workers)
    _, test_files = stratified_split(patch_array, coverage_array, test_size=test_size, seed=seed)
    test_keys  = {file_key(f) for f in test_files}
    return test_keys

def build_splits_from_test_keys(files, test_keys, val_size=0.2, seed=42): 
    test_files = []
    leftover_list = []
    for f in files:
        if file_key(f) in test_keys:
            test_files.append(f)
        else:
            leftover_list.append(f)

    n_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 20))
    leftover_array, leftover_cov = compute_coverage_classes(leftover_list, n_workers=n_workers)

    if len(leftover_array) > 0:
        train_files, val_files = stratified_split(
            leftover_array,
            leftover_cov,
            test_size=val_size,
            seed=seed
        )
    else:
        train_files, val_files = [], []
    
    return {"train": train_files, "val":   val_files, "test":  test_files}

def unify_common_test_across_all(final_splits):
    """
    Given a dictionary like:
      {
        "original_s2": {"train": [...], "val": [...], "test": [...]},
        "original_s1asc": {...},
        ...
        "refined_s1dsc": {...}
      }

    1) Convert each 'test' list to a dictionary: { file_key(...) -> original_path }
    2) Compute the intersection of keys across all combos.
    3) For each dataset key, remove any test items not in the intersection, add them back to 'train'.
    4) Return the updated final_splits.

    After this, all combos have the same test *keys*, ensuring a unified test set.
    """

    # 1) Convert 'test' list to a dict: key -> original path
    test_dicts = {}
    for combo_key, splits_dict in final_splits.items():
        test_list = splits_dict.get("test", [])
        # Build a dict { file_key: original_path }
        d = {}
        for path_str in test_list:
            p = Path(path_str)
            k = file_key(p)
            d[k] = path_str  # store original path string
        test_dicts[combo_key] = d

    # 2) Compute the intersection of keys across all combos
    all_keys = list(test_dicts.keys())
    if not all_keys:
        # no combos, just return
        return final_splits

    # Start with the first combo's set of keys
    combo_iter = iter(all_keys)
    first_combo = next(combo_iter)
    common_keys = set(test_dicts[first_combo].keys())

    for combo_key in combo_iter:
        common_keys &= set(test_dicts[combo_key].keys())

    # 3) For each dataset, remove any test items not in the intersection, move them to 'train'
    for combo_key, splits_dict in final_splits.items():
        # old test dict => {k -> original_path}
        old_test_dict = test_dicts[combo_key]
        # new_test_keys => intersection
        new_test_keys = set(old_test_dict.keys()) & common_keys
        # leftover_keys => old_test_keys - new_test_keys
        leftover_keys = set(old_test_dict.keys()) - new_test_keys

        # Build new test list from the intersection
        new_test_list = [old_test_dict[k] for k in new_test_keys]

        # Move leftover items back to train
        # First convert train list to a set for dedup
        train_set = set(splits_dict.get("train", []))
        for k in leftover_keys:
            train_set.add(old_test_dict[k])

        # Overwrite final splits
        final_splits[combo_key]["train"] = list(train_set)
        final_splits[combo_key]["test"] = new_test_list

    return final_splits

@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):

    base_dir = Path(cfg.base_dir)
    cfg_split = cfg.split_settings
    output_dir = Path(cfg_split.output_dir)    
    criteria = cfg_split.filter_criteria
    n_workers = cfg_split.n_workers
    test_size = cfg_split.test_size
    val_size = cfg_split.val_size
    seed = cfg_split.seed

    # 1) Filter files based on criteria
    satellite_files = {}
    for sat in criteria.satellites:
        logging.info(f"Filtering files for {sat.upper()}...")
        files = list((base_dir / "data" / "final" / sat).glob("*.nc"))
        satellite_files[sat] = filter_files(files, criteria, n_workers)   
        
    # 2) Use reference satellite to build common test split
    ref_files = satellite_files.get(criteria.reference_sat)
    test_keys = build_reference_test_split(ref_files, n_workers, test_size, seed)
    logging.info(f"Reference satellite: {criteria.reference_sat.upper()} Test={len(test_keys)}")

    satellite_splits = {}
    for sat in criteria.satellites:
        satellite_splits[sat] = build_splits_from_test_keys(satellite_files[sat], test_keys, val_size, seed)
        logging.info(f"{sat.upper()}: Train={len(satellite_splits[sat]['train'])} Val={len(satellite_splits[sat]['val'])} Test={len(satellite_splits[sat]['test'])}")
    
    # 3) Unify common test across all
    if criteria.unify_test:
        satellite_splits = unify_common_test_across_all(satellite_splits)

    for satellite, splits_dict in satellite_splits.items():
        logging.info(f"[FINAL SPLIT] {satellite.upper()}: Train={len(splits_dict['train'])} Val={len(splits_dict['val'])} Test={len(splits_dict['test'])}")
        
        sat_output_dir = output_dir / satellite
        sat_output_dir.mkdir(parents=True, exist_ok=True)

        splits_dict = {k: to_rel(base_dir, v) for k, v in splits_dict.items()}

        data_paths_file = sat_output_dir / "data_paths.json"
        with open(data_paths_file, "w") as f:
            f.write(json.dumps(splits_dict, indent=2))
        logging.info(f"Splits saved to {data_paths_file}")

        # Add base_dir to train files to get full paths
        splits_dict["train"] = [base_dir / Path(f) for f in splits_dict["train"]]

        mean, std = compute_mean_std(splits_dict["train"], n_workers=20)
        logging.info(f"Mean={mean}")
        logging.info(f"Std={std}")
        
        mean_std_file = sat_output_dir / "norm_data.json"
        with open(mean_std_file, "w") as f:
            f.write(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))
        logging.info(f"Mean/Std saved to {mean_std_file}")

    cfg_dict = OmegaConf.to_container(cfg_split, resolve=True)
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(cfg_dict, f, indent=2)

if __name__ == "__main__":
    main()