import logging, os, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import numpy as np
import re
from omegaconf import DictConfig
import hydra 
import xarray as xr
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

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

def compute_mean_std(files, n_workers=4):
    """
    Two-pass global mean/std, ignoring NaNs in the data.
    1) In pass 1, we accumulate sum + count of non-NaN pixels => mean.
    2) In pass 2, we accumulate sum of squared diffs => variance => std.
    """

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

def to_rel(base_dir, paths):
    return [str(p.relative_to(base_dir)) for p in paths]

def build_reference_test_split_exp1(files, test_size=0.2, seed=42):
    """
    1) Gather 'refined/s2' patches
    2) Coverage-based or partial-Italy approach => train/val/test
    3) Store the results in a dictionary keyed by the patch stem or unique ID
    4) Return a dictionary of sets: {"train": set_of_keys, "val": set_of_keys, "test": set_of_keys}
    """
    # 1) coverage classes for all
    n_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 20))
    patch_array, coverage_array = compute_coverage_classes(files, n_workers=n_workers)

    # 2) Identify italy vs non-italy
    italy_mask = np.array(["italy" in f.name.lower() or "italy" in f.parent.name.lower() for f in files], dtype=bool)
    italy_files = patch_array[italy_mask]
    italy_cov   = coverage_array[italy_mask]
    nonitaly_files = patch_array[~italy_mask]
    logging.info(f"[Exp1] Italy = {len(italy_files)}, Non-Italy = {len(nonitaly_files)}")

    test_count = int(test_size * len(files))
    # 3) If we have enough italy >= test_count
    if len(italy_files) >= test_count:
        ratio = test_count / len(italy_files)
        ratio = min(1.0, max(0.0, ratio))
        train_files, test_files = stratified_split(italy_files, italy_cov, test_size=ratio, seed=seed)

    logging.info(f"[Exp1] Train count = {len(train_files)+len(nonitaly_files)} Test count = {len(test_files)}")

    test_keys  = {file_key(f) for f in test_files}

    return test_keys

def build_reference_test_split(files, test_size=0.2, seed=42):

    # 1) coverage classes for all
    n_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 20))
    patch_array, coverage_array = compute_coverage_classes(files, n_workers=n_workers)

    train_files, test_files = stratified_split(patch_array, coverage_array, test_size=test_size, seed=seed)
    
    logging.info(f"[Exp] Train count= {len(train_files)} Test count = {len(test_files)}")

    test_keys  = {file_key(f) for f in test_files}

    return test_keys

def build_splits_from_test_keys(base_dir, dataset_type, satellite, test_keys, val_size=0.2, seed=42):
    """
    1) Gather patches in data/<variant>/<satellite>
    2) If a patch's key is in test_keys => test
    3) leftover => coverage-based => train vs val
    4) Return {"train": [...], "val": [...], "test": [...]}
    """    
    data_dir = base_dir / "data" / dataset_type / satellite
    all_files = sorted(data_dir.glob("*.nc"))

    # Separate test vs leftover
    test_list = []
    leftover_list = []
        
    for f in all_files:
        if file_key(f) in test_keys:
            test_list.append(f)
        else:
            leftover_list.append(f)

    # coverage-based on leftover => train vs val
    logging.info(f"[Exp] Computing coverage classes for {dataset_type}/{satellite}")
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
    
    return {
        "train": to_rel(base_dir, train_files),
        "val":   to_rel(base_dir, val_files),
        "test":  to_rel(base_dir, test_list)
    }

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

def build_splits_exp1(base_dir, exp_dir, test_size=0.2, val_size=0.2, seed=42):
    """
    1) Build reference test keys from refined/s2
    2) For each (variant, satellite) in {("original", "s2"), ("original", "s1asc"), ...,
                                         ("refined", "s2"), ("refined", "s1asc"), ...}:
       -> build_splits_from_test_keys
    3) Return a big dictionary or write them to JSON as needed
    """
    ref_patch_data_dir = base_dir / "data" / "final" / "s2"	
    ref_files = sorted(ref_patch_data_dir.glob("*.nc"))
    
    # 1) Reference test keys from refined/s2
    logging.info(f"[Exp1] Using satellite=s2_final as reference with N={len(ref_files)}")
    test_keys = build_reference_test_split(ref_files, test_size, seed)

    # 2) Build splits for each variant, satellite
    combos = [
      ("raw", "s2"),
      ("raw", "s1asc"),
      ("raw", "s1dsc"),
      ("final",  "s2"),
      ("final",  "s1asc"),
      ("final",  "s1dsc")
    ]
    
    final_splits = {}
    for (v, sat) in combos:
        splits = build_splits_from_test_keys(base_dir, v, sat, test_keys, val_size, seed)
        final_splits[f"{v}_{sat}"] = splits
    
    # 3) Unify common test across all
    final_splits = unify_common_test_across_all(final_splits)

    for combo_key, splits_dict in final_splits.items():
        dataset_type, satellite = combo_key.split("_")
        logging.info(f"[Exp1] {satellite.upper()}-{dataset_type}: Train={len(splits_dict['train'])} Val={len(splits_dict['val'])} Test={len(splits_dict['test'])}")
        
        output_dir = Path(exp_dir) / dataset_type / satellite
        output_dir.mkdir(parents=True, exist_ok=True)

        data_paths_file = output_dir / "data_paths.json"
        with open(data_paths_file, "w") as f:
            f.write(json.dumps(splits_dict, indent=2))
        logging.info(f"[Exp1] Splits saved to {data_paths_file}")

        # Add base_dir to train files to get full paths
        splits_dict["train"] = [base_dir / Path(f) for f in splits_dict["train"]]

        # 4) Compute mean/std for each split
        mean, std = compute_mean_std(splits_dict["train"], n_workers=20)
        logging.info(f"[Exp1] Mean={mean}")
        logging.info(f"[Exp1] Std={std}")
        
        mean_std_file = output_dir / "norm_data.json"
        with open(mean_std_file, "w") as f:
            f.write(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))
        logging.info(f"[Exp1] Mean/Std saved to {mean_std_file}")

def build_splits_exp2(base_dir, exp_dir, test_size=0.2, val_size=0.2, seed=42):

    ref_data_dir = base_dir / "data" / "final" / "s2"	
    ref_files = sorted(ref_data_dir.glob("*.nc"))
    
    # 1) Reference test keys from refined/s2
    logging.info(f"[Exp2] Using satellite=s2_final as reference with N={len(ref_files)}")
    test_keys = build_reference_test_split(ref_files, test_size, seed)

    # 2) Build splits for each variant, satellite 
    combos = [
      ("final",  "s2"),
      ("final",  "s1asc"),
      ("final",  "s1dsc")
    ]
    
    final_splits = {}
    for (dt, sat) in combos:
        splits = build_splits_from_test_keys(base_dir, dt, sat, test_keys, val_size, seed)
        final_splits[f"{dt}_{sat}"] = splits

    # 3) Unify common test across all
    final_splits = unify_common_test_across_all(final_splits)

    for combo_key, splits_dict in final_splits.items():
        dataset_type, satellite = combo_key.split("_")
        logging.info(f"[Exp2] {satellite.upper()}-{dataset_type}: Train={len(splits_dict['train'])} Val={len(splits_dict['val'])} Test={len(splits_dict['test'])}")

        train_files = [base_dir / Path(f) for f in splits_dict["train"]]
        mean, std = compute_mean_std(train_files, n_workers=4)
        logging.info(f"[Exp2] Mean={mean}")
        logging.info(f"[Exp2] Std={std}")

        for variant in ["dem", "no_dem"]:
            output_dir = Path(exp_dir) / variant / satellite
            output_dir.mkdir(parents=True, exist_ok=True)

            data_paths_file = output_dir / "data_paths.json"
            with open(data_paths_file, "w") as f:
                f.write(json.dumps(splits_dict, indent=2))
            logging.info(f"[Exp2] Splits saved to {data_paths_file}")

            mean_std_file = output_dir / "norm_data.json"
            with open(mean_std_file, "w") as f:
                f.write(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))
            logging.info(f"[Exp2] Mean/Std saved to {mean_std_file}")

def build_splits_exp3(base_dir, exp_dir, cluster_dict, val_size=0.2, seed=42):
    """
    Build splits for experiment 3 by ensuring no overlap between train and test files.
    Each region in the cluster_dict defines the test set, and the remaining files are used for training/validation.
    """
    patch_dir = base_dir / "data" / "final" / "s2"
    files = sorted(patch_dir.glob("*.nc"))

    for region, locations in cluster_dict.items():
        logging.info(f"[Exp3] Using satellite=s2_final, variant={region}, N={len(files)}")

        # Explicitly separate train and test files
        test_files = [f for f in files if any(loc.lower() in f.stem.lower() for loc in locations)]
        train_files = [f for f in files if f not in test_files]

        logging.info(f"[Exp3] Train count = {len(train_files)} Test count = {len(test_files)}")

        # Ensure no overlap between train and test files
        assert not set(train_files) & set(test_files), "Overlap detected between train and test files!"

        # Compute coverage classes for train files
        n_workers = int(os.getenv('SLURM_CPUS_PER_TASK', 20))
        patch_array, coverage_array = compute_coverage_classes(train_files, n_workers=n_workers)

        # Stratified split for train and validation sets
        if len(patch_array) > 0:
            train_files, val_files = stratified_split(patch_array, coverage_array, test_size=val_size, seed=seed)
        else:
            train_files, val_files = [], []

        splits_dict = {
            "train": to_rel(base_dir, train_files),
            "val": to_rel(base_dir, val_files),
            "test": to_rel(base_dir, test_files)
        }

        logging.info(f"[Exp3] Train={len(train_files)} Val={len(val_files)} Test={len(test_files)}")

        # Save splits to output directory
        output_dir = Path(exp_dir).parents[1] / region / "s2"
        output_dir.mkdir(parents=True, exist_ok=True)

        data_paths_file = output_dir / "data_paths.json"
        with open(data_paths_file, "w") as f:
            f.write(json.dumps(splits_dict, indent=2))
        logging.info(f"[Exp3] Splits saved to {data_paths_file}")

        # Compute mean and std for the training set
        train_files_full_path = [base_dir / Path(f) for f in splits_dict["train"]]
        if train_files_full_path:
            mean, std = compute_mean_std(train_files_full_path, n_workers=4)
            logging.info(f"[Exp3] Mean={mean} Std={std}")

            mean_std_file = output_dir / "norm_data.json"
            with open(mean_std_file, "w") as f:
                f.write(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))
            logging.info(f"[Exp3] Mean/Std saved to {mean_std_file}")
        else:
            logging.warning(f"[Exp3] No training files available for region {region}, skipping mean/std computation.")
            
@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.3.2")
def main(cfg: DictConfig):

    base_dir = Path(cfg.base_dir)
    exp_dir = Path(cfg.exp_dir)
    seed = 42

    np.random.seed(seed)

    if cfg.experiment.name == "exp1":
        test_size = 0.3 # will be reduce by unifying test set
        val_size = 0.2
        build_splits_exp1(base_dir, exp_dir, test_size, val_size, seed)

    elif cfg.experiment.name == "exp2":
        test_size, val_size = 0.2, 0.2
        build_splits_exp2(base_dir, exp_dir, test_size, val_size, seed)

    elif cfg.experiment.name == "exp3":
        test_size, val_size = 0.2, 0.2 
        cluster_dict = {
            "america": ["USA_Alaska", "USA_PuertoRico", "DominicaMaria"],
            "europe": ["Italy"],
            "africa": ["Chimanimani"],
            "central_asia": ["China", "Hokkaido", "Hiroshima", "Kyrgyzstan1", "Kyrgyzstan2"],
            "southeast_asia": ["Itogon", "LanaoDelNorte", "Indonesia", "Thrissur", "Nepal"],
            "oceania": ["Newzealand", "PNG"]
        }
        build_splits_exp3(base_dir, exp_dir, cluster_dict, val_size, seed)

    else:
        raise ValueError(f"Unknown experiment name: {cfg.experiment.name}")
   
if __name__ == "__main__":
    main()

# python create_splits.py experiment=exp1
# python create_splits.py experiment=exp2
# python create_splits.py experiment=exp3