import json
import logging
import os
import re
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

    This function takes a batch of samples and concatenates them along dimension 0.
    Each sample in the batch is first converted to a PyTorch Tensor before concatenation.

    Parameters
    ----------
    batch : list
        A list of samples to be collated.

    Returns
    -------
    torch.Tensor
        A tensor containing all samples concatenated along the first dimension.
    """
    batch = list(map(torch.Tensor, batch))
    batch = torch.concat(batch, dim=0)
    return batch


def compute_mean_std(files, n_workers=4):
    """
    Compute mean and standard deviation for a dataset of NetCDF files.

    This function calculates the per-band mean and standard deviation across all provided files,
    properly handling NaN values. It uses a two-pass algorithm:
    1. First pass: Calculate the sum and count of non-NaN values to compute mean
    2. Second pass: Calculate sum of squared differences from mean to compute variance and std

    Parameters
    ----------
    files : list
        List of paths to NetCDF files to process
    n_workers : int, optional
        Number of worker processes for data loading, by default 4

    Returns
    -------
    mean : torch.Tensor
        Mean value for each band, shape [n_bands]
    std : torch.Tensor
        Standard deviation for each band, shape [n_bands]

    Notes
    -----
    - All calculations are performed in float64 precision to minimize numerical errors
    - NaN values are properly handled and excluded from calculations
    - Returns empty tensors if no files are provided
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
    This function evaluates a satellite image file against a set of criteria, checking:
    1. Filename patterns (include/exclude)
    2. Satellite type (S1 ascending, S1 descending, or S2)
    3. Annotation status
    4. Confidence scores (if annotated)
    Parameters
    ----------
    file_path : Path
        Path object pointing to the satellite image file to check
    criteria : object
        Object containing filtering criteria with the following attributes:
        - exclude_patterns: list of strings that should not be in the filename
        - include_patterns: list of strings, at least one must be in the filename (if provided)
        - satellites: list of allowed satellite types ("s1asc", "s1dsc", "s2")
        - annotated_only: bool, whether to only include annotated files
        - non_annotated_only: bool, whether to only include non-annotated files
        - min_confidence: float or None, minimum confidence score threshold
        - max_confidence: float or None, maximum confidence score threshold
    Returns
    -------
    tuple
        (bool, dict or None)
        - bool: True if the file meets all criteria, False otherwise
        - dict: Metadata extracted from the file if it was successfully read, including:
            * satellite_type: str, detected satellite type
            * annotated: bool, whether the file is annotated
            * confidence: float, average confidence score (if annotated)
          Returns None if the file could not be read or failed before metadata extraction
    Notes
    -----
    The function first performs fast checks based on the filename before attempting to
    open the file, which is a more expensive operation.
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

            if criteria.annotated_only and not is_annotated:
                return False, metadata
            if criteria.non_annotated_only and is_annotated:
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

            return True, metadata

    except Exception as e:
        logging.warning(f"Error reading file {file_path}: {e}")
        return False, None


def filter_files(files, criteria, n_workers=4):
    """
    Filter a list of files based on specified criteria and handle annotated/non-annotated ratio.
    This function applies filtering criteria to a list of file paths and optionally adjusts the ratio
    between annotated and non-annotated samples in the resulting set.
    Parameters
    ----------
    files : list
        List of file paths to filter.
    files : list
        List of file paths to filter.
    criteria : object
        Object containing filtering criteria with attributes:
        - annotated_ratio: Optional float between 0.0 and 1.0 specifying desired ratio of
          annotated to non-annotated samples.
        - (other criteria attributes used by check_file_metadata)
    n_workers : int, optional
        Number of parallel workers for processing, default is 4.
    Returns
    -------
    list
        Filtered list of file paths meeting the criteria and ratio requirements.
    Notes
    -----
    The function uses parallel processing to check file metadata against the criteria.
    If annotated_ratio is specified, the function will randomly sample from annotated and
    non-annotated files to achieve the desired ratio in the final output.
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

    # Handle annotated ratio if specified
    if criteria.annotated_ratio is not None and 0.0 <= criteria.annotated_ratio <= 1.0:

        annotated_files = [
            f for f in filtered_files if metadata_dict[str(f)]["annotated"] == "True"
        ]
        non_annotated_files = [
            f for f in filtered_files if metadata_dict[str(f)]["annotated"] == "False"
        ]

        logging.info(
            f"Before ratio adjustment: {len(annotated_files)} annotated, {len(non_annotated_files)} non-annotated"
        )

        total_desired = len(filtered_files)
        ann_ratio = criteria.annotated_ratio
        non_ann_ratio = 1 - ann_ratio
        annotated_desired = int(total_desired * criteria.annotated_ratio)
        non_annotated_desired = int(total_desired * non_ann_ratio)

        # Adjust the number of annotated and non-annotated files to match the desired ratio
        if len(annotated_files) > annotated_desired:
            annotated_files = np.random.choice(
                annotated_files, annotated_desired, replace=False
            ).tolist()

        if len(non_annotated_files) > non_annotated_desired:
            non_annotated_files = np.random.choice(
                non_annotated_files, non_annotated_desired, replace=False
            ).tolist()

        filtered_files = annotated_files + non_annotated_files
        logging.info(
            f"After ratio adjustment: {len(annotated_files)} annotated, {len(non_annotated_files)} non-annotated"
        )
    return filtered_files


def process_patch_ratio(patch_file):
    """
    Calculate the ratio of foreground (landslide) pixels to total pixels in a patch.

    Parameters:
    -----------
    patch_file : str or Path
        Path to the xarray dataset file containing MASK data.

    Returns:
    --------
    dict
        Dictionary with keys:
        - 'file': The input patch file path
        - 'pixel_ratio': The ratio of foreground (mask value = 1) pixels to total pixels

    Notes:
    ------
    This function assumes the MASK variable in the dataset has a time dimension that
    can be indexed with isel(time=0) and that foreground pixels are labeled with value 1.
    If the mask is empty, the ratio will be 0.
    """
    with xr.open_dataset(patch_file) as ds:
        mask = ds["MASK"].isel(time=0).values  # shape: (H, W) or (1, H, W)
        foreground = np.sum(mask == 1)
        total_pixels = mask.size
        ratio = foreground / total_pixels if total_pixels > 0 else 0

    return {"file": patch_file, "pixel_ratio": ratio}


def compute_coverage_classes(files, n_workers=1):
    """
    Computes coverage classes for a set of files based on the pixel ratio distribution.

    This function processes each file to determine its pixel ratio, then categorizes files
    into three coverage classes based on the 25th and 75th percentiles of the distribution:
    - Class 0: ratio <= 25th percentile
    - Class 1: 25th percentile < ratio <= 75th percentile
    - Class 2: ratio > 75th percentile

    Parameters
    ----------
    files : list
        List of file paths to process
    n_workers : int, optional
        Number of parallel workers for processing, default is 1

    Returns
    -------
    tuple
        - patch_array: numpy array of file paths
        - coverage_array: numpy array of coverage classes (0, 1, or 2)

    Notes
    -----
    If files list is empty, returns empty numpy arrays
    Uses the process_patch_ratio function to calculate pixel ratios for each file
    """
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
    Split an array of patch filenames into training and test sets using stratified sampling.
    The stratification is based on the provided coverage array, ensuring that the distribution
    of coverage values is approximately the same in both the training and test sets.
    Parameters
    ----------
    patch_array : array-like
        Array of patch filenames or identifiers to be split.
    coverage_array : array-like
        Array of coverage values used for stratification. Must have the same length as patch_array.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (between 0.0 and 1.0).
    seed : int, default=42
        Random seed for reproducibility.
    Returns
    -------
    train_files : array-like
        Subset of patch_array for training.
    test_files : array-like
        Subset of patch_array for testing.
    Notes
    -----
    If the coverage_array has only one unique value, stratification is disabled,
    and a simple random split is performed instead.
    If the patch_array is empty, empty lists are returned for both train and test sets.
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
    Extracts a unique key from a file path by removing satellite identifiers from the filename.

    This function takes a file path, extracts the base filename without extension,
    and then removes any satellite identifier patterns (such as '_s1asc', '_s1dsc', or '_s2')
    to create a standardized key that can be used to match related files from different satellites.

    Args:
        file_path (Path): A pathlib.Path object representing the file path.

    Returns:
        str: A standardized key derived from the filename with satellite identifiers removed.

    Example:
        >>> file_key(Path('/path/to/sample_s1asc.tif'))
        'sample'
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


def build_reference_test_split(files, n_workers=4, test_size=0.2, seed=42):
    """
    Build a stratified test split based on landslide coverage classes.

    This function computes coverage classes for the input files and performs
    a stratified split to create a test set with balanced representation of
    different coverage classes.

    Parameters
    ----------
    files : list
        List of file paths to split.
    n_workers : int, optional
        Number of workers for parallel processing, by default 4.
    test_size : float, optional
        Fraction of the dataset to include in the test split, by default 0.2.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    set
        Set of file keys (identifiers) for the test split.
    """
    patch_array, coverage_array = compute_coverage_classes(files, n_workers=n_workers)
    _, test_files = stratified_split(
        patch_array, coverage_array, test_size=test_size, seed=seed
    )
    test_keys = {file_key(f) for f in test_files}
    return test_keys


def build_splits_from_test_keys(files, test_keys, val_size=0.2, seed=42):
    """
    Build train, validation, and test splits from a list of files, where test files are predetermined by their keys.
    This function separates files into test and non-test sets based on provided test keys,
    then performs a stratified split on the non-test files to create train and validation sets.
    Parameters
    ----------
    files : list
        List of file paths to split
    test_keys : list or set
        Collection of keys identifying files that should be in the test set
    val_size : float, default=0.2
        Proportion of non-test files to use for validation
    seed : int, default=42
        Random seed for reproducibility in the train/validation split
    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', and 'test', each containing a list of file paths
    Notes
    -----
    - Uses the `file_key` function to extract keys from file paths
    - Employs `compute_coverage_classes` to obtain coverage data for stratified splitting
    - Uses `stratified_split` to create balanced train and validation sets
    - If no files remain after removing test files, returns empty train and validation sets
    """
    test_files = []
    leftover_list = []
    for f in files:
        if file_key(f) in test_keys:
            test_files.append(f)
        else:
            leftover_list.append(f)

    n_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 20))
    leftover_array, leftover_cov = compute_coverage_classes(
        leftover_list, n_workers=n_workers
    )

    if len(leftover_array) > 0:
        train_files, val_files = stratified_split(
            leftover_array, leftover_cov, test_size=val_size, seed=seed
        )
    else:
        train_files, val_files = [], []

    return {"train": train_files, "val": val_files, "test": test_files}


def unify_common_test_across_all(final_splits):
    """
    Unify the test split across multiple dataset combinations to ensure a common test set.

    This function ensures that the same geographical locations appear in the test split
    for all dataset combinations. It identifies common samples across all combinations
    based on their file keys, keeps only the common elements in the test sets, and moves
    any test samples that are not common across all combinations back to the train split.

    Parameters
    ----------
    final_splits : dict
        A dictionary where keys are dataset combination identifiers and values are
        dictionaries with 'train' and 'test' lists containing file paths.

    Returns
    -------
    dict
        The modified splits dictionary with unified test sets across all combinations.
        Each combination will have the same locations (by file key) in their test sets.

    Notes
    -----
    - The function relies on a 'file_key' function to extract a common identifier from paths
    - If there are no combinations in final_splits, the input is returned unchanged
    - Any test samples that are not common to all combinations are moved to the training set
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
        old_test_dict = test_dicts[combo_key]
        new_test_keys = set(old_test_dict.keys()) & common_keys
        leftover_keys = set(old_test_dict.keys()) - new_test_keys

        # Build new test list from the intersection
        new_test_list = [old_test_dict[k] for k in new_test_keys]
        # Move leftover items back to train
        train_set = set(splits_dict.get("train", []))
        for k in leftover_keys:
            train_set.add(old_test_dict[k])
        final_splits[combo_key]["train"] = list(train_set)
        final_splits[combo_key]["test"] = new_test_list

    return final_splits


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

    # 1) Filter files based on criteria
    satellite_files = {}
    for sat in criteria.satellites:
        logging.info(f"Filtering files for {sat.upper()}...")
        files = list((input_dir / sat).glob("*.nc"))
        satellite_files[sat] = filter_files(files, criteria, n_workers)

    # 2) Use reference satellite to build common test split
    ref_files = satellite_files.get(criteria.reference_sat)
    test_keys = build_reference_test_split(ref_files, n_workers, test_size, seed)
    logging.info(
        f"Reference satellite: {criteria.reference_sat.upper()} Test={len(test_keys)}"
    )

    satellite_splits = {}
    for sat in criteria.satellites:
        satellite_splits[sat] = build_splits_from_test_keys(
            satellite_files[sat], test_keys, val_size, seed
        )
        logging.info(
            f"{sat.upper()}: Train={len(satellite_splits[sat]['train'])} Val={len(satellite_splits[sat]['val'])} Test={len(satellite_splits[sat]['test'])}"
        )

    # 3) Unify common test across all
    if criteria.unify_test:
        satellite_splits = unify_common_test_across_all(satellite_splits)

    for satellite, splits_dict in satellite_splits.items():
        logging.info(
            f"[FINAL SPLIT] {satellite.upper()}: Train={len(splits_dict['train'])} Val={len(splits_dict['val'])} Test={len(splits_dict['test'])}"
        )

        sat_output_dir = output_dir / satellite
        sat_output_dir.mkdir(parents=True, exist_ok=True)

        # Create a shortened version with just the filename and parent folder
        splits_dict = {k: to_rel(input_dir, v) for k, v in splits_dict.items()}

        # Save the shortened paths
        data_paths_file = sat_output_dir / "data_paths.json"
        with open(data_paths_file, "w") as f:
            f.write(json.dumps(splits_dict, indent=2))
        logging.info(f"Splits saved to {data_paths_file}")

        # Add base_dir to train files to get full paths
        splits_dict["train"] = [input_dir / Path(f) for f in splits_dict["train"]]

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
