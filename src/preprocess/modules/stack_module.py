import gc
import logging
import os

import joblib
from datacube import Sentinel1, Sentinel2
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def process_s1_folder(folder):
    S1_image = Sentinel1(folder)
    S1_image.stack_bands()
    gc.collect()


def process_s2_folder(folder):
    S2_image = Sentinel2(folder)
    S2_image.stack_bands()
    gc.collect()


def stack_s1_bands(s1_folder):
    """
    Stacks Sentinel-1 bands for all subfolders within the specified S1 folder using parallel processing.
    This function processes each subfolder within the given S1 folder by applying the 'process_s1_folder'
    function in parallel. It utilizes joblib's Parallel and delayed functions to distribute the workload.
    Parameters
    ----------
    s1_folder : pathlib.Path
        Path to the directory containing Sentinel-1 data subfolders.
    Returns
    -------
    None
        Returns None if the specified folder doesn't exist or is not a directory.
    Notes
    -----
    The number of parallel workers is determined by the SLURM_NTASKS environment variable,
    defaulting to 20 if the variable is not set.
    """

    if not s1_folder.exists() or not s1_folder.is_dir():
        logging.info(f"Directory {s1_folder} does not exist or is not a directory")
        return None

    subfolders = [folder for folder in s1_folder.iterdir() if folder.is_dir()]
    n_workers = int(os.getenv("SLURM_NTASKS", 20))

    with joblib.parallel_backend("loky", n_jobs=n_workers):
        Parallel(verbose=10)(
            delayed(process_s1_folder)(folder) for folder in subfolders
        )

    logging.info(f"Stacked successfully S1 bands of folder: {s1_folder}")


def stack_s2_bands(s2_folder):
    """
    Stack Sentinel-2 bands from individual folders into a single multi-band file.
    This function processes all subfolders within an S2 directory, stacking the individual
    Sentinel-2 bands into a single multi-band GeoTIFF file for each subfolder. The processing
    is performed in parallel using joblib.
    Parameters
    ----------
    s2_folder : Path
        Path to the directory containing Sentinel-2 data organized in subfolders.
        Each subfolder should contain individual band files that will be stacked.
    Returns
    -------
    None
        The function operates in-place and does not return a value.
        If the provided path does not exist or is not a directory, a log message is generated
        and the function returns None.
    Notes
    -----
    - The function uses the number of tasks from SLURM_NTASKS environment variable, or defaults to 20
      for parallel processing.
    - The 'process_s2_folder' function is called for each subfolder in parallel.
    - Progress is logged with verbosity level 10.
    """

    if not s2_folder.exists() or not s2_folder.is_dir():
        logging.info(f"Directory {s2_folder} does not exist or is not a directory")
        return None

    subfolders = [folder for folder in s2_folder.iterdir() if folder.is_dir()]
    n_workers = int(os.getenv("SLURM_NTASKS", 20))

    with joblib.parallel_backend("loky", n_jobs=n_workers):
        Parallel(verbose=10)(
            delayed(process_s2_folder)(folder) for folder in subfolders
        )

    logging.info(f"Stacked successfully S2 bands of folder: {s2_folder}")
