import gc
import json
import logging
import math
import os
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from datacube import Sentinel1DataCube, Sentinel2DataCube
from joblib import Parallel, delayed, parallel_backend
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

logging.basicConfig(level=logging.INFO)


def fill_nans(arr):
    """
    Fill NaN values in numpy arrays using the nearest non-NaN value.
    This function handles both 2D arrays and N-dimensional arrays. For 2D arrays,
    it uses Euclidean distance transform to find the nearest non-NaN values.
    For arrays with more than 2 dimensions, it recursively applies the 2D method
    to each 2D slice.
    Parameters
    ----------
    arr : numpy.ndarray
        The input array containing NaN values to be filled.
    Returns
    -------
    numpy.ndarray
        A copy of the input array with NaN values replaced by their nearest
        non-NaN neighbor values.
    Notes
    -----
    This implementation uses scipy's distance_transform_edt function to efficiently
    find the indices of the nearest non-NaN values.
    """

    if arr.ndim == 2:
        mask = np.isnan(arr)
        if not mask.any():
            return arr
        indices = distance_transform_edt(
            mask, return_distances=False, return_indices=True
        )
        return arr[tuple(indices)]
    else:
        new_arr = np.empty_like(arr)
        for idx in np.ndindex(arr.shape[:-2]):
            new_arr[idx] = fill_nans(arr[idx])
        return new_arr


def compute_bad_pixel_ratio_s2(img_patch):
    """
    Compute the percentage of bad pixels in a Sentinel-2 image patch based on the Scene Classification Layer (SCL).
    The function identifies pixels belonging to any of the following classes as bad:
    - 0: No Data
    - 1: Defective
    - 2: Cloud shadows
    - 3: Cloud low
    - 8: Cloud medium
    - 9: Cloud high
    - 10: Cirrus
    - 11: Snow
    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D numpy array where the last band is the Scene Classification Layer (SCL).
        Expected shape: (bands, height, width)
    Returns
    -------
    float
        The percentage of bad pixels in the image patch.
        Returns 100 (worst quality) for empty patches or if an error occurs.
    """

    if img_patch.shape[0] < 1:
        return 100  # Worst quality

    # Assuming SCL is the last band
    scl = img_patch[-1, :, :]

    # Bad pixel classes (0=No Data, 1=Defective, 2=Cloud shadows, 3=Cloud low,
    # 8=Cloud medium, 9=Cloud high, 10=Cirrus, 11=Snow)
    bad_classes = [0, 1, 2, 3, 8, 9, 10, 11]

    # Calculate ratio
    bad_pixels = np.sum(np.isin(scl, bad_classes))
    total_pixels = scl.size

    # Return percentage of bad pixels
    return (bad_pixels / total_pixels) * 100 if total_pixels > 0 else 100


def compute_valid_pixel_ratio_s1(img_patch):
    """
    Compute the percentage of invalid pixels in a Sentinel-1 image patch.

    This function calculates the ratio of invalid pixels (NaN values)
    in a Sentinel-1 image patch to determine its quality.

    Parameters
    ----------
    img_patch : numpy.ndarray
        The Sentinel-1 image patch to evaluate.

    Returns
    -------
    float
        The percentage of invalid pixels in the image patch.
        Returns np.inf if the image patch is empty.
        Lower values indicate better quality (fewer invalid pixels).
    """

    # Check if the image patch is empty.
    if img_patch.size == 0:
        return np.inf  # Worst quality

    # In S1 imagery, -9999 or NaN values indicate invalid pixels.
    invalid_pixels = np.sum(np.isnan(img_patch))
    total_pixels = img_patch.size

    # Compute the percentage of invalid pixels.
    invalid_ratio = (invalid_pixels / total_pixels) * 100 if total_pixels > 0 else 100
    return invalid_ratio


def extract_pre_post_indices(timesteps, event_dates):
    """
    Extract the indices of pre-event and post-event timesteps based on event dates.
    This function identifies the closest timesteps before and after each event date.
    If an event date is None, or if the event date falls outside the range of
    available timesteps, special handling rules are applied.
    Parameters
    ----------
    timesteps : list or array-like
        A sequence of timesteps as datetime strings that can be parsed by pandas.to_datetime.
        Should be in chronological order.
    event_dates : list or array-like
        A sequence of event dates as datetime strings, or None values.
    Returns
    -------
    list of dict
        A list of dictionaries, one for each event date, with keys 'pre' and 'post'
        containing the indices of the pre-event and post-event timesteps.
        - If event is None: returns {"pre": 0, "post": 0}
        - If event is before first timestep: returns {"pre": 0, "post": 0}
        - If event is after last timestep: returns {"pre": last_index, "post": last_index}
        - Otherwise: returns {"pre": index_before_event, "post": index_after_event}
    """

    ts = pd.to_datetime(timesteps).values

    result = []
    for event in event_dates:
        if event is None or event == "None":
            result.append({"pre": 0, "post": 0})
            continue
        event = pd.to_datetime(event).to_numpy()
        idx = np.searchsorted(ts, event)
        if idx == 0:
            result.append({"pre": 0, "post": 0})
        elif idx == len(ts):
            result.append({"pre": int(len(ts) - 1), "post": int(len(ts) - 1)})
        else:
            result.append({"pre": int(idx - 1), "post": int(idx)})
    return result


def build_and_save_xarray_dataset(
    img_patches, msk_patches, timesteps, dem_data, config, output_file
):
    """
    Build and save an xarray dataset from image patches, mask patches, and additional data.
    This function constructs an xarray Dataset from satellite imagery and associated data,
    then saves it to a NetCDF file. The dataset includes the satellite bands, masks (if provided),
    and elevation data (if provided), along with relevant metadata.
    Parameters
    ----------
    img_patches : list
        List of numpy arrays containing satellite imagery patches.
        Each array should have shape (bands, height, width).
    msk_patches : list or None
        List of numpy arrays containing mask patches, or None if no masks.
        Each array should have shape (1, height, width) or (height, width).
    timesteps : list
        List of datetime objects representing the acquisition times of the images.
    dem_data : numpy.ndarray or None
        Digital elevation model data as a numpy array, or None if not available.
        Should have shape (height, width).
    config : dict
        Configuration dictionary containing:
        - transform: Affine transform for the raster
        - crs: Coordinate reference system
        - satellite: Satellite name (e.g., "s1-asc", "s2")
        - band_names: List of band names
        - anns_data: Dictionary of annotation data (optional)
        - annotated: Boolean indicating if the patch is annotated (optional)
    output_file : str
        Path to save the output NetCDF file.
    Returns
    -------
    None
        The function saves the dataset to disk and performs cleanup operations.
    Notes
    -----
    - Sentinel-1 (S1) data is stored as float32, while other satellite data is stored as int16.
    - DEM data is clipped to the range [0, 10000] and stored as int16.
    - The function handles memory cleanup using garbage collection after saving.
    - Annotation data, if provided, is stored in the dataset attributes.
    - Pre/post-event indices are extracted when event dates are available.
    """

    transform = config["transform"]
    crs = config["crs"]
    satellite = config["satellite"].lower()
    time_coords = np.array(timesteps, dtype="datetime64[ns]")

    patch = (
        np.stack(img_patches, axis=0)
        if len(img_patches) > 1
        else np.expand_dims(img_patches[0], axis=0)
    )
    if dem_data is not None:
        dem_data = np.squeeze(dem_data)

    _, _, ny, nx = patch.shape
    x_coords = np.arange(nx) * transform.a + transform.c
    y_coords = transform.f + np.arange(ny) * transform.e

    data_vars = {}
    for i, band in enumerate(config["band_names"]):
        band_data = patch[:, i, :, :]
        if satellite in ["s1-asc", "s1-dsc"]:
            band_data = band_data.astype(np.float32)
        else:
            band_data = band_data.astype(np.int16)
        data_vars[band] = (("time", "y", "x"), band_data)
    if msk_patches:
        mask_array = [np.nan_to_num(m.astype("uint8"), nan=0) for m in msk_patches]
        mask_array = (
            np.stack(mask_array, axis=0)
            if len(mask_array) > 1
            else np.expand_dims(mask_array[0], axis=0)
        )
        if mask_array.ndim == 4 and mask_array.shape[1] == 1:
            mask_array = mask_array[:, 0, :, :]
        data_vars["MASK"] = (("time", "y", "x"), mask_array)
    if dem_data is not None:
        dem_array = np.repeat(dem_data[np.newaxis, :, :], len(timesteps), axis=0)
        dem_array = np.clip(dem_array, 0, 10000).astype(np.int16)
        data_vars["DEM"] = (("time", "y", "x"), dem_array)

    coords = {"time": time_coords, "y": y_coords, "x": x_coords}
    ds = xr.Dataset(data_vars, coords=coords)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(transform)

    anns_data = config.get("anns_data", {})

    # Always return annotation IDs as a list of ints.
    if anns_data:
        ann_ids = [int(key) for key in anns_data.keys()]
        ann_bboxes = [anno.get("bbox", "None") for anno in anns_data.values()]
        event_dates = list(
            {anno.get("event_date", "None") for anno in anns_data.values()}
        )
        date_confidences = list(
            set(anno.get("confidence", 0.0) for anno in anns_data.values())
        )
        pre_post_idx = extract_pre_post_indices(timesteps, event_dates)
    else:
        ann_ids = []
        ann_bboxes = []
        event_dates = []
        date_confidences = []
        pre_post_idx = []

    ds.attrs["ann_id"] = ",".join(map(str, ann_ids))
    ds.attrs["ann_bbox"] = ",".join(map(str, ann_bboxes))
    ds.attrs["event_date"] = ",".join(map(str, event_dates))
    ds.attrs["date_confidence"] = ",".join(map(str, date_confidences))
    ds.attrs["pre_post_dates"] = ",".join(map(str, pre_post_idx))
    ds.attrs["annotated"] = str(config.get("annotated", "False"))
    ds.attrs["satellite"] = satellite
    ds.attrs["center_lat"] = y_coords[int(ny / 2)]
    ds.attrs["center_lon"] = x_coords[int(nx / 2)]
    ds.attrs["crs"] = crs.to_string()

    ds = ds.sortby("time")
    # Calculate center latitude and longitude
    ds.to_netcdf(output_file, mode="w")
    ds.close()
    del data_vars, coords, patch, ds
    gc.collect()


def select_images(img_patches, timesteps, ts_length, event_dates, satellite_type, seed):
    """
    Select images using an event-centric strategy that also enforces uniform distribution.

    1. Partition the full time range into ts_length equal intervals and pick the best candidate
       (lowest quality metric) from each interval.
    2. For each event date, ensure that there is at least one image immediately before and after.
       For a single event, enforce higher coverage (e.g., min_before=5, min_after=7).
    3. If too many images are selected, trim the ones with worse quality (preferring to keep event-critical images).
       If too few, fill in from the remaining candidates.
    4. If the final number of selected images is less than ts_length, and the shortage is less than 5,
       duplicate some images to reach the desired count.

    Args:
        img_patches (list): List of image patch arrays.
        timesteps (list): List of corresponding timestamps.
        ts_length (int): Desired time series length.
        event_dates (list or pd.DatetimeIndex): List of event dates (as strings or Timestamps).
        satellite_type (str): 'S2' or 'S1'.
        seed (int): Random seed.

    Returns:
        list: Final list of original indices of selected images (sorted by time),
              or None if a valid selection is not possible.
    """
    if seed is not None:
        random.seed(seed)

    # If the total number of images is less than desired and shortage is 5 or more, return None.
    if len(img_patches) < ts_length and (ts_length - len(img_patches)) >= 5:
        return None

    # Convert timesteps to pandas Timestamps and sort them.
    timestamps = pd.to_datetime(timesteps)
    sorted_order = np.argsort(timestamps)
    sorted_timestamps = timestamps[sorted_order]
    sorted_img_patches = [img_patches[i] for i in sorted_order]
    n = len(sorted_timestamps)

    # Convert and sort event_dates if provided.
    if event_dates is not None and len(event_dates) > 0:
        event_dates = pd.to_datetime(sorted(event_dates))
    else:
        event_dates = pd.DatetimeIndex([])

    # 1. Compute quality metric for each image.
    quality = {}
    for i, patch in enumerate(sorted_img_patches):
        if satellite_type.upper() == "S2":
            quality[i] = compute_bad_pixel_ratio_s2(patch)
        else:
            quality[i] = compute_valid_pixel_ratio_s1(patch)

    # For S1 asc/dsc, filter out invalid images.
    if satellite_type in ["S1-asc", "S1-dsc"]:
        valid_idx = [i for i in range(n) if quality[i] != np.inf]
        if not valid_idx:
            return None  # No valid images remain.
        sorted_timestamps = sorted_timestamps[valid_idx]
        sorted_img_patches = [sorted_img_patches[i] for i in valid_idx]
        sorted_order = sorted_order[valid_idx]
        quality = {new_i: quality[old_i] for new_i, old_i in enumerate(valid_idx)}
        n = len(sorted_timestamps)

    # 2. Partition the full time range into ts_length equal intervals and select best candidate in each.
    start_time = sorted_timestamps[0]
    end_time = sorted_timestamps[-1]
    intervals = pd.date_range(start=start_time, end=end_time, periods=ts_length + 1)
    selected = set()
    for j in range(ts_length):
        interval_start = intervals[j]
        interval_end = intervals[j + 1]
        # Get candidates in this interval.
        candidates = [
            i
            for i, t in enumerate(sorted_timestamps)
            if interval_start <= t < interval_end
        ]
        if candidates:
            best = min(candidates, key=lambda i: quality[i])
            selected.add(best)

    # 3. Ensure event coverage.
    if len(event_dates) > 0:
        for event in event_dates:
            # Ensure at least one image immediately before the event.
            before = [i for i in range(n) if sorted_timestamps[i] < event]
            if before:
                best_before = max(before, key=lambda i: sorted_timestamps[i])
                selected.add(best_before)
            # Ensure at least one image immediately after the event.
            after = [i for i in range(n) if sorted_timestamps[i] > event]
            if after:
                best_after = min(after, key=lambda i: sorted_timestamps[i])
                selected.add(best_after)

    # 4. Adjust to exactly ts_length.
    # If too few images, add from remaining candidates (best quality first).
    if len(selected) < ts_length:
        remaining = [i for i in range(n) if i not in selected]
        remaining.sort(key=lambda i: quality[i])
        for i in remaining[: ts_length - len(selected)]:
            selected.add(i)
    # If too many images, remove non-critical ones.
    elif len(selected) > ts_length:
        sel_list = list(selected)
        sel_list.sort(key=lambda i: sorted_timestamps[i])
        to_remove = len(selected) - ts_length
        # Protect images that are event-critical.
        critical = set()
        if len(event_dates) > 0:
            for event in event_dates:
                before = [i for i in sel_list if sorted_timestamps[i] < event]
                after = [i for i in sel_list if sorted_timestamps[i] > event]
                if before:
                    critical.add(max(before, key=lambda i: sorted_timestamps[i]))
                if after:
                    critical.add(min(after, key=lambda i: sorted_timestamps[i]))
        # Remove worst quality ones among non-critical.
        non_critical = [i for i in sel_list if i not in critical]
        non_critical.sort(key=lambda i: quality[i], reverse=True)
        removal = non_critical[:to_remove]
        selected = set(sel_list) - set(removal)

    # Convert back to original indices.
    original_indices = [int(sorted_order[i]) for i in selected]
    original_indices.sort(key=lambda i: pd.to_datetime(timesteps[i]))

    # If still fewer than ts_length, and shortage is less than 5, duplicate the last image.
    if len(original_indices) < ts_length:
        shortage = ts_length - len(original_indices)
        if shortage < 5:
            original_indices.extend([original_indices[-1]] * shortage)
        else:
            return None
    elif len(original_indices) > ts_length:
        original_indices = original_indices[:ts_length]

    return original_indices


def is_valid_patch(img_patch, max_nan_ratio=0.05):
    """
    Determines if a multi-band image patch is valid for further processing.
    This function checks various criteria to determine if an image patch is valid:
    1. Ensures the overall NaN ratio doesn't exceed the specified threshold
    2. For each band, checks that:
        - The band has at least one valid pixel
        - The band has meaningful variance (standard deviation above threshold)
        - The band has more than one unique value
        - All values are within an expected range [-9998, 10001]
    Parameters:
    ----------
    img_patch : numpy.ndarray
         The multi-band image patch to validate, with shape (bands, height, width)
    max_nan_ratio : float, optional
         Maximum allowed ratio of NaN values relative to total pixels (default: 0.05)
    Returns:
    -------
    bool
         True if the patch passes all validation criteria, False otherwise
    """

    # Overall NaN check is still good:
    if np.isnan(img_patch).sum() / img_patch.size > max_nan_ratio:
        return False

    # Then check each band separately:
    for b in range(img_patch.shape[0]):
        band = img_patch[b]
        valid_band = band[~np.isnan(band)]

        # If no valid pixels, skip
        if valid_band.size == 0:
            return False

        if np.nanstd(valid_band) < 1e-7:
            return False

        # If band has exactly one unique value, skip
        if np.unique(valid_band).size == 1:
            return False

        # (Optional) check min/max range
        if valid_band.min() < -9998 or valid_band.max() > 10001:
            return False

    return True


def process_patch(
    patch_id,
    patch_geom,
    anns,
    satellite_images,
    dem_file,
    patch_size_pixels,
    ts_length,
    seed,
    inventory_dir,
    dataset_type,
):
    """
    Process a patch of data for landslide detection.
    This function extracts and processes satellite imagery, DEM data, and landslide annotations
    for a specific geographical patch, preparing the data for machine learning tasks.
    Parameters
    ----------
    patch_id : str
        Unique identifier for the patch.
    patch_geom : shapely.geometry.Polygon
        Geometry defining the patch boundaries.
    anns : geopandas.GeoDataFrame
        Landslide annotations data.
    satellite_images : dict
        Dictionary mapping satellite names to lists of image metadata.
        Each image should have 'path' and 'date' attributes.
    dem_file : str or Path
        Path to the Digital Elevation Model file.
    patch_size_pixels : int
        Size of the patch in pixels.
    ts_length : int
        Number of timesteps (images) to select for the time series.
    seed : int
        Random seed for reproducible image selection.
    inventory_dir : Path
        Directory containing the landslide inventory data.
    dataset_type : str
        Type of dataset to create ('raw' or 'final').
    Returns
    -------
    dict
        Results of the processing, containing status information for each satellite
        and paths to created files.
    Notes
    -----
    The function:
    1. Clips annotations to the patch geometry
    2. Extracts DEM data for the patch
    3. For each satellite:
       - Extracts image patches
       - Validates and filters images
       - Selects a time series of images
       - Creates masks for landslides if annotated
       - Builds and saves an xarray dataset
    """

    inv_name = inventory_dir.name
    results = {}
    # Clip annotations to patch geometry.
    landslides_patch = gpd.clip(anns, patch_geom)
    is_annotated = not landslides_patch.empty

    with rasterio.open(dem_file) as dem_src:
        window = from_bounds(*patch_geom.bounds, transform=dem_src.transform)
        dem_data = dem_src.read(
            window=window, boundless=True, fill_value=np.nan
        )  # already float32
        dem_data[dem_data == -9999] = np.nan
        dem_data = fill_nans(dem_data)

    # Process each satellite independently.
    for satellite, images in satellite_images.items():
        sat_lower = satellite.lower()
        img_candidates = []
        candidate_ts = []
        candidate_trans = []
        candidate_crs = []
        candidate_band_names = None

        for image in images:
            with rasterio.open(image.path) as src:
                window = from_bounds(*patch_geom.bounds, transform=src.transform)
                img_patch = src.read(
                    window=window, boundless=True, fill_value=np.nan
                ).astype("float32")

                # Example for Sentinel-2
                if sat_lower == "s2":
                    img_patch[img_patch == 0] = np.nan
                    img_patch[img_patch == -9999] = np.nan
                    img_patch = np.clip(img_patch, 0, 10000)
                    if not is_valid_patch(img_patch[:-1]):
                        continue
                else:
                    img_patch[img_patch == -9999] = np.nan
                    img_patch = np.clip(img_patch, -50, 1)
                    if not is_valid_patch(img_patch):
                        continue

                img_patch = fill_nans(img_patch)

                trans = src.window_transform(window)
                if candidate_band_names is None:
                    candidate_band_names = src.descriptions

                timestamp = pd.to_datetime(image.date)
                img_candidates.append(img_patch)
                candidate_ts.append(str(timestamp.date()))
                candidate_trans.append(trans)
                candidate_crs.append(src.crs)

        # If no images were extracted, record the result and continue.
        if not img_candidates:
            results = add_results(
                results, patch_id, satellite, None, msg="No images extracted"
            )
            continue

        # Build the base configuration
        config = {
            "satellite": sat_lower,
            "transform": candidate_trans[0],
            "crs": candidate_crs[0],
            "band_names": candidate_band_names,
            "annotated": is_annotated,
            "anns_data": {},
        }

        if is_annotated:
            anns_data = {}
            for _, r in landslides_patch.iterrows():
                anns_data[str(r["id"])] = {
                    "bbox": r["geometry"].bounds,
                    "event_date": str(r["event_date"]),
                    "confidence": r.get("event_conf", None),
                }
            config["anns_data"] = dict(
                sorted(anns_data.items(), key=lambda item: item[1]["event_date"])
            )
            unique_event_dates = pd.to_datetime(
                landslides_patch["event_date"].dropna().unique()
            )
            indices = select_images(
                img_candidates,
                candidate_ts,
                ts_length,
                unique_event_dates,
                satellite,
                seed=seed,
            )
        else:
            indices = select_images(
                img_candidates, candidate_ts, ts_length, None, satellite, seed=seed
            )

        if indices is None:
            results = add_results(
                results, patch_id, satellite, None, msg="Failed to select images"
            )
            continue

        selected_imgs = [img_candidates[i] for i in indices]
        selected_ts = [candidate_ts[i] for i in indices]

        if len(selected_imgs) != ts_length or len(selected_ts) != ts_length:
            # print(f"Skipping creation: not enough valid images for {satellite}.")
            results = add_results(
                results, patch_id, satellite, None, msg="Not enough valid images"
            )
            continue

        # Ensure selected images cover all event dates
        if is_annotated:
            covered_events = set()
            for event_date in unique_event_dates:
                event_covered = any(
                    pd.to_datetime(ts) >= event_date for ts in selected_ts
                )
                if event_covered:
                    covered_events.add(event_date)

            if len(covered_events) != len(unique_event_dates):
                results = add_results(
                    results,
                    patch_id,
                    satellite,
                    None,
                    msg="Not all event dates covered",
                )
                continue

        # Build mask for each selected candidate.
        msk_candidates = []
        if is_annotated:
            valid_geoms = [
                geom
                for geom in landslides_patch["geometry"].tolist()
                if geom is not None and not geom.is_empty and geom.is_valid
            ]
            for i in indices:
                trans = candidate_trans[i]
                msk = geometry_mask(
                    valid_geoms,
                    (patch_size_pixels, patch_size_pixels),
                    trans,
                    all_touched=True,
                    invert=True,
                )
                msk = np.expand_dims(msk, axis=0)
                msk_candidates.append(msk)
        else:
            for _ in indices:
                msk_candidates.append(
                    np.zeros((1, patch_size_pixels, patch_size_pixels), dtype=np.uint8)
                )

        satellite_merged = satellite.replace("-", "").lower()
        if dataset_type == "final":
            patch_file = (
                Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/benchmarking")
                / "final"
                / satellite_merged
                / f"{inv_name.lower()}_{satellite_merged}_{patch_id}.nc"
            )
        else:
            patch_file = (
                Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/benchmarking")
                / "raw"
                / satellite_merged
                / f"{inv_name.lower()}_{satellite_merged}_{patch_id}.nc"
            )
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        if patch_file.exists():
            patch_file.unlink()
        build_and_save_xarray_dataset(
            selected_imgs, msk_candidates, selected_ts, dem_data, config, patch_file
        )
        results = add_results(
            results,
            patch_id,
            satellite,
            landslides_patch if is_annotated else None,
            patch_file=patch_file,
            msg="Processed successfully",
        )
    return results


def add_results(
    results, patch_id, satellite, valid_landslides, patch_file=None, msg=None
):
    """
    Add results from patch processing to a results dictionary.
    Args:
        results (dict): Dictionary to store results, organized by satellite type.
        patch_id (str): Identifier for the processed patch.
        satellite (str): Satellite type (e.g., 'S1', 'S2', 'DEM').
        valid_landslides (list or bool): Landslides valid in this patch (not used in function but required in signature).
        patch_file (Path or str, optional): File path where the patch is stored. Defaults to None.
        msg (str, optional): Additional message or status for this patch. Defaults to None.
    Returns:
        dict: Updated results dictionary with the added patch information.
    """

    if satellite not in results:
        results[satellite] = {"patches": {}}
    results[satellite]["patches"][patch_id] = {
        "file": str(patch_file) if patch_file else None,
        "message": msg,
    }
    return results


def read_annotations(anns_file):
    """
    Read and process geospatial annotations from a file.
    This function reads a geospatial file containing annotations, fixes any invalid
    geometries using the buffer(0) technique, and removes invalid or empty entries.
    Parameters
    ----------
    anns_file : str
        Path to the geospatial annotation file (typically a shapefile)
    Returns
    -------
    geopandas.GeoDataFrame or None
        A GeoDataFrame containing the processed geometries if successful,
        None if an error occurs during processing
    Notes
    -----
    The function performs the following operations:
    - Reads the file using geopandas
    - Fixes invalid geometries using buffer(0)
    - Removes rows with None geometries
    - Removes rows with empty geometries
    - Logs an error and returns None if any exception occurs
    """

    try:
        gdf = gpd.read_file(anns_file)

        # Fix invalid geometries
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom
        )

        # Remove invalid entries
        gdf = gdf.dropna(subset=["geometry"])
        gdf = gdf[~gdf["geometry"].is_empty]

        return gdf
    except Exception as e:
        logging.error(f"Failed to process shapefile: {str(e)}")
        return None


def create_grid_from_raster(metadata, patch_size_pixels, overlap_pixels):
    """
    Creates a regular grid of polygons from raster metadata for patch-based processing.
    This function generates a grid of rectangular polygons (patches) based on the provided
    raster metadata. The grid covers the entire extent of the raster with specified patch size
    and overlap between adjacent patches.
    Parameters
    ----------
    metadata : dict
        Dictionary containing raster metadata with the following keys:
        - 'transform': The affine transformation of the raster
        - 'width': Width of the raster in pixels
        - 'height': Height of the raster in pixels
        - 'crs': Coordinate reference system of the raster
    patch_size_pixels : int
        Size of each patch in pixels
    overlap_pixels : int
        Number of pixels to overlap between adjacent patches
    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the grid cells as polygons with the following columns:
        - 'geometry': The polygon geometry of each grid cell
        - 'grid_id': Unique identifier for each grid cell
    Raises
    ------
    ValueError
        If the patch size minus overlap is less than or equal to zero
    Notes
    -----
    The grid extends slightly beyond the original raster extent to ensure complete coverage.
    Any invalid geometries are automatically fixed using the buffer(0) operation.
    """

    transform = metadata["transform"]
    width = metadata["width"]
    height = metadata["height"]
    crs = metadata["crs"]

    # Calculate dimensions
    resolution_x = transform.a
    resolution_y = abs(transform.e)
    patch_size = patch_size_pixels * resolution_x
    overlap = overlap_pixels * resolution_x
    step = patch_size - overlap

    if step <= 0:
        raise ValueError("Invalid patch size or overlap.")

    # Calculate grid extent
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + resolution_x * width
    ymin = ymax - resolution_y * height

    # Ensure grid covers the entire area
    xmin = np.floor(xmin / step) * step
    ymin = np.floor(ymin / step) * step
    xmax = np.ceil(xmax / step) * step
    ymax = np.ceil(ymax / step) * step

    # Generate grid coordinates
    x_coords = np.arange(xmin, xmax, step)
    y_coords = np.arange(ymin, ymax, step)

    # Create grid cells
    grid_cells = [
        box(x, y, x + patch_size, y + patch_size) for x in x_coords for y in y_coords
    ]

    # Create GeoDataFrame
    grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=crs)
    grid["grid_id"] = range(len(grid))

    # Fix any invalid geometries
    grid["geometry"] = grid["geometry"].apply(
        lambda g: g.buffer(0) if g and not g.is_valid else g
    )

    return grid


def create_patches(
    inventory_dir, dataset_type, patch_size_pixels, overlap_pixels, ts_length, seed=42
):
    """
    Create patches from satellite images for landslide detection.
    This function processes satellite imagery data (Sentinel-1 and Sentinel-2) to create
    patches that can be used for machine learning model training. It loads data cubes,
    creates grid patches based on raster metadata, identifies patches with landslide
    annotations, samples non-annotated patches, and processes each patch in parallel.
    Parameters:
    ----------
    inventory_dir : Path
        Path to the inventory directory containing satellite images and annotations.
    dataset_type : str
        Type of dataset (e.g., 'train', 'val', 'test').
    patch_size_pixels : int
        Size of each patch in pixels.
    overlap_pixels : int
        Number of pixels to overlap between adjacent patches.
    ts_length : int
        Time series length to consider for the patches.
    seed : int, default=42
        Random seed for reproducibility.
    Returns:
    -------
    dict
        Dictionary containing processed patch data for different satellites.
        Structure: {satellite_type: {'patches': {patch_id: patch_data}}}
    Notes:
    -----
    - Loads Sentinel-1 (ascending and descending) and Sentinel-2 data cubes
    - Checks for consistent transforms across satellite data
    - Creates a grid from raster metadata with specified patch size and overlap
    - Balances annotated and non-annotated patches
    - Processes patches in parallel using available CPU cores
    - Saves results to a JSON file
    """

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Load datacubes
    S2_dc = Sentinel2DataCube(inventory_dir, False, True)
    S2_meta = S2_dc.images[0].get_metadata()

    S1_asc_dc = Sentinel1DataCube(inventory_dir, "asc", False, True)
    S1_dsc_dc = Sentinel1DataCube(inventory_dir, "dsc", False, True)
    S1_asc_meta = S1_asc_dc.images[0].get_metadata()
    S1_dsc_meta = S1_dsc_dc.images[0].get_metadata()

    # Define file paths
    anns_file = (
        inventory_dir
        / "annotations"
        / dataset_type
        / f"{inventory_dir.name}_{dataset_type}_with_dates.shp.zip"
    )
    dem_file = inventory_dir / "other" / "dem" / f"{inventory_dir.name}_DEM.tif"

    # Group satellite images
    S12_images = {
        "S1-asc": S1_asc_dc.images,
        "S1-dsc": S1_dsc_dc.images,
        "S2": S2_dc.images,
    }

    # Check for transform consistency
    if (
        S1_asc_meta["transform"] != S1_dsc_meta["transform"]
        or S1_asc_meta["transform"] != S2_meta["transform"]
    ):
        logging.warning("Transforms do not match among satellites.")

    # Load annotations and create grid
    anns = read_annotations(anns_file)
    anns = anns.to_crs(S2_meta["crs"])

    grid_file = (
        inventory_dir
        / "other"
        / f"{inventory_dir.name}_{dataset_type}_grid_patches.shp.zip"
    )
    if grid_file.exists():
        final_grids = gpd.read_file(grid_file)
        logging.info(f"Loaded existing grid patches from {grid_file}")
    else:
        grid = create_grid_from_raster(S2_meta, patch_size_pixels, overlap_pixels)
        grid["is_ann"] = grid.geometry.apply(
            lambda x: not anns[anns.intersects(x)].empty
        )
        annotated_grids = grid[grid["is_ann"]]
        non_annotated_grids = grid[~grid["is_ann"]]

        # Sample non-annotated grids to match the number of annotated grids
        target_non_ann = int(len(annotated_grids))
        sampled_non_ann = non_annotated_grids.sample(
            n=min(target_non_ann, len(non_annotated_grids)), random_state=seed
        )

        # Combine grids and save
        final_grids = pd.concat([annotated_grids, sampled_non_ann])
        final_grids.to_file(grid_file)

        logging.info(
            f"Total grids: {len(grid)}, annotated: {len(annotated_grids)}, sampled non-annotated: {len(sampled_non_ann)}"
        )

    # Configure parallel processing
    total_grids = len(final_grids)
    max_workers = int(os.getenv("SLURM_CPUS_ON_NODE", os.cpu_count()))
    n_workers = min(max_workers, max(1, math.ceil(total_grids / 10)))
    logging.info(f"Using {n_workers} workers for {total_grids} grids.")

    # selected_grid_ids = [5403, 2440, 3193] # Kyrg1
    # selected_grid_ids = [1196, 1246]
    # selected_grid_ids = random.sample(list(final_grids["grid_id"]), 10) # +2440
    # selected_grids = final_grids[final_grids["grid_id"].isin(selected_grid_ids)]
    # for row in selected_grids.itertuples():
    #     process_patch(
    #         row.grid_id, row.geometry, anns, S12_images, dem_file,
    #         patch_size_pixels, ts_length, seed, inventory_dir, dataset_type
    #     )

    # Process patches in parallel
    with parallel_backend("loky", n_jobs=n_workers):
        patch_results = Parallel(verbose=10)(
            delayed(process_patch)(
                row.grid_id,
                row.geometry,
                anns,
                S12_images,
                dem_file,
                patch_size_pixels,
                ts_length,
                seed,
                inventory_dir,
                dataset_type,
            )
            for row in final_grids.itertuples()
        )

    # Merge results
    merged = {}
    for res in patch_results:
        for sat, data in res.items():
            if sat not in merged:
                merged[sat] = {"patches": {}}
            merged[sat]["patches"].update(data["patches"])

    # Save results
    results_file = (
        inventory_dir / f"{inventory_dir.name}_{dataset_type}_patch_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(merged, f, indent=4)
    logging.info(f"Results saved to {results_file}")

    logging.info("Patch creation completed.")
    return merged
