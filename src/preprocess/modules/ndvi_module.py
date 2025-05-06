import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from datacube import Sentinel2DataCube
from joblib import Parallel, delayed
from rasterio.mask import mask
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
np.seterr(divide="ignore", invalid="ignore")


def read_annotations(anns_file):
    """
    Read and clean a GeoDataFrame from a shapefile.

    This function reads a shapefile using geopandas, performs several cleaning operations:
    - Fixes invalid geometries using buffer(0)
    - Removes rows with null geometries
    - Removes rows with empty geometries
    - Removes duplicate geometries

    Parameters
    ----------
    anns_file : str
        Path to the shapefile to read

    Returns
    -------
    geopandas.GeoDataFrame or None
        A cleaned GeoDataFrame if successful, None if an error occurs
    """

    try:
        gdf = gpd.read_file(anns_file)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom
        )
        gdf = gdf.dropna(subset=["geometry"])
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf.drop_duplicates(subset="geometry")  # Remove duplicates based on geom
        return gdf
    except Exception as e:
        logging.error(f"Failed to process shapefile: {e}")
        return None


def extract_raster_data_for_annotations(
    gdf, raster_file, all_landslides_union, buffer_dist=300
):
    """
    Extract raster data values for each polygon in a GeoDataFrame.
    This function extracts raster values from a given raster file for each geometry in the provided
    GeoDataFrame and buffers around these geometries. It organizes the results by annotation ID
    and date extracted from the raster filename.
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing landslide polygons with geometries and IDs.
    raster_file : str or Path
        Path to the raster file to extract data from.
    all_landslides_union : geometry
        Union of all landslide polygons (not used in current implementation).
    buffer_dist : int, optional
        Buffer distance in map units, defaults to 300.
    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            annotation_id: {
                date_str: {
                    'Polygon': array or None,  # Values within the polygon
                    'Polygon_Buffer': array or None  # Values within the buffer area
        Where arrays are flattened NumPy arrays of valid (non-NaN) pixel values.
    Raises
    ------
    ValueError
        If raster_file is not a string or Path object.
    FileNotFoundError
        If the raster file does not exist.
    Notes
    -----
    - The function expects each row in the GeoDataFrame to have 'id' and optionally 'buffer_geom' fields.
    - The date is extracted from the raster filename, assuming a format where date is the second-to-last
      element when split by underscores.
    - If geometries are invalid or extraction fails, None values are stored and warnings are logged.
    """

    # Input validation
    if not isinstance(raster_file, (str, Path)):
        raise ValueError("raster_file must be a string or Path object")

    raster_file = Path(raster_file)
    if not raster_file.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_file}")

    # Extract date from filename
    try:
        date_str = raster_file.stem.split("_")[-2]
    except IndexError:
        logging.warning(f"Could not extract date from filename: {raster_file.stem}")
        date_str = "unknown_date"

    results = defaultdict(dict)

    def mask_raster(src, geometry):
        """Return the flattened array of valid pixels for a given geometry mask."""
        try:
            data, _ = mask(src, [geometry], crop=True)
        except ValueError:
            return None
        arr = data[0].flatten()
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None
        return arr

    try:
        with rasterio.open(raster_file) as src:
            # Process each annotation
            for idx, row in gdf.iterrows():
                ann_id = int(row.get("id", idx))

                geom = row.geometry
                if geom is None or geom.is_empty or not geom.is_valid:
                    logging.warning(f"Invalid geometry for annotation {ann_id}")
                    results[ann_id][date_str] = {
                        "Polygon": None,
                        "Polygon_Buffer": None,
                    }
                    continue

                geom_buffer = row.get("buffer_geom", None)
                if (
                    geom_buffer is None
                    or geom_buffer.is_empty
                    or not geom_buffer.is_valid
                ):
                    logging.warning(f"Invalid buffer geometry for annotation {ann_id}")
                    results[ann_id][date_str] = {
                        "Polygon": None,
                        "Polygon_Buffer": None,
                    }
                    continue

                data_geom = mask_raster(src, geom)
                data_buf = mask_raster(src, geom_buffer)

                # Store results
                results[ann_id][date_str] = {
                    "Polygon": data_geom,
                    "Polygon_Buffer": data_buf,
                }

    except Exception as e:
        logging.warning(f"Skipping '{raster_file}' due to unexpected error: {e}")
        return {}

    return results


def combine_compute_means(ndvi_dict, scl_dict, valid_pixel_values=None):
    """
    Combines NDVI and SCL data and computes statistical metrics for each annotation polygon.

    This function processes NDVI (Normalized Difference Vegetation Index) and SCL (Scene Classification Layer)
    data to calculate statistical measures on valid pixels within both polygon areas and their buffers.

    Parameters:
    ----------
    ndvi_dict : dict
        Nested dictionary with structure {annotation_id: {date: {"Polygon": ndvi_array, "Polygon_Buffer": ndvi_buffer_array}}}
        containing NDVI values for each polygon and its buffer.

    scl_dict : dict
        Nested dictionary with structure {annotation_id: {date: {"Polygon": scl_array, "Polygon_Buffer": scl_buffer_array}}}
        containing SCL classification values for each polygon and its buffer.

    valid_pixel_values : list, optional
        List of SCL pixel values to consider as valid. Default is [4, 5, 6, 7], which correspond to
        vegetation, bare soils, water, and cloud-free areas in Sentinel-2 SCL classification.

    Returns:
    -------
    dict
        Nested dictionary with structure {annotation_id: {date: {"NDVI": median_value,
                                                        "NDVI_undist": median_buffer_value,
                                                        "NDVI_stats": full_stats_dict,
                                                        "NDVI_undist_stats": full_buffer_stats_dict}}}
        where:
        - "NDVI" is the median NDVI value for valid pixels within the polygon
        - "NDVI_undist" is the median NDVI value for valid pixels within the buffer (undisturbed area)
        - "NDVI_stats" contains detailed statistics for the polygon
        - "NDVI_undist_stats" contains detailed statistics for the buffer area

    Notes:
    -----
    - The function filters NDVI values based on valid SCL classifications before computing statistics
    - If arrays have different sizes, they will be truncated to match the smaller size
    - Statistics are computed using the compute_robust_stats function (not shown)
    - Results will be None for dates/areas where no valid pixels exist
    """

    if valid_pixel_values is None:
        valid_pixel_values = [4, 5, 6, 7]

    results = defaultdict(dict)

    for ann_id, date_dict in ndvi_dict.items():
        for date_str, ndvi_arrays in date_dict.items():
            ndvi_polygon = ndvi_arrays["Polygon"]
            ndvi_buffer = ndvi_arrays["Polygon_Buffer"]

            scl_arrays = scl_dict.get(ann_id, {}).get(date_str, {})
            scl_polygon = scl_arrays.get("Polygon", None)
            scl_buffer = scl_arrays.get("Polygon_Buffer", None)

            stats_ndvi_polygon = None
            stats_ndvi_buffer = None

            # Polygon calculation
            if ndvi_polygon is not None and scl_polygon is not None:
                # Ensure arrays have matching sizes
                min_length = min(len(ndvi_polygon), len(scl_polygon))
                ndvi_polygon = ndvi_polygon[:min_length]
                scl_polygon = scl_polygon[:min_length]

                scl_mask = np.isin(scl_polygon, valid_pixel_values)
                valid_ndvi = ndvi_polygon[scl_mask]
                if valid_ndvi.size > 0:
                    stats_ndvi_polygon = compute_robust_stats(valid_ndvi)

            # Buffer calculation
            if ndvi_buffer is not None and scl_buffer is not None:
                # Ensure arrays have matching sizes
                min_length = min(len(ndvi_buffer), len(scl_buffer))
                ndvi_buffer = ndvi_buffer[:min_length]
                scl_buffer = scl_buffer[:min_length]

                scl_mask_buf = np.isin(scl_buffer, valid_pixel_values)
                valid_ndvi_buf = ndvi_buffer[scl_mask_buf]
                if valid_ndvi_buf.size > 0:
                    stats_ndvi_buffer = compute_robust_stats(valid_ndvi_buf)

            # Store results
            results[ann_id][date_str] = {
                "NDVI": stats_ndvi_polygon["median"] if stats_ndvi_polygon else None,
                "NDVI_undist": (
                    stats_ndvi_buffer["median"] if stats_ndvi_buffer else None
                ),
                "NDVI_stats": stats_ndvi_polygon,
                "NDVI_undist_stats": stats_ndvi_buffer,
            }

    return results


def compute_robust_stats(data):
    """
    Calculate robust statistical measures on the provided data.

    This function computes statistical metrics on the input data after filtering outliers.
    Outliers are defined using the 1.5 IQR method. If the filtered data becomes too small
    (less than 50% of the original data), it falls back to using a 10% trimmed dataset.

    Parameters
    ----------
    data : array-like
        The input data for statistical calculation.

    Returns
    -------
    dict
        A dictionary containing the following statistical measures:
        - mean: The mean of the filtered data.
        - median: The median of the filtered data.
        - std: The standard deviation of the filtered data.
        - n_samples: The number of data points after filtering.
        - n_samples_original: The original number of data points.
    """

    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    if len(filtered_data) < len(data) * 0.5:
        filtered_data = stats.trimboth(data, 0.1)

    return {
        "mean": float(np.mean(filtered_data)),
        "median": float(np.median(filtered_data)),
        # "trimmed_mean": float(stats.trim_mean(filtered_data, 0.1)),
        "std": float(np.std(filtered_data)),
        # "q25": float(np.percentile(filtered_data, 25)),
        # "q75": float(np.percentile(filtered_data, 75)),
        "n_samples": len(filtered_data),
        "n_samples_original": len(data),
    }


def chunkify(lst, chunk_size):
    """
    Divide a list into chunks of specified size.
    This function creates a generator that yields successive chunks from the input list.
    Parameters:
    -----------
    lst : list
        The list to be divided into chunks.
    chunk_size : int
        The size of each chunk.
    Yields:
    -------
    list
        A chunk of the original list with length of at most `chunk_size`.
    Examples:
    ---------
    >>> list(chunkify([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]
    """

    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def extract_ndvi(
    inventory_dir, dataset_type="final", batch_size=20, buffer_dist=300
):
    """
    Extract NDVI (Normalized Difference Vegetation Index) data for landslide annotations.

    This function processes Sentinel-2 imagery to extract NDVI values for landslide polygons
    and buffered regions around them. It also filters pixels based on the Scene Classification
    Layer (SCL) to ensure only valid land pixels are considered.

    Args:
        inventory_dir (str or Path): Directory path containing the landslide inventory data.
        dataset_type (str, optional): Type of dataset to process ('original', 'test', etc.).
            Defaults to "original".
        batch_size (int, optional): Number of images to process in each batch.
            Defaults to 20.
        buffer_dist (int, optional): Buffer distance in meters around landslide polygons.
            Defaults to 300.

    Returns:
        None: Results are saved to a JSON file at {inventory_dir}/other/ndvi/{inventory_name}_{dataset_type}_ndvi_data.json

    Notes:
        - The function uses parallel processing to speed up extraction.
        - SCL values 4, 5, 6, 7 are considered valid land pixels.
        - For each landslide annotation, NDVI statistics are computed for both the landslide area
          and the buffered area surrounding it.
        - Missing dates are filled with None values in the output.
    """

    inventory_dir = Path(inventory_dir)
    anns_file = (
        inventory_dir
        / "annotations"
        / dataset_type
        / f"{inventory_dir.name}_{dataset_type}_final.shp.zip"
    )
    out_file = (
        inventory_dir
        / "other"
        / "ndvi"
        / f"{inventory_dir.name}_{dataset_type}_ndvi_data.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    s2_images = Sentinel2DataCube(inventory_dir, load=False).images
    ndvi_files = [img.get_ndvi_path() for img in s2_images]
    scl_files = [img.band_files["SCL"] for img in s2_images]

    if len(ndvi_files) == 0:
        logging.warning("No NDVI files found.")
        return

    gdf = read_annotations(anns_file)
    if gdf is None or gdf.empty:
        logging.warning("Annotation GeoDataFrame is empty or invalid.")
        return

    crs = s2_images[0].get_metadata()["crs"]
    gdf = gdf.to_crs(crs)

    gdf["geometry"] = gdf["geometry"].simplify(1.0)
    all_landslides_union = gdf.geometry.buffer(50).unary_union
    gdf["buffer_geom"] = (
        gdf.geometry.buffer(buffer_dist).simplify(1.0).difference(all_landslides_union)
    )

    results = defaultdict(dict)
    logging.info("Starting NDVI extraction in batches...")
    batch_index = 0
    n_workers = int(os.getenv("SLURM_NTASKS", 10))

    with Parallel(n_jobs=n_workers, backend="loky", verbose=0) as parallel:
        for ndvi_chunk, scl_chunk in zip(
            chunkify(ndvi_files, batch_size), chunkify(scl_files, batch_size)
        ):
            batch_index += 1
            logging.info(
                f"Processing batch {batch_index} with up to {len(ndvi_chunk)} images"
            )

            ndvi_data_list = parallel(
                delayed(extract_raster_data_for_annotations)(
                    gdf,
                    ndvi_file,
                    all_landslides_union,
                    buffer_dist=buffer_dist,
                )
                for ndvi_file in ndvi_chunk
            )

            scl_data_list = parallel(
                delayed(extract_raster_data_for_annotations)(
                    gdf,
                    scl_file,
                    all_landslides_union,
                    buffer_dist=buffer_dist,
                )
                for scl_file in scl_chunk
            )

            ndvi_combined = defaultdict(dict)
            scl_combined = defaultdict(dict)

            for ndvi_dict in ndvi_data_list:
                for ann_id, date_dict in ndvi_dict.items():
                    ndvi_combined[ann_id].update(date_dict)

            for scl_dict in scl_data_list:
                for ann_id, date_dict in scl_dict.items():
                    scl_combined[ann_id].update(date_dict)

            batch_results = combine_compute_means(
                ndvi_combined, scl_combined, valid_pixel_values=[4, 5, 6, 7]
            )

            for ann_id, date_data in batch_results.items():
                results[ann_id].update(date_data)

    all_dates = set()
    for ann_id in results:
        all_dates.update(results[ann_id].keys())

    for ann_id in results:
        for date in all_dates:
            if date not in results[ann_id]:
                results[ann_id][date] = {
                    "NDVI": None,
                    "NDVI_undist": None,
                    "NDVI_stats": None,
                    "NDVI_undist_stats": None,
                }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info(f"All results written to {out_file}")
