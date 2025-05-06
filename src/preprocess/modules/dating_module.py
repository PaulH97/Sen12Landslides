import json
import logging
import random
import shutil
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from datacube import Sentinel2DataCube
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def read_annotations(anns_file):
    """
    Read a geospatial annotations file and process the geometries.
    This function reads a shapefile or other geospatial vector data format using GeoPandas,
    performs basic cleaning operations including:
    - Fixing invalid geometries using buffer(0)
    - Removing rows with null geometries
    - Removing rows with empty geometries
    - Removing duplicate geometries
    Parameters:
    ----------
    anns_file : str
        Path to the annotations file (shapefile or other format supported by GeoPandas)
    Returns:
    -------
    geopandas.GeoDataFrame or None
        Processed GeoDataFrame with valid geometries, or None if the file couldn't be processed
    Raises:
    ------
        No exceptions are raised directly; errors are logged instead and None is returned
    """

    try:
        gdf = gpd.read_file(anns_file)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom
        )
        gdf = gdf.dropna(subset=["geometry"])
        gdf = gdf[~gdf["geometry"].is_empty]
        gdf = gdf.drop_duplicates(subset="geometry")
        return gdf
    except Exception as e:
        logging.error(f"Failed to process shapefile: {str(e)}")
        return None


def harmonic_function(t, beta0, beta1, beta2, beta3):
    """
    Calculate a harmonic function for time series analysis.
    This function implements a harmonic model with a linear trend and seasonal components,
    represented as: β₀ + β₁*t + β₂*cos(2πωt) + β₃*sin(2πωt).
    Parameters:
    ----------
    t : float or numpy.ndarray
        Time values
    beta0 : float
        Constant term (intercept)
    beta1 : float
        Linear trend coefficient
    beta2 : float
        Coefficient for the cosine component
    beta3 : float
        Coefficient for the sine component
    Returns:
    -------
    float or numpy.ndarray
        Computed values of the harmonic function at time t
    Notes:
    -----
    The frequency omega is fixed at 1 for simplicity, representing annual seasonality.
    """

    omega = 1  # Frequency set to 1 for simplicity
    return (
        beta0
        + beta1 * t
        + beta2 * np.cos(2 * np.pi * omega * t)
        + beta3 * np.sin(2 * np.pi * omega * t)
    )


def fit_harmonic(time, ndvi):
    """
    Fit a harmonic function to time series NDVI data.
    This function fits a harmonic function to NDVI time series data using curve fitting
    with suppressed optimization warnings. It automatically generates an initial guess
    for the parameters based on the input data.
    Parameters:
    -----------
    time : array-like
        Time values for the NDVI measurements (usually in days or similar units).
    ndvi : array-like
        NDVI values corresponding to the time points.
    Returns:
    --------
    params : array
        Fitted parameters of the harmonic function in the order:
        [mean, phase, amplitude1, amplitude2]
    Notes:
    ------
    The function uses scipy.optimize.curve_fit to fit a harmonic function to the data,
    with warnings suppressed to avoid OptimizeWarning messages.
    """

    # Suppress warnings within this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        # Improved initial guess based on data
        initial_guess = [np.mean(ndvi), 0, np.ptp(ndvi) / 2, np.ptp(ndvi) / 2]
        params, _ = curve_fit(harmonic_function, time, ndvi, p0=initial_guess)
    return params


def compute_cdndvi(time, ndvi, harmonic_fit):
    """
    Compute Cumulative Difference NDVI (CDNDVI) by subtracting fitted harmonic values from actual NDVI values
    and accumulating the differences.
    Parameters
    ----------
    time : array_like
        Time values used for fitting and evaluation.
    ndvi : array_like
        NDVI values from which to calculate differences.
    harmonic_fit : tuple
        Parameters of the harmonic function fit.
    Returns
    -------
    cdndvi : array_like
        Cumulative sum of differences between actual NDVI values and harmonic fitted values.
    differences : array_like
        Differences between actual NDVI values and harmonic fitted values.
    """

    fitted_values = harmonic_function(time, *harmonic_fit)
    differences = ndvi - fitted_values
    cdndvi = np.cumsum(differences)
    return cdndvi, differences


def clean_data(df):
    """
    Clean and process NDVI data from the input DataFrame.

    This function performs the following operations:
    1. Clips NDVI values to the valid range of -1 to 1
    2. Interpolates short gaps (up to 3 consecutive NaNs) while preserving existing values
    3. Applies exponential weighted moving average smoothing to reduce noise
    4. Handles infinite values and removes rows with remaining NaN values

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least 'NDVI' and 'NDVI_undist' columns with time index

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame containing only the 'NDVI' and 'NDVI_undist' columns

    Notes
    -----
    - 'NDVI' is smoothed with a span of 3
    - 'NDVI_undist' is smoothed with a span of 5
    - The function assumes the input DataFrame has a datetime index for time-based interpolation
    """

    # Clip NDVI values to valid range
    df = df[(df["NDVI"] >= -1) & (df["NDVI"] <= 1)]
    df = df[(df["NDVI_undist"] >= -1) & (df["NDVI_undist"] <= 1)]

    # Interpolate short gaps (up to 3 consecutive NaNs) without changing existing values
    df["NDVI"] = df["NDVI"].where(
        ~df["NDVI"].isna(),
        df["NDVI"].interpolate(method="time", limit=3, limit_direction="both"),
    )
    df["NDVI_undist"] = df["NDVI_undist"].where(
        ~df["NDVI_undist"].isna(),
        df["NDVI_undist"].interpolate(method="time", limit=3, limit_direction="both"),
    )

    # Rolling median smooth
    df["NDVI"] = df["NDVI"].ewm(span=3, adjust=False).mean()
    df["NDVI_undist"] = df["NDVI_undist"].ewm(span=5, adjust=False).mean()

    # Handle infinities and drop rows with remaining NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, subset=["NDVI", "NDVI_undist"], inplace=True)

    return df[["NDVI", "NDVI_undist"]]


def detect_potential_drops(ndvi, threshold):
    """
    Detect potential drops in NDVI values by finding points where the decrease between consecutive values exceeds a threshold.
    This function identifies significant decreases in NDVI time series which may indicate vegetation loss events.
    Parameters
    ----------
    ndvi : numpy.ndarray
        Time series of NDVI values
    threshold : float
        The minimum absolute decrease in NDVI values between consecutive time points to be considered a drop
    Returns
    -------
    numpy.ndarray
        Indices of the time series where NDVI drops exceed the threshold
    """

    # Compute differences between consecutive NDVI values
    ndvi_diff = np.diff(ndvi)
    # Identify indices where the drop exceeds the threshold
    drop_indices = np.where(ndvi_diff <= -threshold)[0]
    return drop_indices


def evaluate_drops(
    drop_indices, ndvi1, ndvi2, time_index, post_window=365, pre_window=365
):
    """
    Evaluate detected NDVI drop events by calculating various metrics to assess their significance.
    This function analyzes each potential drop point by examining the behavior of NDVI values
    before and after the drop, calculating persistence metrics, and assigning confidence scores.
    Parameters
    ----------
    drop_indices : list or array
        Indices of detected drop points in the NDVI time series
    ndvi1 : array-like
        Primary NDVI time series (typically the one experiencing drops)
    ndvi2 : array-like
        Secondary NDVI time series (typically a reference or baseline)
    time_index : pandas.DatetimeIndex
        Datetime index corresponding to the NDVI time series
    post_window : int, default=365
        Number of days to consider after the drop date
    pre_window : int, default=365
        Number of days to consider before the drop date
    Returns
    -------
    list of dict
        Sorted list of drop event dictionaries (sorted by confidence in descending order).
        Each dictionary contains:
        - 'start_index': Index position of the drop
        - 'date': Timestamp of the drop
        - 'pre_median': Median NDVI1 value before the drop
        - 'post_median': Median NDVI1 value after the drop
        - 'post_persistence': Fraction of post-drop values below the pre-drop median
        - 'below_duration': Number of consecutive timesteps where NDVI1 remains below NDVI2
        - 'magnitude': Difference between NDVI2 and NDVI1 at the drop point
        - 'confidence': Composite score indicating the likelihood of a significant drop
    Notes
    -----
    The confidence score is calculated with weighted components:
    - 40% for normalized below_duration
    - 40% for post_persistence
    - 20% for normalized magnitude
    """

    drop_events = []

    for idx in drop_indices:
        # Get the drop date and define pre and post windows
        drop_date = time_index[idx]
        pre_start_date = drop_date - pd.Timedelta(days=pre_window)
        post_end_date = drop_date + pd.Timedelta(days=post_window)

        # Filter pre-drop and post-drop NDVI values
        pre_mask = (time_index >= pre_start_date) & (time_index < drop_date)
        post_mask = (time_index > drop_date) & (time_index <= post_end_date)

        pre_values = ndvi1[pre_mask]
        post_values = ndvi1[post_mask]

        # Calculate pre-drop and post-drop medians
        pre_median = np.median(pre_values) if pre_values.size > 0 else np.nan
        post_median = np.median(post_values) if post_values.size > 0 else np.nan

        # Post-persistence: Fraction of post values below pre-median
        if not np.isnan(pre_median) and not post_values.size == 0:
            post_persistence = (post_values < pre_median).sum() / len(post_values)
        else:
            post_persistence = 0

        # Calculate below_duration: Consecutive timesteps where NDVI1 remains below NDVI2
        below_duration = 0
        i = idx + 1  # Start from the drop point
        while i < len(ndvi1) and ndvi1[i] < ndvi2[i]:
            below_duration += 1
            i += 1
        below_duration_normalized = below_duration / len(ndvi1)  # Normalize

        # Magnitude of the drop: Difference between NDVI1 and NDVI2 at the drop point
        magnitude = max(0, ndvi2[idx] - ndvi1[idx])  # Ensure non-negative magnitude
        magnitude_normalized = (
            magnitude / (ndvi2.max() - ndvi1.min())
            if (ndvi2.max() - ndvi1.min()) > 0
            else 0
        )

        # Confidence: Adjusted weights for better balance
        confidence = (
            below_duration_normalized * 0.4  # 40% weight for below_duration
            + post_persistence * 0.4  # 40% weight for post_persistence
            + magnitude_normalized * 0.2  # 20% weight for magnitude
        )

        drop_events.append(
            {
                "start_index": idx,
                "date": drop_date,
                "pre_median": pre_median,
                "post_median": post_median,
                "post_persistence": post_persistence,
                "below_duration": below_duration,
                "magnitude": magnitude,
                "confidence": confidence,
            }
        )

    # Sort drop events by confidence in descending order
    return sorted(drop_events, key=lambda x: x["confidence"], reverse=True)


def create_plot(ann_ndvis, drop_events, most_confident_drop, output_file):
    """
    Creates and saves a 2x2 plot grid visualizing NDVI time series data and detected landslide events.
    Parameters
    ----------
    ann_ndvis : pandas.DataFrame
        DataFrame containing the time series data with index as dates and columns:
        - 'NDVI': NDVI values for landslide area
        - 'NDVI_undist': NDVI values for undisturbed area
        - 'CDNDVI': Difference between undisturbed and landslide area NDVIs
    drop_events : list of dict
        List of detected drop events, where each event is a dictionary containing:
        - 'start_index': Index in the time series where the drop starts
        - 'confidence': Confidence score of the drop event
        - and potentially other metadata
    most_confident_drop : dict or None
        Dictionary containing information about the most confident drop event:
        - 'start_index': Index in the time series where the drop starts
        - 'pre_date': Date before the landslide event
        - 'post_date': Date after the landslide event
        None if no drop was detected
    output_file : str
        Path where the output plot will be saved
    Returns
    -------
    None
        Function saves the plot to the specified output file but does not return any values
    Notes
    -----
    The function creates four subplots:
    1. NDVI time series for landslide and undisturbed areas
    2. CDNDVI (difference between undisturbed and landslide NDVIs) over time
    3. Confidence scores of all detected drops
    4. NDVI time series highlighting the most confident drop event
    """

    ndvi1 = ann_ndvis["NDVI"]
    ndvi2 = ann_ndvis["NDVI_undist"]
    cdndvi = ann_ndvis["CDNDVI"]

    plt.figure(figsize=(16, 14))

    # Common date formatter for all x-axes
    date_formatter = DateFormatter("%Y-%m-%d")

    # Plot 1: NDVI Time Series
    plt.subplot(2, 2, 1)
    plt.plot(ann_ndvis.index, ndvi1, label="NDVI$_{L}$ (Landslide Area)", color="blue")
    plt.plot(
        ann_ndvis.index, ndvi2, label="NDVI$_{U}$ (Undisturbed Area)", color="green"
    )
    plt.title("NDVI Time Series with Most Confident Drop")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)

    # Plot 2: CDNDVI
    plt.subplot(2, 2, 2)
    plt.plot(ann_ndvis.index, cdndvi, label="CDNDVI", color="purple")
    plt.title("Difference of CDNDVI$_{U}$ and CDNDVI$_{L}$")
    plt.xlabel("Date")
    plt.ylabel("CDNDVI")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)

    # Plot 3: Confidence Scores
    plt.subplot(2, 2, 3)
    confidences = [event["confidence"] for event in drop_events]
    drop_times = [
        ann_ndvis.index[event["start_index"]]
        for event in drop_events
        if 0 <= event["start_index"] < len(ann_ndvis.index)
    ]

    if len(ann_ndvis.index) > 1:
        bar_width = (ann_ndvis.index[1] - ann_ndvis.index[0]).days / 2
    else:
        bar_width = 1  # Default width

    if drop_events:
        plt.bar(drop_times, confidences, width=bar_width, color="orange")
    else:
        plt.annotate(
            "No drops detected",
            (0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
        )

    plt.title("Confidence Scores of Detected Drops")
    plt.xlabel("Date")
    plt.ylabel("Confidence")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)

    # Plot 4: Detected Drops
    plt.subplot(2, 2, 4)
    plt.plot(ann_ndvis.index, ndvi1, label="NDVI$_{L}$ (Landslide Area)", color="blue")

    if most_confident_drop:
        drop_index = most_confident_drop["start_index"]
        if 0 <= drop_index < len(ann_ndvis.index):
            drop_time = ann_ndvis.index[drop_index]
            drop_value = ndvi1.iloc[drop_index]
            plt.axvline(
                most_confident_drop["pre_date"],
                color="orange",
                linestyle="--",
                linewidth=2,
                label="Pre-Date",
            )
            plt.axvline(
                most_confident_drop["post_date"],
                color="purple",
                linestyle="--",
                linewidth=2,
                label="Post-Date",
            )
            plt.plot(drop_time, drop_value, "ro", label="Most Confident Drop")
        else:
            logging.warning(
                f"Invalid drop index {drop_index} for most confident drop in annotation."
            )

    plt.title("Detected Drops in NDVI")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)

    # Adjust layout and save
    plt.tight_layout(pad=2.0)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def get_pre_post_dates(ndvi1, drop_index, time, window_size=10):
    """
    Determine pre-landslide and post-landslide dates based on NDVI values around a drop index.

    This function examines NDVI values within windows before and after a significant drop
    (indicated by drop_index) to determine the optimal pre and post landslide dates.
    It finds the date with highest NDVI before the drop (healthy vegetation)
    and the date with lowest NDVI after the drop (damaged vegetation).

    Parameters
    ----------
    ndvi1 : array-like
        Time series of NDVI values
    drop_index : int
        Index in the time series where a significant drop in NDVI is detected,
        presumed to be related to a landslide event
    time : array-like
        Corresponding dates/times for each NDVI value
    window_size : int, default=10
        Number of time steps to consider before and after the drop index

    Returns
    -------
    tuple
        (pre_date, post_date): The dates identified as best representing
        pre-landslide and post-landslide conditions

    Raises
    ------
    ValueError
        If drop_index is outside the valid range for the ndvi1 array
    """

    # Ensure drop_index is within bounds
    if drop_index < 0 or drop_index >= len(ndvi1):
        raise ValueError(
            f"Invalid drop_index: {drop_index}. Must be between 0 and {len(ndvi1) - 1}."
        )

    # Define the pre-window (before the drop index)
    pre_window_start = max(0, drop_index - window_size)
    pre_window_end = drop_index
    pre_window_ndvi = ndvi1[pre_window_start:pre_window_end]

    # Find the highest NDVI1 value in the pre-window
    if len(pre_window_ndvi) > 0:
        pre_relative_index = np.argmax(pre_window_ndvi)
        pre_global_index = pre_window_start + pre_relative_index
        pre_date = time[pre_global_index]
    else:
        pre_date = time[
            max(0, drop_index)
        ]  # Default to drop time or first available time

    # Define the post-window (after the drop index)
    post_window_start = drop_index + 1
    post_window_end = min(len(ndvi1), drop_index + window_size)
    post_window_ndvi = ndvi1[post_window_start:post_window_end]

    # Find the lowest NDVI1 value in the post-window
    if len(post_window_ndvi) > 0:
        post_relative_index = np.argmin(post_window_ndvi)
        post_global_index = post_window_start + post_relative_index
        post_date = time[post_global_index]
    else:
        post_date = time[
            min(len(time) - 1, drop_index)
        ]  # Default to drop time or last available time

    # Ensure pre_date and post_date are within time range
    pre_date = max(time.min(), pre_date)
    post_date = min(time.max(), post_date)

    return pre_date, post_date


def apply_dating(inventory_dir, dataset_type):
    """
    Apply automated dating to landslide annotations using NDVI time series analysis.
    This function processes landslide annotations to determine pre-event, post-event, and event dates
    by analyzing temporal patterns in the Normalized Difference Vegetation Index (NDVI) data.
    It detects significant drops in NDVI values that likely correspond to landslide events
    and assigns dates accordingly.
    Parameters
    ----------
    inventory_dir : str or Path
        Directory path to the inventory where data is stored
    dataset_type : str
        Type of dataset ('train', 'val', 'test') to process
    Returns
    -------
    None
        Function saves the updated shapefile with dates to disk
    Notes
    -----
    The function:
    1. Loads NDVI time series data from JSON files
    2. Processes each annotation to detect NDVI drops indicating landslide events
    3. Uses harmonic modeling to account for seasonal vegetation patterns
    4. Assigns pre-event, event, and post-event dates based on NDVI patterns
    5. Falls back to manual dates if available or estimates based on partial information
    6. Saves the updated annotations with dates to a new shapefile
    The dating confidence is stored in the 'event_conf' field, with higher values
    indicating greater confidence in the detected event date.
    """

    inventory_dir = Path(inventory_dir)
    inventory_base_name = f"{inventory_dir.name}_{dataset_type}"

    ndvi_file = (
        inventory_dir / "other" / "ndvi" / f"{inventory_base_name}_ndvi_data.json"
    )
    anns_file = (
        inventory_dir
        / "annotations"
        / dataset_type
        / f"{inventory_base_name}_final.shp.zip"
    )
    anns_file_updated = (
        inventory_dir
        / "annotations"
        / dataset_type
        / f"{inventory_base_name}_with_dates.shp.zip"
    )

    figure_dir = inventory_dir / "other" / "figures"
    if figure_dir.is_dir():
        shutil.rmtree(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    with open(ndvi_file) as file:
        anns_ndvi = json.load(file)

    S2_images = Sentinel2DataCube(inventory_dir, load=False).images
    S2_meta = S2_images[0].get_metadata()

    anns = read_annotations(anns_file)
    anns = anns.to_crs(S2_meta["crs"])

    # random_figures = set(random.sample(anns["id"].tolist(), k=10))

    def assign_dates(idx, pre_date, post_date, event_date, confidence):
        # Convert current pre_date, post_date, and event_date to string for safe checking
        current_pre = str(anns.loc[idx, "pre_date"]).strip().lower()
        current_post = str(anns.loc[idx, "post_date"]).strip().lower()
        current_event = str(anns.loc[idx, "event_date"]).strip().lower()

        # If `pre_date` is missing, use `pre_date2` if it exists
        if current_pre in ["", "none", "nan"]:
            if pre_date:
                anns.loc[idx, "pre_date"] = pre_date
            elif anns.loc[idx, "pre_dt2"] not in ["", "none", "nan"]:
                anns.loc[idx, "pre_date"] = anns.loc[idx, "pre_dt2"]

        # If `post_date` is missing, use `post_date2` if it exists
        if current_post in ["", "none", "nan"]:
            if post_date:
                anns.loc[idx, "post_date"] = post_date
            elif anns.loc[idx, "post_dt2"] not in ["", "none", "nan"]:
                anns.loc[idx, "post_date"] = anns.loc[idx, "post_dt2"]

        # If `event_date` is missing, use `event_date2` if it exists
        if current_event in ["", "none", "nan"]:
            if event_date:
                anns.loc[idx, "event_date"] = event_date
            elif anns.loc[idx, "event_dt2"] not in ["", "none", "nan"]:
                anns.loc[idx, "event_date"] = anns.loc[idx, "event_dt2"]

        # Always assign the fallback dates and confidence values
        anns.loc[idx, "pre_dt2"] = pre_date
        anns.loc[idx, "post_dt2"] = post_date
        anns.loc[idx, "event_dt2"] = event_date
        if current_event in ["", "none", "nan"]:
            anns.loc[idx, "event_conf"] = confidence
        else:
            anns.loc[idx, "event_conf"] = 1.0

    def handle_missing_data(row, ann_id, idx):
        # Convert row['event_date'] to datetime
        event_date_ts = pd.to_datetime(row["event_date"])
        if pd.isna(event_date_ts):
            pre_date = post_date = event_date = ""
        else:
            # Check for incomplete date components and assign fallback dates if necessary
            if pd.isna(event_date_ts.month) or pd.isna(event_date_ts.day):
                pre_date = pd.Timestamp(
                    year=event_date_ts.year, month=6, day=1
                ).strftime("%Y-%m-%d")
                post_date = pd.Timestamp(
                    year=event_date_ts.year, month=6, day=30
                ).strftime("%Y-%m-%d")
            else:
                pre_date = (event_date_ts - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                post_date = (event_date_ts + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            # Now format event_date since it is valid
            event_date = event_date_ts.strftime("%Y-%m-%d")
        assign_dates(idx, pre_date, post_date, event_date, 0.0)

    for idx, row in tqdm(
        anns.iterrows(), total=len(anns["id"]), desc="Extracting LS dates"
    ):
        ann_id = row.id
        ann_data = anns_ndvi.get(str(ann_id))
        ann_ndvis = pd.DataFrame.from_dict(ann_data, orient="index")
        ann_ndvis.index = pd.to_datetime(ann_ndvis.index)
        ann_ndvis.sort_index(inplace=True)
        ann_ndvis = clean_data(ann_ndvis)

        if ann_ndvis.empty or len(ann_ndvis) < 4:
            logging.warning(f"Insufficient data for annotation {ann_id}. Skipping...")
            handle_missing_data(row, ann_id, idx)
            continue

        time = ann_ndvis.index
        time_numeric = (
            (time - time[0]).days if isinstance(time, pd.DatetimeIndex) else time
        )

        # Fit the harmonic model to NDVI values
        ndvi1 = ann_ndvis["NDVI"].values
        ndvi2 = ann_ndvis["NDVI_undist"].values

        if len(ndvi1) != len(ann_ndvis.index):
            logging.error(
                f"NDVI1 size does not match ann_ndvis index size after cleaning: {len(ndvi1)} vs {len(ann_ndvis.index)}"
            )
            continue

        drop_indices = detect_potential_drops(ndvi1, threshold=0.20)
        drop_indices = [idx for idx in drop_indices if idx < len(ndvi1)]

        # Filter out drops in the first and last 3 months
        if len(time) > 0:
            start_date = time[0]
            end_date = time[-1]
            three_months = pd.DateOffset(months=3)
            valid_start_date = start_date + three_months
            valid_end_date = end_date - three_months
            drop_indices = [
                idx
                for idx in drop_indices
                if valid_start_date <= time[idx] <= valid_end_date
            ]

        drop_events = evaluate_drops(drop_indices, ndvi1, ndvi2, time)

        most_confident_drop = None
        if drop_events:
            most_confident_drop = drop_events[0]
            drop_index = most_confident_drop["start_index"]
            event_date = time[drop_index]
            pre_date, post_date = get_pre_post_dates(
                ndvi1, drop_index, time, window_size=10
            )
            most_confident_drop["pre_date"] = pre_date
            most_confident_drop["post_date"] = post_date
            most_confident_drop["event_date"] = time[drop_index]
            assign_dates(
                idx,
                pre_date.strftime("%Y-%m-%d"),
                post_date.strftime("%Y-%m-%d"),
                event_date.strftime("%Y-%m-%d"),
                most_confident_drop["confidence"],
            )
        else:
            handle_missing_data(row, ann_id, idx)

        # if ann_id in random_figures and most_confident_drop:
        #     harmonic_fit = fit_harmonic(time_numeric, ndvi1)
        #     cdndvi1 = compute_cdndvi(time_numeric, ndvi1, harmonic_fit)[0]
        #     cdndvi2 = compute_cdndvi(time_numeric, ndvi2, harmonic_fit)[0]
        #     cdndvi = cdndvi2 - cdndvi1
        #     ann_ndvis["CDNDVI"] = cdndvi
        #     figure_file = figure_dir / f"{ann_id}.png"

        #     create_plot(ann_ndvis, drop_events, most_confident_drop, figure_file)
        #     logging.info(f"Saved NDVI dating under: {figure_file}")

    anns.to_file(anns_file_updated, driver="ESRI Shapefile")
    print(anns.head())
    logging.info(f"Saved updated annotations to: {anns_file_updated}")
