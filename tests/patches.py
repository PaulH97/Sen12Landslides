import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


class Tests(unittest.TestCase):
    """
    Test suite for validating Sentinel-1 and Sentinel-2 patch data quality and format.
    This class contains a variety of tests to ensure that NetCDF files containing
    satellite imagery patches meet expected quality standards and formatting requirements.
    Tests include:
    - Dimension validation (time, x, y dimensions)
    - Data completeness checks (no NaN or fill values)
    - Variable presence and data type validation
    - Value range validation for different bands
    - Statistical distribution analysis
    - Metadata and attribute validation
    - Coordinate reference system checks
    - Validation of event dates in time series
    - Annotation attribute consistency checks
    The test suite is designed to work with both Sentinel-1 and Sentinel-2 data,
    with appropriate expectations for each satellite type.
    Attributes:
        folder (Path): Path to the directory containing NetCDF files to test
        files (list): Sorted list of NetCDF files found in the folder
        expected_time (int): Expected number of timesteps in each patch (15)
        expected_y (int): Expected y-dimension of patches (128)
        expected_x (int): Expected x-dimension of patches (128)
        s2_vars (set): Expected variables for Sentinel-2 files
        s1_vars (set): Expected variables for Sentinel-1 files
        s2_range (tuple): Expected data range for Sentinel-2 bands
        s1_range (tuple): Expected data range for Sentinel-1 bands
        dem_range (tuple): Expected data range for DEM data
        required_attrs (set): Required attributes for all files
    """

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(
            "/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/data/raw/s2"
        )
        cls.files = sorted(list(cls.folder.glob("*.nc")))
        print(f"Found {len(cls.files)} files in {cls.folder}")

        # Expected patch dimensions
        cls.expected_time = 15
        cls.expected_y = 128
        cls.expected_x = 128

        # Expected variable sets for each satellite type
        cls.s2_vars = {
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
            "DEM",
            "MASK",
            "SCL",
        }
        cls.s1_vars = {"VV", "VH", "DEM", "MASK"}

        # Expected data ranges
        cls.s2_range = (0, 10000)  # For S2 bands and DEM
        cls.s1_range = (-50, 1)  # For S1 VV/VH
        cls.dem_range = (0, 10000)  # For DEM in any file

        # Required attributes (including annotation details)
        cls.required_attrs = {
            "center_lat",
            "center_lon",
            "annotated",
            "satellite",
            "event_date",
            "ann_id",
            "ann_bbox",
            "event_date",
            "date_confidence",
            "pre_post_dates",
        }

    def open_file(self, file_path):
        return xr.open_dataset(file_path, engine="netcdf4")

    def get_satellite_type(self, file_path):
        fp = str(file_path).lower()
        if "s2" in fp:
            return "S2"
        elif "s1asc" in fp or "s1dsc" in fp:
            return "S1"
        return "Unknown"

    def test_no_nan_or_fill_values(self):
        """
        Test that no variable in any dataset file contains NaN values or -9999 fill values.

        This test iterates through all files and checks each data variable to ensure they do not
        contain any NaN or the specific fill value -9999, which would indicate missing or invalid data.
        Skips the 'spatial_ref' variable as it's a metadata/reference variable.
        """
        for file_path in tqdm(self.files, desc="Testing NaNs and fill values"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref":
                            continue
                        data_array = ds[var].values
                        if np.isnan(data_array).any() or (data_array == -9999).any():
                            self.fail(
                                f"Variable '{var}' in file {file_path} contains NaN or -9999 fill values."
                            )

    def test_no_constant_timestep(self):
        """
        Test that ensures no variable in the dataset has constant values for an entire timestep.

        For each file, checks all data variables except "spatial_ref", "MASK", "SCL", and "DEM"
        to verify that each timestep contains more than one unique value.
        Fails if any timestep has only a single unique value across the entire spatial extent.
        """
        for file_path in tqdm(self.files, desc="Testing unique values per timestep"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref" or var in {"MASK", "SCL", "DEM"}:
                            continue
                        data_array = ds[var].values
                        for t in range(data_array.shape[0]):
                            unique_vals = np.unique(data_array[t, :, :])
                            if unique_vals.size == 1:
                                self.fail(
                                    f"Variable '{var}' in file {file_path} contains a single unique value at timestep {t}: {unique_vals}"
                                )

    def test_statistical_distribution(self):
        """
        Calculate and store summary statistics for each variable in the dataset files.
        For each file in self.files, computes statistical measures (mean, std, min, max,
        1st and 99th percentiles) for all data variables except 'spatial_ref', 'MASK', and 'SCL'.
        Results are saved to a CSV file named based on the folder structure.
        """
        summary_stats = []

        for file_path in tqdm(self.files, desc="Computing summary statistics"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref" or var in {"MASK", "SCL"}:
                            continue
                        data = ds[var].values.astype(np.float32)
                        stats = {
                            "file": file_path.stem,
                            "variable": var,
                            "mean": np.mean(data),
                            "std": np.std(data),
                            "min": np.min(data),
                            "max": np.max(data),
                            "p1": np.percentile(data, 1),
                            "p99": np.percentile(data, 99),
                        }
                        summary_stats.append(stats)

        df_stats = pd.DataFrame(summary_stats)
        # Create a unique name for the summary stats file
        csv_file = f"{self.folder.parts[-2]}_{self.folder.parts[-1]}_stats.csv"
        df_stats.to_csv(csv_file, index=False)

    def test_patch_dimensions(self):
        """
        Test that each patch file has the expected dimensions.

        Verifies that all data variables (except 'spatial_ref') in each file
        have dimensions of (time=15, y=128, x=128).
        """
        for file_path in tqdm(self.files, desc="Testing patch dimensions"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref":
                            continue
                        shape = ds[var].shape
                        if (
                            len(shape) != 3
                            or shape[0] != self.expected_time
                            or shape[-2:] != (self.expected_y, self.expected_x)
                        ):
                            self.fail(
                                f"File {file_path} variable '{var}' has shape {shape} "
                                f"instead of ({self.expected_time}, {self.expected_y}, {self.expected_x})."
                            )
                        # Only check the first variable's shape
                        break

    def test_data_ranges(self):
        """
        Tests that data values in the satellite imagery files are within expected ranges.
            - Checks each variable in each file against predefined min/max values.
            - Verifies DEM, Sentinel-2, and Sentinel-1 (VV, VH) data ranges.
        """
        for file_path in tqdm(self.files, desc="Testing data ranges"):
            with self.subTest(file=file_path):
                satellite = self.get_satellite_type(file_path)
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref":
                            continue
                        data = ds[var].values
                        if var == "DEM":
                            expected_min, expected_max = self.dem_range
                        elif satellite == "S2":
                            expected_min, expected_max = self.s2_range
                        elif satellite == "S1" and var in {"VV", "VH"}:
                            expected_min, expected_max = self.s1_range
                        else:
                            continue

                        dmin, dmax = np.min(data), np.max(data)
                        if dmin < expected_min or dmax > expected_max:
                            self.fail(
                                f"In file {file_path}, variable '{var}' has range "
                                f"({dmin}, {dmax}) outside expected ({expected_min}, {expected_max})."
                            )

    def test_expected_variables(self):
        """
        Test that each data file contains all the expected variables based on its satellite type.
        S2 files should have all S2 variables, and S1 files should have all S1 variables.
        """
        for file_path in tqdm(self.files, desc="Testing data variables"):
            with self.subTest(file=file_path):
                satellite = self.get_satellite_type(file_path)
                with self.open_file(file_path) as ds:
                    present = set(ds.data_vars.keys()) - {"spatial_ref"}
                    if satellite == "S2":
                        missing = self.s2_vars - present
                        if missing:
                            self.fail(
                                f"S2 file {file_path} is missing variables: {missing}"
                            )
                    elif satellite == "S1":
                        missing = self.s1_vars - present
                        if missing:
                            self.fail(
                                f"S1 file {file_path} is missing variables: {missing}"
                            )

    def test_attributes(self):
        """
        Test if all required attributes are present in dataset files.
        Iterates through each file and checks for the presence of required attributes.
        """
        for file_path in tqdm(self.files, desc="Testing attributes"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for attr in self.required_attrs:
                        if attr not in ds.attrs:
                            self.fail(f"File {file_path} is missing attribute '{attr}'")

    def test_data_types(self):
        """
        Test that all variables in the NetCDF files have the correct data types.
        - MASK should be uint8
        - Sentinel-2 bands should be int16
        - Sentinel-1 VV and VH bands should be float32
        """
        for file_path in tqdm(self.files, desc="Testing data types"):
            with self.subTest(file=file_path):
                satellite = self.get_satellite_type(file_path)
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref":
                            continue
                        dt = ds[var].dtype
                        if var == "MASK":
                            if dt != np.uint8:
                                self.fail(
                                    f"File {file_path} variable '{var}' dtype {dt} is not uint8."
                                )
                        elif satellite == "S2":
                            if dt != np.int16:
                                self.fail(
                                    f"File {file_path} variable '{var}' dtype {dt} is not int16."
                                )
                        elif satellite == "S1" and var in {"VV", "VH"}:
                            if dt != np.float32:
                                self.fail(
                                    f"File {file_path} variable '{var}' dtype {dt} is not float32."
                                )

    def test_crs_exists(self):
        """
        Test that each file in the dataset has a valid Coordinate Reference System (CRS).

        This test checks all files to ensure they have a defined CRS, either in the
        'spatial_ref' attribute or as a rioxarray CRS property. It fails for any file
        that lacks a CRS definition.
        """
        for file_path in tqdm(self.files, desc="Testing CRS"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    if "spatial_ref" in ds:
                        crs = ds["spatial_ref"].attrs.get("crs_wkt", None)
                        if crs:
                            ds.rio.write_crs(crs, inplace=True)
                    if ds.rio.crs is None:
                        self.fail(f"File {file_path} does not have a CRS.")

    def test_event_dates_in_timeseries(self):
        """
        Test that all event dates in each file's 'event_date' attribute fall within the file's time coordinate range.

        For each file, validates that:
        1. The file has a 'time' coordinate
        2. Any event dates in the 'event_date' attribute can be parsed as dates
        3. All event dates fall within the range of the file's time coordinates
        Currently prints warnings rather than failing when dates are outside the time range.
        """
        for file_path in tqdm(self.files, desc="Testing event dates in timeseries"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    if "time" not in ds.coords:
                        self.fail(
                            f"File {file_path} does not have a 'time' coordinate."
                        )
                    times = pd.to_datetime(ds.coords["time"].values)
                    event_dates_str = ds.attrs.get("event_date", "")
                    if event_dates_str and event_dates_str != "None":
                        event_dates_list = [
                            date.strip()
                            for date in event_dates_str.split(",")
                            if date.strip()
                        ]
                        event_dates_list = [
                            date
                            for date in event_dates_list
                            if date and date.lower() != "none"
                        ]
                        try:
                            event_dates = pd.to_datetime(event_dates_list)
                        except Exception as e:
                            self.fail(
                                f"File {file_path} has invalid event_date values: {e}"
                            )
                        outside_dates = [
                            event
                            for event in event_dates
                            if event < times.min() or event > times.max()
                        ]
                        if outside_dates:
                            print(
                                f"File {file_path} has some event dates not present in the time coordinates: {outside_dates}"
                            )
                        # self.fail(f"File {file_path} has some event dates not present in the time coordinates: {outside_dates}")

    def test_annotation_attributes(self):
        """
        Test that annotation attributes in files are consistent with their annotation status.

        This test verifies:
        - For annotated files (where attrs["annotated"] == "True"), required annotation
            attributes are present and non-empty
        - For non-annotated files, annotation attributes must be empty
        """
        for file_path in tqdm(self.files, desc="Testing annotation attributes"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    annotated = ds.attrs.get("annotated", "False")
                    keys = [
                        "ann_id",
                        "ann_bbox",
                        "event_date",
                        "date_confidence",
                        "pre_post_dates",
                    ]
                    if annotated == "True":
                        for key in keys:
                            if key not in ds.attrs or not ds.attrs[key]:
                                self.fail(
                                    f"Annotated file {file_path} is missing or has empty attribute '{key}'."
                                )
                    else:
                        for key in keys:
                            if ds.attrs.get(key, "") != "":
                                self.fail(
                                    f"Non-annotated file {file_path} should have an empty attribute '{key}', but got '{ds.attrs[key]}'."
                                )


if __name__ == "__main__":
    unittest.main(failfast=True)
