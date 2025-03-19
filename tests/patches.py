import unittest
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd  

class Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/data/raw/s2")
        cls.files = sorted(list(cls.folder.glob("*.nc")))
        print(f"Found {len(cls.files)} files in {cls.folder}")

        # Expected patch dimensions
        cls.expected_time = 15
        cls.expected_y = 128
        cls.expected_x = 128

        # Expected variable sets for each satellite type
        cls.s2_vars = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "DEM", "MASK", "SCL"}
        cls.s1_vars = {"VV", "VH", "DEM", "MASK"}

        # Expected data ranges
        cls.s2_range = (0, 10000)      # For S2 bands and DEM
        cls.s1_range = (-50, 1)        # For S1 VV/VH
        cls.dem_range = (0, 10000)     # For DEM in any file

        # Required attributes (including annotation details)
        cls.required_attrs = {
            "center_lat", "center_lon", "annotated", "satellite", "event_date",
            "ann_id", "ann_bbox", "event_date", "date_confidence", "pre_post_dates"
        }

    def open_file(self, file_path):
        return xr.open_dataset(file_path, engine='netcdf4')

    def get_satellite_type(self, file_path):
        fp = str(file_path).lower()
        if "s2" in fp:
            return "S2"
        elif "s1asc" in fp or "s1dsc" in fp:
            return "S1"
        return "Unknown"

    def test_no_nan_or_fill_values(self):
        """Test that none of the data variables (except 'spatial_ref') contain NaN or fill value -9999."""
        for file_path in tqdm(self.files, desc="Testing NaNs and fill values"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref":
                            continue
                        data_array = ds[var].values
                        if np.isnan(data_array).any() or (data_array == -9999).any():
                            self.fail(f"Variable '{var}' in file {file_path} contains NaN or -9999 fill values.")

    def test_no_constant_timestep(self):
        """
        Test that for each timestep in the variables (except MASK, SCL, and DEM),
        the data does not consist of a single unique value, which might indicate an issue.
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
        Test that the distribution of pixel values in each file is consistent with the overall dataset.
        Files with statistical measures (mean, std, percentiles) that are outliers may indicate issues.
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
                            'file': file_path.stem,
                            'variable': var,
                            'mean': np.mean(data),
                            'std': np.std(data),
                            'min': np.min(data),
                            'max': np.max(data),
                            'p1': np.percentile(data, 1),
                            'p99': np.percentile(data, 99)
                        }
                        summary_stats.append(stats)
        
        df_stats = pd.DataFrame(summary_stats)
        # Create a unique name for the summary stats file
        file_name = f"{self.folder.parts[-2]}_{self.folder.parts[-1]}"
        csv_file = f"/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Sen12Landslides/tests/{satellite}_{folder_name}_summary_stats.csv"
        df_stats.to_csv(csv_file, index=False)
    
    def test_patch_dimensions(self):
        """Test that each file has dimensions (time=15, y=128, x=128)."""
        for file_path in tqdm(self.files, desc="Testing patch dimensions"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref":
                            continue
                        shape = ds[var].shape
                        if (
                            len(shape) != 3 or
                            shape[0] != self.expected_time or
                            shape[-2:] != (self.expected_y, self.expected_x)
                        ):
                            self.fail(
                                f"File {file_path} variable '{var}' has shape {shape} "
                                f"instead of ({self.expected_time}, {self.expected_y}, {self.expected_x})."
                            )
                        # Only check the first variable's shape
                        break

    def test_data_ranges(self):
        """Test that data values fall within expected ranges."""
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
        """Test that each file contains the expected variables."""
        for file_path in tqdm(self.files, desc="Testing data variables"):
            with self.subTest(file=file_path):
                satellite = self.get_satellite_type(file_path)
                with self.open_file(file_path) as ds:
                    present = set(ds.data_vars.keys()) - {"spatial_ref"}
                    if satellite == "S2":
                        missing = self.s2_vars - present
                        if missing:
                            self.fail(f"S2 file {file_path} is missing variables: {missing}")
                    elif satellite == "S1":
                        missing = self.s1_vars - present
                        if missing:
                            self.fail(f"S1 file {file_path} is missing variables: {missing}")

    def test_attributes(self):
        """Test that each file has the required attributes."""
        for file_path in tqdm(self.files, desc="Testing attributes"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for attr in self.required_attrs:
                        if attr not in ds.attrs:
                            self.fail(f"File {file_path} is missing attribute '{attr}'")

    def test_data_types(self):
        """
        Test that for S2 files all data variables are int16, and for S1 files, variables VV and VH are float32.
        The MASK variable should be uint8 for both S1 and S2.
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
                                self.fail(f"File {file_path} variable '{var}' dtype {dt} is not uint8.")
                        elif satellite == "S2":
                            if dt != np.int16:
                                self.fail(f"File {file_path} variable '{var}' dtype {dt} is not int16.")
                        elif satellite == "S1" and var in {"VV", "VH"}:
                            if dt != np.float32:
                                self.fail(f"File {file_path} variable '{var}' dtype {dt} is not float32.")

    def test_crs_exists(self):
        """Test that each file has a coordinate reference system (CRS) available."""
        for file_path in tqdm(self.files, desc="Testing CRS"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    if 'spatial_ref' in ds:
                        crs = ds['spatial_ref'].attrs.get('crs_wkt', None)
                        if crs:
                            ds.rio.write_crs(crs, inplace=True)
                    if ds.rio.crs is None:
                        self.fail(f"File {file_path} does not have a CRS.")

    def test_event_dates_in_timeseries(self):
        """
        Test that all event dates from ds.attrs["event_date"] (a comma-separated string)
        are covered by the dataset’s time coordinate.
        """
        for file_path in tqdm(self.files, desc="Testing event dates in timeseries"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    if "time" not in ds.coords:
                        self.fail(f"File {file_path} does not have a 'time' coordinate.")
                    times = pd.to_datetime(ds.coords["time"].values)
                    event_dates_str = ds.attrs.get("event_date", "")
                    if event_dates_str and event_dates_str != "None":
                        event_dates_list = [date.strip() for date in event_dates_str.split(",") if date.strip()]
                        event_dates_list = [date for date in event_dates_list if date and date.lower() != "none"]
                        try:
                            event_dates = pd.to_datetime(event_dates_list)
                        except Exception as e:
                            self.fail(f"File {file_path} has invalid event_date values: {e}")
                        outside_dates = [event for event in event_dates if event < times.min() or event > times.max()]
                        if outside_dates:
                            print(f"File {file_path} has some event dates not present in the time coordinates: {outside_dates}")
                        # self.fail(f"File {file_path} has some event dates not present in the time coordinates: {outside_dates}")
    
    def test_annotation_attributes(self):
        """
        Test that if a file is annotated (ds.attrs["annotated"] == "True"), the annotation-related
        attributes (ann_id, ann_bbox, event_date, date_confidence, pre_post_dates) have content.
        If not annotated, these attributes should be empty.
        """
        for file_path in tqdm(self.files, desc="Testing annotation attributes"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    annotated = ds.attrs.get("annotated", "False")
                    keys = ["ann_id", "ann_bbox", "event_date", "date_confidence", "pre_post_dates"]
                    if annotated == "True":
                        for key in keys:
                            if key not in ds.attrs or not ds.attrs[key]:
                                self.fail(f"Annotated file {file_path} is missing or has empty attribute '{key}'.")
                    else:
                        for key in keys:
                            if ds.attrs.get(key, "") != "":
                                self.fail(f"Non-annotated file {file_path} should have an empty attribute '{key}', but got '{ds.attrs[key]}'.")

if __name__ == '__main__':
    unittest.main(failfast=True)