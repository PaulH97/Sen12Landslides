import unittest
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

class Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = Path("/dss/dsstbyfs02/pn49cu/pn49cu-dss-0006/Inventories/patches_refined_test/S1-dsc")
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

        # Required attributes
        cls.required_attrs = {
            "center_lat", "center_lon", "anns_data", 
            "annotated", "satellite", "mean_confidence", "event_dates"
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

    def test_no_nan_values(self):
        """Test that none of the data variables (except 'spatial_ref') contain NaN"""
        for file_path in tqdm(self.files, desc="Testing no NaNs and no fill values"):
            with self.subTest(file=file_path):
                with self.open_file(file_path) as ds:
                    for var in ds.data_vars:
                        if var == "spatial_ref":
                            continue
                        data_array = ds[var].values
                        # Check for NaN values
                        if np.isnan(data_array).any():
                            self.fail(f"NaN values found in variable '{var}' in file {file_path}")

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
                                f"instead of ({self.expected_time}, {self.expected_y}, {self.expected_x})"
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
                                f"({dmin}, {dmax}) outside expected ({expected_min}, {expected_max})"
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
                                self.fail(f"File {file_path} variable '{var}' dtype {dt} is not uint8")
                        elif satellite == "S2":
                            if dt != np.int16:
                                self.fail(f"File {file_path} variable '{var}' dtype {dt} is not int16")
                        elif satellite == "S1" and var in {"VV", "VH"}:
                            if dt != np.float32:
                                self.fail(f"File {file_path} variable '{var}' dtype {dt} is not float32")

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
                        self.fail(f"File {file_path} does not have a CRS")

if __name__ == '__main__':
    unittest.main(failfast=True)
