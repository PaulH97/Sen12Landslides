import gc
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from matplotlib import pyplot as plt
from rasterio.warp import Resampling as RioResampling
from rasterio.warp import reproject
from tqdm import tqdm

np.seterr(divide="ignore", invalid="ignore")


class Sentinel2DataCube:
    """
    Class representing a Sentinel-2 data cube.
    This class organizes and loads multiple Sentinel-2 images into a data cube structure,
    making it easier to work with time series of satellite imagery.
    Parameters
    ----------
    base_folder : str or Path
        The base directory containing the Sentinel-2 data.
        Expected structure: base_folder/sentinel-2/images/[image_folders]
    load : bool, optional
        Whether to load the data into memory immediately upon initialization.
        Default is True.
    print : bool, optional
        Whether to print initialization information. Default is True.
    Attributes
    ----------
    base_folder : Path
        Path to the base directory.
    satellite : str
        Name of the satellite mission ("sentinel-2").
    si_folder : Path
        Path to the satellite-specific directory.
    images : list
        List of Sentinel2 image objects.
    data : xarray.Dataset, optional
        The loaded data cube if load=True.
    Methods
    -------
    _print_initialization_info()
        Prints information about the initialized data cube.
    _init_images()
        Initializes the list of Sentinel2 image objects.
    _load_data()
        Loads the image data into an xarray Dataset.
    """

    def __init__(self, base_folder, load=True, print=True):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-2"
        self.si_folder = self.base_folder / self.satellite
        self.images = self._init_images()
        if load:
            self.data = self._load_data()
        if print:
            self._print_initialization_info()

    def _print_initialization_info(self):
        divider = "-" * 20
        dates = [image.date for image in self.images]
        dates.sort()
        # print(f"{divider} {os.path.basename(self.base_folder)} {divider}")
        print(f"{2*divider}")
        print("Initialized data-cube with following parameter:")
        print(f"- Base folder: {self.base_folder}")
        print(f"- Satellite mission: {self.satellite}")
        print(f"- Start-End: {min(dates)} -> {max(dates)}")
        print(f"- Length of data-cube: {len(self.images)}")
        print(f"{2*divider}")
        del dates

    def _init_images(self):
        images = []
        images_folder = self.si_folder / "images"
        for folder in images_folder.iterdir():
            if folder.is_dir():
                image_instance = Sentinel2(folder=folder)
                images.append(image_instance)
        images.sort(key=lambda image: image.date)
        return images

    def _load_data(self):
        s2_files = [s2.path for s2 in self.images if s2.path.exists()]
        datasets = [load_file(file) for file in s2_files]
        self.data = xr.concat(datasets, dim="time")

        crs = self.data.attrs["spatial_ref"].crs_wkt
        transform = extract_transform(self.data)

        self.data.rio.write_crs(crs, inplace=True)
        self.data.rio.write_transform(transform, inplace=True)

        self.data.attrs.pop("spatial_ref", None)

        return self.data


class Sentinel1DataCube:
    """
    A class representing a Sentinel-1 data cube for satellite imagery analysis.
    This class loads and organizes Sentinel-1 satellite imagery from a specified base folder
    and orbit type, allowing for easy access to time-series satellite data.
    Attributes:
        base_folder (Path): Path to the base directory containing satellite data.
        satellite (str): Satellite identifier, fixed as "sentinel-1".
        orbit (str): Orbit type of the Sentinel-1 data (e.g., 'ascending' or 'descending').
        si_folder (Path): Path to the specific Sentinel-1 data folder for the specified orbit.
        images (list): List of Sentinel1 objects representing individual images.
        data (xarray.Dataset): Combined xarray Dataset containing all imagery data when loaded.
    Parameters:
        base_folder (str or Path): Path to the base directory containing satellite data.
        orbit (str): Orbit type of the Sentinel-1 data.
        load (bool, optional): Whether to load the data during initialization. Defaults to True.
        print (bool, optional): Whether to print initialization information. Defaults to True.
    Methods:
        _print_initialization_info(): Prints information about the initialized data cube.
        _init_images(): Initializes Sentinel1 image objects from available folders.
        _load_data(): Loads and combines all Sentinel-1 data into a single xarray dataset.
    """

    def __init__(self, base_folder, orbit, load=True, print=True):
        self.base_folder = Path(base_folder)
        self.satellite = "sentinel-1"
        self.orbit = orbit
        self.si_folder = self.base_folder / "sentinel-1-new" / self.orbit
        self.images = self._init_images()
        if load:
            self.data = self._load_data()
        if print:
            self._print_initialization_info()

    def _print_initialization_info(self):
        try:
            divider = "-" * 20
            dates = [image.date for image in self.images]
            dates.sort()
            print(f"{2*divider}")
            print("Initialized data-cube with following parameter:")
            print(f"- Base folder: {self.base_folder}")
            print(f"- Satellite mission: {self.satellite}")
            print(f"- Start-End: {min(dates)} -> {max(dates)}")
            print(f"- Length of data-cube: {len(self.images)}")
            print(f"{2*divider}")
            del dates
        except Exception as e:
            print(f"Error during printing initialization info: {e}")

    def _init_images(self):
        images = []
        for folder in self.si_folder.iterdir():
            if folder.is_dir():
                image = Sentinel1(folder=folder)
                images.append(image)
        images.sort(key=lambda image: image.date)
        return images

    def _load_data(self):
        s1_files = [s1.path for s1 in self.images if s1.path.exists()]
        datasets = [load_file(file) for file in s1_files]
        self.data = xr.concat(datasets, dim="time")

        crs = self.data.attrs["spatial_ref"].crs_wkt
        transform = extract_transform(self.data)

        self.data.rio.write_crs(crs, inplace=True)
        self.data.rio.write_transform(transform, inplace=True)

        self.data.attrs.pop("spatial_ref", None)

        return self.data


class Sentinel1:
    """
    A class for handling Sentinel-1 satellite imagery data.
    This class processes Sentinel-1 data by finding individual band files,
    extracting metadata, and providing methods to stack bands into a single GeoTIFF file.
    Attributes
    ----------
    folder : Path
        Path to the folder containing Sentinel-1 band files
    band_files : dict
        Dictionary mapping band names to file paths
    name : str
        Name of the Sentinel-1 product derived from file naming
    date : datetime.date
        Acquisition date of the Sentinel-1 data
    orbit_state : str
        Orbit direction ('ascending' or 'descending')
    path : Path
        Path to the output stacked GeoTIFF file
    Methods
    -------
    stack_bands()
        Stacks all available bands into a single multi-band GeoTIFF file
    load_data()
        Loads the stacked data as an xarray Dataset
    get_metadata()
        Returns rasterio metadata for the stacked file
    plot(band_name)
        Visualizes a specified band from the stacked dataset
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.band_files = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = self._initialize_date()
        self.orbit_state = self._initialize_orbit_state()
        self.path = self.folder / f"{self.name}.tif"

    def _initialize_name(self):
        raster_filepath = Path(next(iter(self.band_files.values())))
        name_parts = raster_filepath.stem.split("_")
        return "_".join(name_parts[:-2])

    def _initialize_date(self):
        match = re.search(
            r"\d{8}", self.name
        )  # Updated to search for 8-digit date format
        if match:
            date = datetime.strptime(
                match.group(0), "%Y%m%d"
            ).date()  # Updated format to YYYYMMDD
            return date
        else:
            raise ValueError(f"Date string does not match the expected pattern.")

    def _initialize_bands(self):
        bands = {}
        for tif_file in self.folder.glob("*.tif"):
            band_name = extract_S1_band_name(tif_file)
            if band_name:
                band_name = band_name.upper()
                bands[band_name] = tif_file
        sorted_keys = sorted(bands.keys())
        bands = {k: bands[k] for k in sorted_keys}
        return bands

    def _initialize_orbit_state(self):
        orbit_names = {"asc": "ascending", "dsc": "descending"}
        for tif_file in self.band_files.values():
            if "asc" in tif_file.stem.lower():
                return orbit_names["asc"]
            elif "dsc" in tif_file.stem.lower():
                return orbit_names["dsc"]
        raise ValueError("No valid orbit state found in the file names.")

    def stack_bands(self):
        try:
            # Use the transform and CRS of the first band as the reference
            first_band_path = next(iter(self.band_files.values()))
            with rasterio.open(first_band_path) as ref:
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_width = ref.width
                ref_height = ref.height

            stacked_bands = []
            band_names = []

            for band_name, band_path in self.band_files.items():
                with rasterio.open(band_path) as src:
                    if src.transform != ref_transform or src.crs != ref_crs:
                        # Reproject the raster to match the reference transform and CRS
                        data = src.read(1)
                        reprojected_data = np.empty(
                            (ref_height, ref_width), dtype=src.dtypes[0]
                        )
                        reproject(
                            source=data,
                            destination=reprojected_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=RioResampling.bilinear,
                        )
                    else:
                        reprojected_data = src.read(1)

                    stacked_bands.append(reprojected_data)
                    if band_name:
                        band_names.append(band_name)

            # Save the stacked bands as a GeoTIFF with band descriptions
            with rasterio.open(
                self.path,
                "w",
                driver="GTiff",
                height=ref_height,
                width=ref_width,
                count=len(stacked_bands),
                dtype=stacked_bands[0].dtype,
                crs=ref_crs,
                transform=ref_transform,
            ) as dst:
                for idx, (band_data, band_name) in enumerate(
                    zip(stacked_bands, band_names), start=1
                ):
                    dst.write(band_data, idx)
                # Set all band descriptions at once:
                dst.descriptions = tuple(band_names)

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        if self.path.exists():
            return load_file(self.path)
        else:
            return None

    def get_metadata(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                meta = src.meta
            return meta
        else:
            return None

    def plot(self, band_name):
        image = self.load_data()
        if image is None:
            raise ValueError("No data available. Please run 'stack_bands' first.")
        if band_name not in image:
            raise ValueError(f"Band {band_name} is not available in the dataset.")

        image[band_name].plot(figsize=(10, 10), cmap="gray")
        plt.title(f"{band_name} for {self.name}")
        plt.axis("off")
        plt.show()


class Sentinel2:
    class Sentinel2:
        """
        Class for handling and processing Sentinel-2 satellite imagery data.
        This class provides functionality to initialize, process, and analyze Sentinel-2
        satellite imagery, including band stacking, NDVI calculation, and visualization.
        Attributes:
            folder (Path): Path to the folder containing Sentinel-2 band files.
            band_files (dict): Dictionary mapping band names to file paths.
            name (str): Name of the Sentinel-2 product derived from file names.
            date (datetime.date): Acquisition date of the Sentinel-2 data.
            path (Path): Path to the stacked GeoTIFF file.
        Methods:
            _initialize_name(): Extract the product name from band file names.
            _initialize_date(): Parse the acquisition date from the product name.
            _initialize_bands(): Discover and organize available band files.
            stack_bands(): Stack all bands into a single GeoTIFF file.
            load_data(): Load the stacked GeoTIFF file as a xarray dataset.
            get_metadata(): Retrieve metadata from the stacked GeoTIFF file.
            calculate_bad_pixel_ratio(): Calculate the ratio of bad pixels using the SCL band.
            get_ndvi_path(): Get the path to the NDVI file, calculating it if necessary.
            calculate_ndvi(save=False): Calculate NDVI from NIR and red bands.
            plot(band_name): Plot a specific band from the dataset.
            plot_rgb(): Create and display an RGB composite image.
        """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.band_files = self._initialize_bands()
        self.name = self._initialize_name()
        self.date = self._initialize_date()
        self.path = self.folder / f"{self.name}.tif"

    def _initialize_name(self):
        filepath = next(iter(self.band_files.values()))
        file_name = filepath.stem
        return "_".join(file_name.split("_")[:-1])

    def _initialize_date(self):
        date_str = self.name.split("_")[-1]
        date = datetime.strptime(date_str, "%Y%m%d").date()
        return date

    def _initialize_bands(self):
        bands = {}
        for tif_file in self.folder.glob("*.tif"):
            band_name = extract_S2_band_name(tif_file)
            if band_name:
                bands[band_name] = tif_file
        sorted_keys = sorted(bands.keys(), key=extract_band_number)
        bands = {k: bands[k] for k in sorted_keys}
        if "SCL" not in bands.keys():
            print(f"SCL missing for: {self.folder}")
        return bands

    def stack_bands(self):
        try:
            # Use the transform and CRS of the first band as the reference
            first_band_path = next(iter(self.band_files.values()))
            with rasterio.open(first_band_path) as ref:
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_width = ref.width
                ref_height = ref.height

            stacked_bands = []
            band_names = []

            for band_name, band_path in self.band_files.items():
                with rasterio.open(band_path) as src:
                    if src.transform != ref_transform or src.crs != ref_crs:
                        # Reproject the raster to match the reference transform and CRS
                        data = src.read(1)
                        reprojected_data = np.empty(
                            (ref_height, ref_width), dtype=src.dtypes[0]
                        )
                        reproject(
                            source=data,
                            destination=reprojected_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=RioResampling.bilinear,
                        )
                    else:
                        reprojected_data = src.read(1)

                    stacked_bands.append(reprojected_data)
                    if band_name:
                        band_names.append(band_name)

            # Save the stacked bands as a GeoTIFF with band descriptions
            with rasterio.open(
                self.path,
                "w",
                driver="GTiff",
                height=ref_height,
                width=ref_width,
                count=len(stacked_bands),
                dtype=stacked_bands[0].dtype,
                crs=ref_crs,
                transform=ref_transform,
            ) as dst:
                for idx, (band_data, band_name) in enumerate(
                    zip(stacked_bands, band_names), start=1
                ):
                    dst.write(band_data, idx)
                # Set all band descriptions at once:
                dst.descriptions = tuple(band_names)

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_data(self):
        if self.path.exists():
            return load_file(self.path)
        else:
            return None

    def get_metadata(self):
        if self.path.exists():
            with rasterio.open(self.path) as src:
                meta = src.meta
            return meta
        else:
            return None

    def calculate_bad_pixel_ratio(self):
        data = self.load_data()
        if data is None:
            raise ValueError(
                f"No data available for {self.name}. Please ensure the bands are stacked."
            )
        if "SCL" not in data:
            raise ValueError(
                "SCL band is missing from the dataset. Cannot calculate bad pixel ratio."
            )
        scl_band = data["SCL"].values
        bad_pixels = np.isin(scl_band, [0, 1, 2, 3, 8, 9, 10, 11])
        bad_pixel_ratio = np.sum(bad_pixels) / scl_band.size * 100
        return bad_pixel_ratio

    def get_ndvi_path(self):
        ndvi_path = self.path.parent / f"{self.name}_NDVI.tif"
        if not ndvi_path.exists():
            self.calculate_ndvi(save=True)
        return ndvi_path

    def calculate_ndvi(self, save=False):
        # Early exit if the input path does not exist
        if not self.path.exists():
            raise FileNotFoundError(f"The path {self.path} does not exist.")

        image = self.load_data()
        nir = np.clip(image.B08.values, 0, 10000)
        red = np.clip(image.B04.values, 0, 10000)

        # Scale to 0-1 range
        nir = nir / 10000.0
        red = red / 10000.0

        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))

        ndvi = np.nan_to_num(ndvi, nan=np.nan, posinf=np.nan, neginf=np.nan)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi_data_array = xr.DataArray(
            ndvi,
            dims=["time", "y", "x"],
            coords={
                "time": image["time"].data,
                "y": image["y"].data,
                "x": image["x"].data,
            },
            name="NDVI",
        )

        image = image.assign(NDVI=ndvi_data_array)

        if save:
            ndvi_path = self.path.parent / f"{self.name}_NDVI.tif"
            if ndvi_path.exists():
                ndvi_path.unlink()
            image["NDVI"].rio.to_raster(ndvi_path)
            self.band_files["NDVI"] = ndvi_path

        return image

    def plot(self, band_name):
        image = self.load_data()
        if image is None:
            raise ValueError("No data available. Please run 'stack_bands' first.")
        if band_name not in image:
            raise ValueError(f"Band {band_name} is not available in the dataset.")

        image[band_name].plot(figsize=(10, 10), cmap="gray")
        plt.title(f"{band_name} for {self.name}")
        plt.axis("off")
        plt.show()

    def plot_rgb(self):
        if self.path.exists():
            image = self.load_data()
            red = image["B04"].values
            green = image["B03"].values
            blue = image["B02"].values

            red_norm = normalize(red)
            green_norm = normalize(green)
            blue_norm = normalize(blue)

            # Stack bands to form an RGB image
            rgb = np.dstack((red_norm, green_norm, blue_norm))

            plt.figure(figsize=(10, 10))
            plt.imshow(rgb)
            plt.title(f"RGB Image for {self.name}")
            plt.axis("off")
            plt.show()


def extract_S2_band_name(file_name):
    """Extracts the band name from a Sentinel-2 file name."""
    pattern = r"(B\d+[A-Z]?|SCL)\.tif"
    match = re.search(pattern, str(file_name))
    return match.group(1) if match else None


def extract_S1_band_name(file_name):
    """Extracts the band name from a Sentinel-1 file name."""
    pattern = r"(vv|vh).*\.tif"
    match = re.search(pattern, str(file_name))
    return match.group(1) if match else None


def extract_transform(xds):
    """Extracts the GeoTransform from a xarray Dataset."""
    new_order_indices = [1, 2, 0, 4, 5, 3]
    original_transform = xds.spatial_ref.attrs["GeoTransform"].split(" ")
    original_transform = [float(val) for val in original_transform]
    original_transform = [original_transform[i] for i in new_order_indices]
    return Affine(*original_transform)


def load_file(file):
    """
    Load a raster file into an xarray Dataset with proper time and spatial dimensions.
    Args:
        file: Path to the raster file
    Returns:
        xr.Dataset: Dataset containing data arrays for each band with time, x, y coordinates
    """
    data = rxr.open_rasterio(file, chunks="auto")

    time = data.attrs["time"]
    data_arrays = []

    for i, long_name in enumerate(data.attrs["long_name"]):
        band_data = data.sel(band=i + 1)
        band_data = band_data.squeeze(drop=True).drop_vars("band")
        band_data = band_data.expand_dims({"time": [time]})

        new_da = xr.DataArray(
            band_data,
            coords={"time": [time], "y": band_data["y"], "x": band_data["x"]},
            dims=["time", "y", "x"],
            name=long_name,
        )
        data_arrays.append(new_da)
    ds = xr.Dataset({da.name: da for da in data_arrays})
    ds = ds.rio.write_crs(data.rio.crs)
    ds = ds.rio.write_transform(data.rio.transform())
    ds.attrs["spatial_ref"] = data.spatial_ref.copy()
    return ds


def normalize(array):
    """ " Normalizes a numpy array to the range [0, 1]."""
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)
