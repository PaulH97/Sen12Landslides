import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class SITS(Dataset):
    """
    Satellite Image Time Series (SITS) Dataset.
    This class implements a PyTorch Dataset for satellite image time series data, loading
    data from NetCDF/xarray files. Each file is expected to contain multidimensional
    time series data with different variables/bands and a mask.
    Parameters
    ----------
    data_paths : list
        List of file paths to the dataset files.
    transforms : callable, optional
        Optional transform to be applied on a sample.
    Returns
    -------
    dict
        A dictionary containing:
            - 'img': Tensor of shape (time, bands, y, x) with the image data
            - 'msk': Tensor of shape (1, y, x) with the mask data
    Notes
    -----
    - The dataset expects each file to have a 'MASK' variable that will be extracted.
    - The dataset automatically sorts data by time if a time coordinate exists.
    - 'MASK', 'spatial_ref', and 'SCL' variables are dropped from the input data.
    - All remaining variables are concatenated along the 'bands' dimension.
    """

    def __init__(self, data_paths, transforms=None):
        super().__init__()
        self.data_paths = data_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        with xr.open_dataset(file_path) as patch:
            msk = patch["MASK"].expand_dims("bands", axis=1).isel(time=0)
            msk = torch.from_numpy(msk.values.astype(np.float32))

            if "time" in patch.coords:
                patch = patch.sortby("time")

            # Drop unnecessary variables
            patch = patch.drop_vars(["MASK", "spatial_ref", "SCL"], errors="ignore")

            # Process data arrays
            data_arrays = [patch[var] for var in patch.data_vars]
            img = xr.concat(data_arrays, dim="bands").transpose(
                "time", "bands", "y", "x"
            )
            img = torch.from_numpy(img.values.astype(np.float32))

        sample = {"img": img, "msk": msk}

        if self.transforms:
            sample = self.transforms(sample)
        return sample
