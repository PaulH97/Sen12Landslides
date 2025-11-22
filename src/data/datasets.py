import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


class Sen12Landslides(Dataset):
    """
    Single-modality time series dataset for landslide detection.
    Returns a single tensor with all channels (image bands + optional DEM).
    """
    
    def __init__(
        self,
        files: list[dict],
        modality: str = "s2",
        min_date: str = "2015-12-03",
        use_dem: bool = True,
        transforms: Optional[callable] = None,
    ):
        super().__init__()
        self.files = files
        self.modality = modality.lower()
        self.use_dem = use_dem
        self.min_date = pd.to_datetime(min_date)
        self.transforms = transforms
        self.exclude_vars = {"MASK", "DEM", "SCL", "spatial_ref"}

    def __len__(self) -> int:
        return len(self.files)

    def _to_float_days(self, date, ref_date: pd.Timestamp) -> float:
        if date is None or pd.isna(date):
            return float("nan")
        delta = pd.to_datetime(date) - ref_date
        return delta.days + (delta.seconds / 86400.0)

    def _extract_dates(self, ds: xr.Dataset) -> np.ndarray:
        if "time" not in ds.sizes:
            return np.array([0.0], dtype=np.float32)
        
        time_vals = ds["time"].values
        if np.issubdtype(time_vals.dtype, np.number):
            return time_vals.astype(np.float32)
        
        return np.array(
            [self._to_float_days(t, self.min_date) for t in pd.to_datetime(time_vals)],
            dtype=np.float32
        )

    def _extract_mask(self, ds: xr.Dataset) -> torch.Tensor:
        """Extract binary segmentation mask [H, W]."""
        mask_var = ds["MASK"]
        
        if "time" in mask_var.dims:
            mask_var = mask_var.isel(time=0)
        
        mask_var = mask_var.transpose("y", "x")
        spatial_mask = (mask_var.values > 0).astype(np.int64)
        
        return torch.from_numpy(spatial_mask)

    def _extract_image(self, ds: xr.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract image bands and optionally concatenate DEM.
        
        Returns
        -------
        img : torch.Tensor [T, C, H, W]
        dates : torch.Tensor [T]
        """
        ds_img = ds.drop_vars(self.exclude_vars, errors="ignore")
        
        bands = [ds_img[var] for var in ds_img.data_vars]
        if not bands:
            raise ValueError(f"No image bands found for {self.modality}")
        
        img = xr.concat(bands, dim="bands").transpose("time", "bands", "y", "x")
        img_arr = img.values.astype("float32")
        
        # Concatenate DEM if enabled
        if self.use_dem and "DEM" in ds.data_vars:
            dem = ds["DEM"]
            if "time" in dem.dims:
                dem = dem.isel(time=0)
            dem = dem.transpose("y", "x")
            dem_arr = dem.values.astype("float32")
            
            T, C, H, W = img_arr.shape
            dem_expanded = np.broadcast_to(dem_arr[np.newaxis, np.newaxis, ...], (T, 1, H, W))
            img_arr = np.concatenate([img_arr, dem_expanded], axis=1)
        
        dates = self._extract_dates(ds)
        
        return torch.from_numpy(img_arr), torch.tensor(dates, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        sample : dict
            {
                "img": [T, C, H, W],
                "dates": [T],
                "msk": [H, W]
            }
        """
        entry = self.files[idx]
        
        if self.modality not in entry or entry[self.modality] is None:
            raise KeyError(f"Missing modality '{self.modality}' for sample {entry.get('id', idx)}")
        
        file_path = entry[self.modality]
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with xr.open_dataset(file_path) as ds:
            if "time" in ds.coords:
                ds = ds.sortby("time")

            img, dates = self._extract_image(ds)
            msk = self._extract_mask(ds)

        sample = {
            "img": img,      # [T, C, H, W]
            "dates": dates,  # [T]
            "msk": msk       # [H, W]
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample