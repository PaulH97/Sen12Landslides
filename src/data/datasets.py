from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
import xarray as xr

class SITS_HDF5(Dataset):
    def __init__(self, h5_path, transforms=None):
        super().__init__()
        self.h5_file = h5py.File(h5_path, "r")
        self.data = self.h5_file["data"]   # shape: (N, T, C, H, W)
        self.mask = self.h5_file["mask"]   # shape: (N, 1, H, W)
        self.N = self.data.shape[0]
        self.transforms = transforms

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img_np = self.data[idx]   # shape: (T, C, H, W)
        msk_np = self.mask[idx]   # shape: (1, H, W)
        sample = {
            "img": torch.from_numpy(img_np),
            "msk": torch.from_numpy(msk_np)
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample

class SITS(Dataset):
    def __init__(self, data_paths, transforms=None):
        super().__init__()
        self.data_paths = data_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        with xr.open_dataset(file_path) as patch:
            msk = patch['MASK'].expand_dims('bands', axis=1).isel(time=0)
            msk = torch.from_numpy(msk.values.astype(np.float32))
            
            if "time" in patch.coords:
                patch = patch.sortby("time")
            
            # Drop unnecessary variables
            patch = patch.drop_vars(["MASK", "spatial_ref", "SCL"], errors='ignore')

            # Process data arrays
            data_arrays = [patch[var] for var in patch.data_vars] 
            img = xr.concat(data_arrays, dim='bands').transpose('time', 'bands', 'y', 'x')
            img = torch.from_numpy(img.values.astype(np.float32))
        
        sample = {"img": img, "msk": msk}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

