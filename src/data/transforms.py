import json
import torch
import numpy as np
from typing import Optional


def _flip_h(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (width dimension)."""
    return x.flip(-1)


def _flip_v(x: torch.Tensor) -> torch.Tensor:
    """Vertical flip (height dimension)."""
    return x.flip(-2)


def _rot90k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate by k*90 degrees."""
    if k % 4 == 0:
        return x
    return torch.rot90(x, k, dims=(-2, -1))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() >= self.p:
            return sample
        sample["img"] = _flip_h(sample["img"])
        sample["msk"] = _flip_h(sample["msk"])
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() >= self.p:
            return sample
        sample["img"] = _flip_v(sample["img"])
        sample["msk"] = _flip_v(sample["msk"])
        return sample


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() >= self.p:
            return sample
        k = int(torch.randint(1, 4, (1,)).item())
        sample["img"] = _rot90k(sample["img"], k)
        sample["msk"] = _rot90k(sample["msk"], k)
        return sample


class Normalize:
    """Normalize image tensor using pre-computed statistics."""
    def __init__(self, norm_json: str, clip_data: bool = False):
        with open(norm_json) as f:
            self.norm_stats = json.load(f)
        
        self.clip_data = clip_data
        self.clip_ranges = {
            's1asc': (-50, 10), 
            's1dsc': (-50, 10), 
            's2': (0, 10000), 
            'dem': (0, 8800)
        }

    def __call__(self, sample: dict) -> dict:
        img = sample["img"]  # [T, C, H, W]
        
        # Determine modality from number of channels
        C = img.shape[1]
        if C <= 3:
            modality = 's1asc'  # or s1dsc, same stats
        else:
            modality = 's2'
        
        if modality not in self.norm_stats:
            return sample
        
        # Clip if enabled
        if self.clip_data and modality in self.clip_ranges:
            clip_min, clip_max = self.clip_ranges[modality]
            img = torch.clamp(img, min=clip_min, max=clip_max)
        
        band_stats = self.norm_stats[modality]
        band_names = [b for b in band_stats["mean"].keys() if b != "DEM"]
        
        # Handle DEM channel separately if present
        has_dem = C > len(band_names)
        n_img_bands = len(band_names)
        
        # Build normalization tensors for image bands
        means = torch.tensor(
            [band_stats["mean"][b] for b in band_names], 
            dtype=torch.float32
        ).view(1, n_img_bands, 1, 1)
        
        stds = torch.tensor(
            [band_stats["std"][b] for b in band_names], 
            dtype=torch.float32
        ).view(1, n_img_bands, 1, 1)
        
        # Normalize image bands
        img_bands = img[:, :n_img_bands]
        img_bands = (img_bands - means) / stds
        
        if has_dem and "DEM" in band_stats["mean"]:
            dem_mean = band_stats["mean"]["DEM"]
            dem_std = band_stats["std"]["DEM"]
            dem = img[:, n_img_bands:]
            
            if self.clip_data:
                dem = torch.clamp(dem, min=0, max=8800)
            dem = (dem - dem_mean) / dem_std
            
            img = torch.cat([img_bands, dem], dim=1)
        else:
            img = img_bands
        
        sample["img"] = img
        return sample