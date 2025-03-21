import torch
import numpy as np
import pandas as pd

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class RandomFlip:
    def __init__(self, horizontal_flip_prob=0.5, vertical_flip_prob=0.5):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, sample):
        img, msk = sample["img"], sample["msk"]
        if torch.rand(1).item() < self.horizontal_flip_prob:
            img = img.flip(-1)
            msk = msk.flip(-1)
        if torch.rand(1).item() < self.vertical_flip_prob:
            img = img.flip(-2)
            msk = msk.flip(-2)
        sample["img"] = img
        sample["msk"] = msk
        return sample

class Normalize:
    """
    Normalize sample["img"] by subtracting mean and dividing by std.
    Optionally remove one channel (by index) from both the data and stats.
    
    :param mean: list/array of channel-wise means
    :param std:  list/array of channel-wise stds
    :param rm_channel_idx: None or int. If an int, remove that channel from both
                           the data and the stats. Negative indices are allowed
                           (e.g., -1 removes the last channel).
    """
    def __init__(self, mean, std, rm_channel_idx=None):
        mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

        # If we need to remove a channel from the stats
        if rm_channel_idx is not None:
            C = mean.shape[1]  
            actual_idx = rm_channel_idx if rm_channel_idx >= 0 else C + rm_channel_idx
            channels = list(range(C))
            channels.remove(actual_idx)
            mean = mean[:, channels, :, :]
            std = std[:, channels, :, :]

        self.mean = mean
        self.std = std
        self.rm_channel_idx = rm_channel_idx

    def __call__(self, sample):
        img = sample["img"]
        sample["img"] = (img - self.mean) / self.std
        return sample

class AddDOYTransform:
    def __init__(self, normalized=True, max_doy=365.0):
        self.normalized = normalized
        self.max_doy = max_doy

    def __call__(self, img, time):
        # Convert time to pandas datetime objects and extract day-of-year
        doy = np.array([t.dayofyear for t in pd.to_datetime(time)], dtype=np.float32)
        
        if self.normalized:
            doy = doy / self.max_doy
        
        T, _, H, W = img.shape
    
        # Create a constant DOY channel for each timestep (shape: (T, 1, H, W))
        doy_channel = np.stack([np.full((H, W), val, dtype=np.float32) for val in doy], axis=0)
        doy_channel = torch.from_numpy(doy_channel).unsqueeze(1)  # (T, 1, H, W)
        # Concatenate the DOY channel to the image channels along dimension 1
        img_with_doy = torch.cat([img, doy_channel], dim=1)
        return img_with_doy
    
class RemoveChannel:
    def __init__(self, channel_idx=None):
        self.channel_idx = channel_idx

    def __call__(self, sample):
        img = sample["img"]
        T, C, H, W = img.shape

        # Only remove the channel if channel_idx is actually set
        if self.channel_idx is not None:
            # Handle negative indexing
            actual_idx = self.channel_idx if self.channel_idx >= 0 else C + self.channel_idx
            channels = list(range(C))
            channels.remove(actual_idx)
            img = img[:, channels, :, :]

        sample["img"] = img
        return sample

class FiveCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def five_crops(self, tensor, crop_h, crop_w):
        # tensor shape: [T, C, H, W]
        T, C, H, W = tensor.shape
        crops = [
            tensor[:, :, :crop_h, :crop_w],        # top-left
            tensor[:, :, :crop_h, -crop_w:],        # top-right
            tensor[:, :, -crop_h:, :crop_w],        # bottom-left
            tensor[:, :, -crop_h:, -crop_w:],        # bottom-right
            tensor[:, :, (H - crop_h) // 2:(H - crop_h) // 2 + crop_h,
                         (W - crop_w) // 2:(W - crop_w) // 2 + crop_w]  # center
        ]
        return torch.cat(crops, dim=0)

    def __call__(self, sample):
        crop_h, crop_w = self.size
        for key in ["img", "mask"]:
            if key in sample:
                tensor = sample[key]
                if crop_h > tensor.shape[-2] or crop_w > tensor.shape[-1]:
                    raise ValueError(f"Crop size must be smaller than the {key} size")
                sample[key] = self.five_crops(tensor, crop_h, crop_w).view(-1, tensor.shape[1], crop_h, crop_w)
        return sample