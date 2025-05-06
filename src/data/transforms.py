import numpy as np
import pandas as pd
import torch


class Compose:
    """
    A class that composes several transforms together to be applied on a sample.

    The transforms are applied sequentially in the order they are provided.

    Parameters
    ----------
    transforms : list
        List of transform callables to be applied on the sample.

    Returns
    -------
    sample : object
        The transformed sample after applying all transforms.

    Examples
    --------
    >>> transforms = Compose([
    ...     RandomCrop(size=(10, 10)),
    ...     ToTensor()
    ... ])
    >>> transformed_sample = transforms(sample)
    """

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
    Normalizes image data by subtracting the mean and dividing by the standard deviation.

    This transformer normalizes the 'img' component of the sample dictionary using provided
    mean and standard deviation values. It can also optionally remove a specified channel
    before normalization.

    Args:
        mean (list or torch.Tensor): Mean values for each channel in the image.
        std (list or torch.Tensor): Standard deviation values for each channel in the image.
        rm_channel_idx (int, optional): Index of the channel to remove before normalization.
            If negative, counts from the end. Defaults to None (no channel removed).

    Returns:
        dict: The sample dictionary with the normalized image under the 'img' key.
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
    """
    Transform that adds a Day of Year (DOY) channel to image data.
    This transform computes the day of year for each timestep in the provided time
    array and adds it as an additional channel to the input image tensor. The DOY
    values can be optionally normalized to the range [0, 1].
    Parameters
    ----------
    normalized : bool, default=True
        If True, normalizes the day of year values by dividing by max_doy.
    max_doy : float, default=365.0
        Maximum value for day of year normalization.
    Returns
    -------
    torch.Tensor
        Image tensor with an additional DOY channel. The output tensor shape will be
        (T, C+1, H, W) where T is the number of timesteps, C is the original number
        of channels, and H, W are the height and width of the image.
    """

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
        doy_channel = np.stack(
            [np.full((H, W), val, dtype=np.float32) for val in doy], axis=0
        )
        doy_channel = torch.from_numpy(doy_channel).unsqueeze(1)  # (T, 1, H, W)
        # Concatenate the DOY channel to the image channels along dimension 1
        img_with_doy = torch.cat([img, doy_channel], dim=1)
        return img_with_doy


class RemoveChannel:
    """
    A transform that removes a specified channel from the image in a sample.

    This transform takes a dictionary containing an image tensor and removes a specified channel.
    The channel to remove is determined by the `channel_idx` parameter.

    Parameters:
    ----------
    channel_idx : int, optional
        The index of the channel to remove from the image tensor. If negative, it counts
        from the end of the channel dimension. If None, no channel is removed.

    Returns:
    -------
    dict
        The modified sample with the specified channel removed from the image tensor.

    Example:
    --------
    >>> transform = RemoveChannel(channel_idx=3)
    >>> sample = {'img': torch.randn(12, 13, 64, 64)}  # T=12, C=13, H=64, W=64
    >>> result = transform(sample)
    >>> result['img'].shape
    torch.Size([12, 12, 64, 64])  # C is now 12 after removing channel 3
    """

    def __init__(self, channel_idx=None):
        self.channel_idx = channel_idx

    def __call__(self, sample):
        img = sample["img"]
        T, C, H, W = img.shape

        # Only remove the channel if channel_idx is actually set
        if self.channel_idx is not None:
            # Handle negative indexing
            actual_idx = (
                self.channel_idx if self.channel_idx >= 0 else C + self.channel_idx
            )
            channels = list(range(C))
            channels.remove(actual_idx)
            img = img[:, channels, :, :]

        sample["img"] = img
        return sample
