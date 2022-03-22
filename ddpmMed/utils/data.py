import torch
import numpy as np
from PIL import Image
import blobfile as bf
import tifffile as tiff
from typing import Union, Any
from torch.utils.data import DataLoader


def imread(path: str):
    """
    A Generic imread for our use-cases, returns a PIL image for normal images
    and a torch tensor for multi-page tiff images
    """
    if not bf.exists(path):
        raise FileExistsError(f"file ({path}) does not exist")

    extension = path.split('.')[-1].lower()
    if extension in ['tif', 'tiff']:
        image = _read_tiff(path)
    elif extension in ['jpeg', 'jpg', 'png']:
        image = Image.open(path)
    else:
        raise RuntimeError(f"unknown image format ({extension})")
    return image


def _read_tiff(path: str):
    """
    reads tiff images and multi-page tiff images, returns a torch tensor
    with a shape of [channels, height, width]
    """
    image = tiff.imread(path)
    if image.ndim > 2:
        # format is (C, H, W)
        channels = image.shape[-1]
        if channels >= 4:
            _images = list()
            for i in range(0, channels):
                _images.append(torch.from_numpy(image[:, :, i]))
            image = torch.stack(_images, dim=0).squeeze()
    else:
        # format is (H, W)
        image = torch.from_numpy(image).unsqueeze(0)
    return image


def torch2np(x: torch.Tensor, squeeze: bool = False) -> np.ndarray:
    """
    Converts a PyTorch tensor from (BATCH, CHANNELS, H, W) to (W, H, CHANNELS, BATCH)

    :param x: Input tensor
    :param squeeze: Boolean to squeeze single dimensions in output
    :return: numpy tensor in requested format
    """
    if isinstance(x, torch.Tensor):
        if x.device != 'cpu':
            x = x.detach().cpu()
        x = x.numpy()

        if x.ndim == 4:
            # x has shape (b, c, rows, cols)
            x = np.transpose(x, (2, 3, 1, 0))
        elif x.ndim == 3:
            # x has shape (c, rows, cols)
            x = np.transpose(x, (1, 2, 0))

    if squeeze:
        x = x.squeeze()
    return x


def normalize(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes an input x using zi = (xi - min(x))/(max(x) - min(x))

    :param x: input image
    :return: Returns normalized data with the same type
    """
    if isinstance(x, np.ndarray):
        x_min, x_max = np.min(x), np.max(x)
        x = (x - x_min) / ((x_max - x_min) + 1e-12)
    elif isinstance(x, torch.Tensor):
        x_min, x_max = torch.min(x), torch.max(x)
        x = (x - x_min) / ((x_max - x_min) + 1e-12)
    else:
        raise NotImplementedError("Unsupported type: {}".format(type(x)))

    return x
