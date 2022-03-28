import os.path

import torch
import numpy as np
from PIL import Image
import blobfile as bf
import tifffile as tiff
from typing import Union, Any, List, Callable
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


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


def dump_brats_dataset(dataset: Dataset, dump_folder: str):
    """ Brats Specific dataset dump """

    dump_folder = os.path.join(dump_folder, "dataset")
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder, exist_ok=True)

    for i, (image, mask) in enumerate(dataset):
        fig, ax = plt.subplots(1, 5)
        ax[0].imshow(torch2np(image)[:, :, 0], cmap="gray")
        ax[1].imshow(torch2np(image)[:, :, 1], cmap="gray")
        ax[2].imshow(torch2np(image)[:, :, 2], cmap="gray")
        ax[3].imshow(torch2np(image)[:, :, 3], cmap="gray")
        ax[4].imshow(torch2np(mask), cmap="gray")

        ax[0].set_title("T1")
        ax[1].set_title("T1ce")
        ax[2].set_title("T2")
        ax[3].set_title("Flair")
        ax[4].set_title("Ground Truth")

        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()
        ax[3].set_axis_off()
        ax[4].set_axis_off()
        plt.savefig(os.path.join(dump_folder, f"sample_{i}.jpeg"))
        plt.close()


def scale_features(activations: List[torch.Tensor], size: int):
    """ Scales a list of activations to a given size """
    assert all([isinstance(act, torch.Tensor) for act in activations])
    resized = []
    for features in activations:
        resized.append(
            interpolate(features, size, mode='bilinear', align_corners=False)[0]
        )
    return torch.cat(resized, dim=0)


def prepare_brats_pixels(data: Any,
                         feature_extractor: Callable,
                         image_size: int,
                         num_features: int):

    image_size = (image_size, image_size)
    x = torch.zeros((len(data), num_features, *image_size), dtype=torch.float32)
    y = torch.zeros((len(data), *image_size), dtype=torch.uint8)

    for i in range(0, len(data)):
        image, mask = data[i]

        # dimensions, and create a features list
        c, h, w = image.shape
        features = feature_extractor(image)
        features = scale_features(features, h)
        x[i] = features
        y[i] = mask
    x = x.permute(1, 0, 2, 3).reshape(num_features, -1).permute(1, 0)
    y = y.flatten()
    y = brats_labels(y)

    return x, y


def brats_labels(mask: torch.Tensor) -> torch.Tensor:
    """ map brats labels """
    mask[mask == 4] = 3
    return mask
