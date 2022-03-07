import torch
import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

__all__ = [
    "Compose",
    "CenterCrop",
    "RandomHorizontalFlip",
    "Resize",
    "PILToTensor",
    "ConvertImageDtype",
    "Lambda"]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.size)
        mask = F.center_crop(mask, self.size)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask


class PILToTensor:
    def __call__(self, image, mask):
        if not isinstance(image, torch.Tensor):
            image = F.pil_to_tensor(image)

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return image, mask


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, mask):
        image = F.convert_image_dtype(image, self.dtype)
        return image


class Lambda:
    def __init__(self, lam):
        if not callable(lam):
            raise TypeError("argument should be callable.")
        self.lam = lam

    def __call__(self, image, mask):
        return self.lam(image), mask
