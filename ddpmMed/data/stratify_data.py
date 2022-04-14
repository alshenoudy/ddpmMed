import torch
import numpy as np
from .datasets import PixelDataset


def stratify_features(dataset: PixelDataset):
    """ function to balance pixels in a pixel dataset """
