import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.data import DataLoader
from torch.utils.data import random_split
from Experiments.config import brats_128x128_config
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from torch.nn.functional import interpolate
from ddpmMed.core.pixel_classifier import Ensemble
from ddpmMed.data.datasets import SegmentationDataset
from ddpmMed.utils.data import scale_features, torch2np, prepare_brats_pixels, balance_labels
from ddpmMed.data.brats import BRATS


brats_data = BRATS(path=r"E:\1. Datasets\1. BRATS 2021\3D\Training")
print(len(brats_data))
train, val, test = random_split(dataset=brats_data, lengths=[50, 201, 1000])

