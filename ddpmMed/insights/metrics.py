import os
from types import Union
from typing import Any

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


class SegmentationMetrics:
    def __init__(self):
        self.overlap = sitk.LabelOverlapMeasuresImageFilter()
        self.all_metrics = [
            self.jaccard_coefficient,
            self.dice_coefficient,
            self.hausdorff_distance,
            self.false_negatives,
            self.false_positives
        ]

    def _convert_to_sitk_image(self, image: Union[sitk.Image, np.ndarray, torch.Tensor]):
        raise NotImplementedError

    def jaccard_coefficient(self, prediction: Any, ground_truth: Any):
        raise NotImplementedError

    def dice_coefficient(self, prediction: Any, ground_truth: Any):
        raise NotImplementedError

    def false_negatives(self, prediction: Any, ground_truth: Any):
        raise NotImplementedError

    def false_positives(self, prediction: Any, ground_truth: Any):
        raise NotImplementedError

    def hausdorff_distance(self, prediction: Any, ground_truth: Any):
        raise NotImplementedError


