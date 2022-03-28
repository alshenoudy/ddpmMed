import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


class SegmentationMetrics:
    def __init__(self):
        self.overlap = sitk.LabelOverlapMeasuresImageFilter()

    def calculate_metrics(self, prediction, image):

        self.overlap.Execute(sitk.GetImageFromArray(prediction),
                             sitk.GetImageFromArray(image))
        metrics = {
            'jaccard': self.overlap.GetJaccardCoefficient(),
            'dice': self.overlap.GetDiceCoefficient(),
            'fp_error': self.overlap.GetFalsePositiveError(),
            'fn_error': self.overlap.GetFalseNegativeError()
        }
        return metrics
