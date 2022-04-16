import torch
import numpy as np
import SimpleITK as sitk
from torch.nn.functional import one_hot
from monai.metrics import compute_meandice, compute_hausdorff_distance
from torchmetrics.functional import jaccard_index, specificity


class SegmentationMetrics:
    def __init__(self, num_classes: int, include_background: bool, device: str = 'cpu'):
        self.num_classes = num_classes
        self.include_background = include_background
        self.device = device
        self.SMOOTH = 1e-09

    def _convert_to_one_hot(self, x: torch.Tensor):
        """
        Takes in an input tensor with  [B, H, W] and converts it
        to a one hot tensor of format [B, Labels, H, W]
        """

        if (x.ndim == 3 and x.shape[0] != 1) or x.ndim == 4:
            x = x.squeeze()
        elif x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim == 4:
            b, c, h, w = x.shape
            raise RuntimeError(f"expected x with [{b}, {h}, {w}] but got [{b}, {c}, {h}, {w}]")

        x = one_hot(x, num_classes=self.num_classes)
        return torch.permute(x, dims=(0, -1, 1, 2)).to(self.device)

    def dice_score(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """
        Calculates Dice score using MONAI metrics
        """
        prediction = self._convert_to_one_hot(prediction)
        ground_truth = self._convert_to_one_hot(ground_truth.squeeze().long())

        dice_scores = compute_meandice(prediction, ground_truth, include_background=self.include_background)
        mean_dice = torch.nanmean(dice_scores)

        return mean_dice, dice_scores

    def hausdorff_distance(self, prediction: torch.Tensor, ground_truth: torch.Tensor, percentile: int = 95):
        """
        Calculates Hausdorff distance using MONAI metrics
        """
        prediction = self._convert_to_one_hot(prediction)
        ground_truth = self._convert_to_one_hot(ground_truth.squeeze().long())

        distances = compute_hausdorff_distance(prediction, ground_truth,
                                               include_background=self.include_background,
                                               percentile=percentile)
        mean_hd = torch.nanmean(distances)

        return mean_hd, distances

    def jaccard_score(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """
        Calculates Jaccard index as implemented in torch-metrics package
        """
        background_index = None if self.include_background else 0
        # pred = self._convert_to_one_hot(prediction)
        # target = self._convert_to_one_hot(ground_truth.squeeze().long())
        ious = jaccard_index(preds=prediction.cpu(), target=ground_truth.cpu(),
                             reduction='none',
                             num_classes=self.num_classes,
                             ignore_index=background_index)
        mean_iou = torch.nanmean(ious)
        return mean_iou, ious

    def specificity_score(self, prediction: torch.Tensor, ground_truth: torch.Tensor):
        """
        Calculates specificity score (TN/TN + FP) as it's used in brats leaderboard
        """
        background_index = 0 if self.include_background else None
        pred = self._convert_to_one_hot(prediction)
        target = self._convert_to_one_hot(ground_truth.squeeze().long())
        spec = specificity(preds=pred, target=target,
                           ignore_index=background_index,
                           num_classes=self.num_classes,
                           average='none',
                           mdmc_average='samplewise')

        return torch.nanmean(spec), spec

    def get_all_metrics(self, prediction: torch.Tensor, ground_truth: torch.Tensor):

        mean_dice, dice = self.dice_score(prediction, ground_truth)
        mean_hd95, hd95 = self.hausdorff_distance(prediction, ground_truth)
        # mean_spec, spec = self.specificity_score(prediction, ground_truth)
        mean_iou, iou = self.jaccard_score(prediction, ground_truth)

        mean_dict = {
            "mean_dice": mean_dice,
            "mean_jaccard": mean_iou,
            # "mean_specificity": mean_spec,
            "mean_hd95": mean_hd95
        }

        scores = {
            "dice": dice.squeeze(),
            "jaccard": iou.squeeze(),
            # "specificity": spec,
            "hd95": hd95.squeeze()
        }

        return mean_dict, scores

    # TODO: implement sensitivity
    # TODO: implement stat_scores at pixel level



