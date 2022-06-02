import numpy as np
from medpy.metric import dc, hd95, specificity, sensitivity


class BraTSMetrics:
    """
    Based on nnUNet's data conversion scripts for BraTS task
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task082_BraTS_2020.py

    A callable object that computes all BraTS related scores
    """

    def __init__(self) -> None:
        pass

    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray):
        """
        Computes all BraTS specific scores and returns a dictionary containing all values for
        all labels Whole Tumor (WT), Enhancing Tumor (ET) and Tumor Core (TC).

        :param ground_truth: numpy array containing ground truth/reference array
        :param prediction: numpy array containing predictions
        :return: dictionary containing all results for all three labels
        """

        # compute scores for whole tumor
        mask_gt = (ground_truth != 0).astype(int)
        mask_pred = (prediction != 0).astype(int)
        dc_whole = self.compute_dice(ground_truth=mask_gt, prediction=mask_pred)
        hd95_whole = self.compute_hd95(ground_truth=mask_gt, prediction=mask_pred)
        spec_whole = self.compute_specificity(ground_truth=mask_gt, prediction=mask_pred)
        sens_whole = self.compute_sensitivity(ground_truth=mask_gt, prediction=mask_pred)
        del mask_gt, mask_pred

        # compute scores for tumor core
        mask_gt = (ground_truth > 1).astype(int)
        mask_pred = (prediction > 1).astype(int)
        dc_core = self.compute_dice(ground_truth=mask_gt, prediction=mask_pred)
        hd95_core = self.compute_hd95(ground_truth=mask_gt, prediction=mask_pred)
        spec_core = self.compute_specificity(ground_truth=mask_gt, prediction=mask_pred)
        sens_core = self.compute_sensitivity(ground_truth=mask_gt, prediction=mask_pred)
        del mask_gt, mask_pred

        # compute scores for enhancing tumor
        mask_gt = (ground_truth == 3).astype(int)
        mask_pred = (prediction == 3).astype(int)
        dc_enhancing = self.compute_dice(ground_truth=mask_gt, prediction=mask_pred)
        hd95_enhancing = self.compute_hd95(ground_truth=mask_gt, prediction=mask_pred)
        spec_enhancing = self.compute_specificity(ground_truth=mask_gt, prediction=mask_pred)
        sens_enhancing = self.compute_sensitivity(ground_truth=mask_gt, prediction=mask_pred)
        del mask_gt, mask_pred

        results = {
            'dice_WT': dc_whole,
            'dice_ET': dc_enhancing,
            'dice_TC': dc_core,
            'sensitivity_WT': sens_whole,
            'sensitivity_ET': sens_enhancing,
            'sensitivity_TC': sens_core,
            'specificity_WT': spec_whole,
            'specificity_ET': spec_enhancing,
            'specificity_TC': spec_core,
            'hd95_WT': hd95_whole,
            'hd95_ET': hd95_enhancing,
            'hd95_TC': hd95_core
        }
        return results

    @staticmethod
    def compute_dice(ground_truth: np.ndarray, prediction: np.ndarray):
        """
        calculates dice score
        """
        # TODO: size/shape handling
        total_gt = np.sum(ground_truth)
        total_pred = np.sum(prediction)

        if total_gt == 0:
            if total_pred == 0:
                return 1
            else:
                return 0
        else:
            return dc(result=prediction, reference=ground_truth)

    @staticmethod
    def compute_hd95(ground_truth: np.ndarray, prediction: np.ndarray):
        """
        calculates hausdorff distance
        """
        total_gt = np.sum(ground_truth)
        total_pred = np.sum(prediction)

        if total_gt == 0:
            if total_pred == 0:
                return 0
            else:
                return 373.12866
        elif total_pred == 0 and total_gt != 0:
            return 373.12866
        else:
            return hd95(result=prediction, reference=ground_truth)

    @staticmethod
    def compute_specificity(ground_truth: np.ndarray, prediction: np.ndarray):
        """ calculates specificity score """
        return specificity(result=prediction, reference=ground_truth)

    @staticmethod
    def compute_sensitivity(ground_truth: np.ndarray, prediction: np.ndarray):
        """ calculates sensitivity score, or recall """
        return sensitivity(result=prediction, reference=ground_truth)
