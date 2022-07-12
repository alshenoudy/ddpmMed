import os
import glob
import json
from typing import Tuple

import SimpleITK as sitk


class NiftiEvaluator:
    def __init__(self, predictions: str, references: str) -> None:
        """
        Evaluator class, takes in two directories and evaluates them against specific metrics.
        Both directories should contain the same file names, where files should be nifti files
        (.nii.gz)

        Args:
            predictions (str): A string pointing to the directory exported prediction files.
            references (str): A string  pointing to the directory of ground truth files.
        """
        self.predictions = predictions
        self.references = references

        # Check prediction files against references
        all_preds = glob.glob(os.path.join(self.predictions, f"*.nii.gz"))
        all_refs = glob.glob(os.path.join(self.references, f"*.nii.gz"))

        if len(all_preds) != len(all_refs):
            raise RuntimeError(f"Mismatch between references and predictions {len(references)} != {len(predictions)}")

        for pred, ref in zip(all_preds, all_refs):
            if os.path.basename(pred) != os.path.basename(ref):
                raise RuntimeError(f"File name mismatch {os.path.basename(pred)} != {os.path.basename(ref)}")
        print(f"\nChecked predictions and references directory successfully..\n")

    @staticmethod
    def load_files(pred_path: str, ref_path: str) -> Tuple:
        """
        Reads Nifti files and returns them as numpy arrays
        Args:
            pred_path: path to prediction file
            ref_path: path to reference file

        Returns: Tuple of numpy arrays of prediction and reference images
        """
        prediction = sitk.ReadImage(pred_path)
        prediction = sitk.GetArrayFromImage(prediction)

        reference = sitk.ReadImage(ref_path)
        reference = sitk.GetArrayFromImage(reference)
        return prediction.squeeze(), reference.squeeze()

    def evaluate_file(self, pred_path: str, ref_path: str):
        """
        Evaluates a single prediction against a reference
        Args:
            pred_path: path to prediction file
            ref_path: path to reference file

        Returns: Dictionary of evaluated metrics
        """

        # read files into numpy arrays
        prediction, reference = self.load_files(pred_path=pred_path, ref_path=ref_path)


nifti_evaluator = NiftiEvaluator(
    predictions=
    r"F:\Structured Experiments\Parameter Search\Layer 16\MLP Experiment - 50 Samples"
    r"\Experiment Seed 16\Predictions\Test Data\nifti_predictions",
    references=r"E:\1. Datasets\2. BraTS 2D Exported\dst\Task501_BraTS\labelsTs"
)





