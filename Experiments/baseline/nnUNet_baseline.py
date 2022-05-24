import os
import json
import torch
import numpy as np
import SimpleITK as sitk


def convert_brats(dataset_path: str,
                  target_dir: str,
                  slices: list = None) -> None:
    r"""
    Converts BRATS2021 challenge dataset to comply with nnUNet dataset
    structure requirements. Expects a root folder containing each patient
    cases, where each case looks like:
        - \BraTS2021_00000
            - BraTS2021_00000_flair.nii.gz
            - BraTS2021_00000_seg.nii.gz
            - BraTS2021_00000_t1.nii.gz\n
            - BraTS2021_00000_t1ce.nii.gz\n
            - BraTS2021_00000_t2.nii.gz\n

    Afterwards, for each 3D volume multiple slices will be generated and stored as
    follows:

    :param dataset_path:
    :param target_dir:
    :param slices:
    :return: None
    """

    # validate dataset_path
    if dataset_path is None or not isinstance(dataset_path, str):
        raise TypeError(f"expected dataset_path to be of type string but got {type(dataset_path)}")

    if not os.path.exists(dataset_path):
        raise FileExistsError(f"dataset_path [{dataset_path}] does not exist")

    try:
        # try to create target_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
    except Exception as ex:
        raise RuntimeError(f"failed at creating target_dir [{target_dir}]\n{ex}")

    # constants
    if slices is None or len(slices) == 0:
        slices = [70, 75, 80, 85, 90, 95, 100]
    brats_modalities = {
        't1': '0000',
        't1ce': '0001',
        't2': '0002',
        'flair': '0003',
        'seg': '0004'
    }
    total_cases = len(os.listdir(dataset_path))

    for i, (root, dir_name, files) in enumerate(os.walk(dataset_path)):

        if len(files) > 1:
            file_modalities = [f.split('_')[-1] for f in files]
            file_modalities = [f.split('.')[0].lower() for f in file_modalities]
            file_modalities.sort()   # ['flair', 'seg', 't1', 't1ce', 't2']

            # check if there are any missing modalities
            if file_modalities != list(brats_modalities.keys()) or len(file_modalities) != 5:
                raise RuntimeError(f"missing or mismatching file modalities in folder: [{root}]")

            case_id = os.path.basename(root)
            for m in list(brats_modalities.keys()):
                file_name = os.path.join(root, f'{case_id}_{m}.nii.gz')
                volume = sitk.ReadImage(file_name)
                for s in slices:
                    slice_output_name = f"patient_{case_id.split('_')[-1]}s{s:03d}"
                    slice_2d = sitk.JoinSeries(volume[:, :, s])
                    print(slice_output_name)

            # print(f"\rCopying and extracting BraTS 3D to 2D data [{i + 1}/{total_cases}] ..", end="")


src = r"H:\BRATS\3D Training Data"
dst = r"D:\Brats_new_structure"

convert_brats(dataset_path=src, target_dir=dst)









