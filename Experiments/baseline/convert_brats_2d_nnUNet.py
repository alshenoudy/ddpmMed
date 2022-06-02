import os
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *


def convert_brats_labels(segmentation: sitk.Image):
    """
    Converts/maps original BraTS labels to continuous labels.
    BraTS raw labels:
        - Label 0: Background/ healthy tissue
        - label 1: Necrotic Tumor core
        - label 2: Edema/Invaded Tissue
        - label 4: Enhancing Tumor

    :param segmentation:
    :return:
    """
    seg = sitk.GetArrayFromImage(segmentation)
    seg_copy = np.zeros_like(seg, dtype=np.uint32)
    seg_copy[seg == 4] = 3
    seg_copy[seg == 1] = 2
    seg_copy[seg == 2] = 1
    return sitk.GetImageFromArray(seg_copy)


def convert_brats_to_2d(dataset_path: str,
                        target_dir: str,
                        slices: list = None) -> None:
    r"""
    Converts BRATS2021 challenge dataset to comply with nnUNet dataset
    structure requirements. Expects a root folder containing each patient
    cases, where each case looks like:
        - \BraTS2021_00000
            - BraTS2021_00000_flair.nii.gz
            - BraTS2021_00000_seg.nii.gz
            - BraTS2021_00000_t1.nii.gz
            - BraTS2021_00000_t1ce.nii.gz
            - BraTS2021_00000_t2.nii.gz

    Exported slices are in the following format: BraTS2021_XXXXXsYYY_ZZZ.nii.gz; where
    XXXXX is the unique patient identifier, sYYY indicates the slice index and ZZZ corresponds
    to the modality identifier

    :param dataset_path: a string like path to BraTS 3D data
    :param target_dir: a string like path to a target directory
    :param slices: a list of indices to slice the images at
    :return: None

    :raises TypeError: when dataset_path is not a string or passed as None
    :raises FileExistsError: if dataset_path does not exist
    :raises RuntimeError: if the program fails at creating the target directory or one of its subdirectories
    :raises RuntimeError: if BraTS data is not properly structured and/or some modalities are missing
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

        # sub-directories
        images_dir = os.path.join(target_dir, "all_images")
        labels_dir = os.path.join(target_dir, "all_labels")

        # create sub-directories
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    except Exception as ex:
        raise RuntimeError(f"failed at creating target_dir or one of its sub-directories [{target_dir}]\n{ex}")

    # constants
    total_cases = len(os.listdir(dataset_path))
    if slices is None or len(slices) == 0:
        # default slices that we selected
        slices = [70, 75, 80, 85, 90, 95, 100]
    brats_modalities = ['flair', 'seg', 't1', 't1ce', 't2']
    modality_mapping = {
        't1': '0000',
        't1ce': '0001',
        't2': '0002',
        'flair': '0003'
    }

    # go over all volumes, slice and save them
    for i, (root, dir_name, files) in enumerate(os.walk(dataset_path)):

        if len(files) > 1:
            file_modalities = [f.split('_')[-1] for f in files]
            file_modalities = [f.split('.')[0].lower() for f in file_modalities]
            file_modalities.sort()  # ['flair', 'seg', 't1', 't1ce', 't2']

            # check if there are any missing modalities
            if file_modalities != brats_modalities or len(file_modalities) != 5:
                raise RuntimeError(f"missing or mismatching file modalities in folder: [{root}]")

            case_id = os.path.basename(root)
            print(f"\rCopying and extracting BraTS 3D to 2D data [{i + 1}/{total_cases}] ..", end="")
            for m in brats_modalities:
                file_name = os.path.join(root, f'{case_id}_{m}.nii.gz')
                volume = sitk.ReadImage(file_name)
                slices_2d = [sitk.JoinSeries(volume[:, :, s]) for s in slices]
                if m.lower() != 'seg':
                    s_paths = [f"BraTS_{case_id.split('_')[-1]}s{s:03d}_{modality_mapping[m]}.nii.gz" for s in slices]
                    s_paths = [os.path.join(images_dir, s_name) for s_name in s_paths]
                    for name, axial_slice in zip(s_paths, slices_2d):
                        axial_slice.SetOrigin(origin=(0.0, 0.0, 0.0))
                        axial_slice.SetSpacing([999, 1, 1])
                        sitk.WriteImage(image=axial_slice, fileName=name)
                else:
                    labels_path = [f"BraTS_{case_id.split('_')[-1]}s{s:03d}.nii.gz" for s in slices]
                    labels_path = [os.path.join(labels_dir, label) for label in labels_path]
                    for name, axial_slice in zip(labels_path, slices_2d):
                        axial_slice.SetOrigin(origin=(0.0, 0.0, 0.0))
                        axial_slice.SetSpacing([999, 1, 1])
                        sitk.WriteImage(image=convert_brats_labels(axial_slice), fileName=name)
    print(f"\n\nFinished converting BraTS data from 3D to 2D\n\n")


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!",
                          dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


convert_brats_to_2d(dataset_path=r"H:\BRATS\3D Training Data",
                    target_dir=r"C:\Users\aalsheno\Desktop\Brats_folds\all_Brats_data"
                    )

generate_dataset_json(
    output_file=
    r"C:\Users\aalsheno\Desktop\Brats_folds\Task501_BraTS2Ds16\dataset.json",

    imagesTr_dir=
    r"C:\Users\aalsheno\Desktop\Brats_folds\Task501_BraTS2Ds16\imagesTr",

    imagesTs_dir=
    r"C:\Users\aalsheno\Desktop\Brats_folds\Task501_BraTS2Ds16\imagesTs",
    modalities=("T1", "T1ce", "T2", "Flair"),
    labels={0: 'background',
            1: 'edema',
            2: 'non-enhancing',
            3: 'enhancing'},
    dataset_name='BraTS2021'
)
