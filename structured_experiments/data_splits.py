import os
import json
import torch
from torch.utils.data import random_split
from ddpmMed.data.datasets import SegmentationDataset


def generate_split_json(images_dir: str, masks_dir: str, train_pool_size: int, train_size: int,
                        test_size: int, output_dir: str, seeds: list) -> dict:
    """
    Generates a json file for data splits based on given seeds

    """

    # entire dataset
    all_data = SegmentationDataset(images_dir=images_dir,
                                   masks_dir=masks_dir,
                                   image_size=128,
                                   transforms=None,
                                   seed=42)

    # first split (training_pool, test)
    training_pool, test = random_split(dataset=all_data, lengths=[train_pool_size, test_size],
                                       generator=torch.Generator().manual_seed(42))
    data_output = {}

    for seed in seeds:

        # split data into actual training set and validation set
        training, validation = random_split(
            dataset=training_pool, lengths=[train_size, (train_pool_size - train_size)],
            generator=torch.Generator().manual_seed(seed))

        # dictionary holding IDs to cases/splits that will be used for identifying files later
        data_output = {
            'training': list(),
            'testing': list()
        }

        train_indices = training.indices
        test_indices = test.indices

        for index in train_indices:
            brats_id = all_data.dataset[index]['image']
            brats_id = os.path.basename(brats_id).split('.')[0]
            brats_id = brats_id.split('_')
            brats_id = f"BraTS_{int(brats_id[1]):05d}s{int(brats_id[-1]):03d}"
            data_output['training'].append(brats_id)

        for index in test_indices:
            brats_id = all_data.dataset[index]['image']
            brats_id = os.path.basename(brats_id).split('.')[0]
            brats_id = brats_id.split('_')
            brats_id = f"BraTS_{int(brats_id[1]):05d}s{int(brats_id[-1]):03d}"
            data_output['testing'].append(brats_id)

        with open(os.path.join(output_dir, f"dataset_split_{seed}.json"), 'w') as jf:
            json.dump(data_output, jf)
        jf.close()

    return data_output


data = generate_split_json(images_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\scans",
                           masks_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\masks",
                           output_dir=r"F:\splits",
                           train_pool_size=757, test_size=8000, train_size=50, seeds=[16, 42, 256, 1024, 2048])


# TODO: add cmd line approach and proper argparse
