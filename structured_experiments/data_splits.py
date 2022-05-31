import os
import json
import torch
from torch.utils.data import random_split
from ddpmMed.data.datasets import SegmentationDataset

# split data into a training pool and a testing pool
all_data = SegmentationDataset(images_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\scans",
                               masks_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\masks",
                               image_size=128,
                               transforms=None,
                               seed=42)

training_pool, test = random_split(dataset=all_data, lengths=[757, 8000],
                                   generator=torch.Generator().manual_seed(42))

test_data_indices = test.indices
train_data_indices = training_pool.indices
data = {
    'ImagesPoolTr': list(),
    'ImagesTs': list(),
    'LabelsPoolTr': list(),
    'LabelsTs': list()
}
modalities = ['0000', '0001', '0002', '0003']

for idx in test_data_indices:
    brats_id = all_data.dataset[idx]['image']
    brats_id = os.path.basename(brats_id)
    brats_id = brats_id.split('.tif')[0]
    brats_id = brats_id.split('_')
    brats_id = f"{brats_id[0]}_{brats_id[1]}s{brats_id[-1]}"
    data['labelsTs'].append(f"{brats_id}.nii.gz")
    for m in modalities:
        data['imagesTs'].append(f"{brats_id}_{m}.nii.gz")

for idx in train_data_indices:
    brats_id = all_data.dataset[idx]['image']
    brats_id = os.path.basename(brats_id)
    brats_id = brats_id.split('.tif')[0]
    brats_id = brats_id.split('_')
    brats_id = f"{brats_id[0]}_{brats_id[1]}s{brats_id[-1]}"
    data['labelsTr'].append(f"{brats_id}.nii.gz")
    for m in modalities:
        data['imagesTr'].append(f"{brats_id}_{m}.nii.gz")

with open('dataset_splits.json', 'w') as jf:
    json.dump(data, jf)
jf.close()

# %% save the different folds
seeds = [16, 42, 256, 1024, 2048]

for seed in seeds:
    train, val = random_split(dataset=training_pool, lengths=[50, 707],
                              generator=torch.Generator().manual_seed(seed))

    data = {
        'training': list(),
        'validation': list()
    }
    train_idxs = train.indices
    val_idxs = val.indices

    for tr_idx in train_idxs:
        brats_id = all_data.dataset[tr_idx]['image']
        brats_id = os.path.basename(brats_id)
        brats_id = brats_id.split('.tif')[0]
        brats_id = brats_id.split('_')
        brats_id = f"{brats_id[0]}_{brats_id[1]}s{brats_id[-1]}"
        data['training'].append({
            'image': f"./imagesTr/{brats_id}.nii.gz",
            'label': f"./labelsTr/{brats_id}.nii.gz"
        })

    for val_idx in val_idxs:
        brats_id = all_data.dataset[val_idx]['image']
        brats_id = os.path.basename(brats_id)
        brats_id = brats_id.split('.tif')[0]
        brats_id = brats_id.split('_')
        brats_id = f"{brats_id[0]}_{brats_id[1]}s{brats_id[-1]}"
        data['validation'].append({
            'image': f"./imagesTr/{brats_id}.nii.gz",
            'label': f"./labelsTr/{brats_id}.nii.gz"
        })

    with open(f'dataset_splits_{seed}.json', 'w') as jf:
        json.dump(data, jf)
    jf.close()
