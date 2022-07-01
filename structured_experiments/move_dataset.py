import os
import json
import shutil


source = r"C:\Users\aalsheno\Desktop\Brats_folds\all_Brats_data"
destination = r"E:\1. Datasets\2. BraTS 2D Exported\dst"
data_split = r"F:\splits\dataset_split_16.json"
seed = 16

task_folder = os.path.join(destination, f'Task501_BraTS2Ds16')
os.makedirs(task_folder, exist_ok=True)

imagesTr = os.path.join(task_folder, 'imagesTr')
os.makedirs(imagesTr, exist_ok=True)

imagesTs = os.path.join(task_folder, 'imagesTs')
os.makedirs(imagesTs, exist_ok=True)

labelsTr = os.path.join(task_folder, 'labelsTr')
os.makedirs(labelsTr, exist_ok=True)

labelsTs = os.path.join(task_folder, 'labelsTs')
os.makedirs(labelsTs, exist_ok=True)

with open(data_split, 'r') as jf:
    data = json.load(jf)
jf.close()

modalities = ['0000', '0001', '0002', '0003']

for entry in data['training']:
    name = os.path.basename(entry['image'])
    name = name.split('.nii.gz')[0]
    shutil.copy(
        src=os.path.join(source, "labelsTr", f"{name}.nii.gz"),
        dst=os.path.join(labelsTr, f"{name}.nii.gz")
    )

    # move all modalities
    for m in modalities:
        base_name = f"{name}_{m}.nii.gz"
        file = os.path.join(source, "imagesTr", base_name)
        shutil.copy(file, os.path.join(imagesTr, base_name))


for entry in data['testing']:
    name = os.path.basename(entry['image'])
    name = name.split('.nii.gz')[0]
    shutil.copy(
        src=os.path.join(source, "labelsTr", f"{name}.nii.gz"),
        dst=os.path.join(labelsTs, f"{name}.nii.gz")
    )

    # move all modalities
    for m in modalities:
        base_name = f"{name}_{m}.nii.gz"
        file = os.path.join(source, "imagesTr", base_name)
        shutil.copy(file, os.path.join(imagesTs, base_name))
