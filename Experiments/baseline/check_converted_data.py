import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

image = sitk.ReadImage(fileName=
                       r"C:\Users\aalsheno\nnUNet\nnUNet_files\nnUNet_raw_data_base\nnUNet_raw_data\Task501_BraTsx16"
                       r"\imagesTr\BraTS_00012s095_0000.nii.gz")

label = sitk.ReadImage(fileName=
                       r"C:\Users\aalsheno\nnUNet\nnUNet_files\nnUNet_raw_data_base\nnUNet_raw_data\Task501_BraTsx16"
                       r"\labelsTr\BraTS_00012s095.nii.gz")

print(f"image size: {image.GetSize()}\nlabel size: {label.GetSize()}\n\n")
print(f"image origin: {image.GetOrigin()}\nlabel origin: {label.GetOrigin()}")

image.SetOrigin(origin=(0.0, 0.0, 0.0))

image_np = sitk.GetArrayFromImage(image)
label_np = sitk.GetArrayFromImage(label)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image_np.squeeze())
ax[1].imshow(label_np.squeeze())
plt.show()
