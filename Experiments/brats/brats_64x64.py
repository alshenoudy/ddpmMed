import os.path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split, DataLoader
from Experiments.config import brats_64x64_config
from ddpmMed.data.datasets import SegmentationDataset, PixelDataset
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from ddpmMed.core.pixel_classifier import Ensemble, Classifier
from ddpmMed.utils.data import dump_brats_dataset, prepare_brats_pixels, scale_features
from ddpmMed.utils.helpers import get_feature_clusters, torch2np

# Setup dataset and feature extractor
config = brats_64x64_config()
config["cache_dir"] = r"F:\diffusion"
blocks = [19, 20]

dataset = SegmentationDataset(images_dir=r"E:\Datasets\BRATS\Stacked 2D BRATS Data\scans",
                              masks_dir=r"E:\Datasets\BRATS\Stacked 2D BRATS Data\masks",
                              image_size=64,
                              device='cuda')

feature_extractor = FeatureExtractorDDPM(steps=[100],
                                         blocks=blocks,
                                         model_path=r"F:\diffusion\model200000.pt",
                                         config=config)

# %% Define training pool, and train/test sets
train_pool, test = random_split(dataset=dataset, lengths=[757, 8000],
                                generator=torch.Generator().manual_seed(42))

train, val = random_split(dataset=train_pool, lengths=[50, 707],
                          generator=torch.Generator().manual_seed(42))

# Dump training data to visualize the content of the split
dump_brats_dataset(dataset=train, dump_folder=r"F:\ddpmMed\Experiments\brats")

# %% Create a pixel classifier dataset
x_data, y_data = prepare_brats_pixels(data=train, feature_extractor=feature_extractor, image_size=64, num_features=256)
pixel_dataset = PixelDataset(x_data=x_data, y_data=y_data)
pixel_dataloader = DataLoader(dataset=pixel_dataset, batch_size=32, shuffle=True)

ensemble = Ensemble(in_features=256, num_classes=4, size=10, init_weights="normal")
# ensemble.load_ensemble(ensemble_folder=r"F:\ddpmMed\Experiments\brats\Ensemble")
ensemble.train(epochs=8, data=pixel_dataloader, cache_folder=r"F:\ddpmMed\Experiments\brats")

#%%
validation_folder = os.path.join(r"F:\ddpmMed\Experiments\brats", "validation")
if not os.path.exists(validation_folder):
    os.makedirs(validation_folder, exist_ok=True)

for i, (image, mask) in enumerate(val):

    # predict on current image
    features = scale_features(feature_extractor(image), size=64)
    features = features.reshape(128, (64 * 64)).T
    pred = ensemble.predict(features.cpu()).reshape(64, 64)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(torch2np(pred))
    ax[0].set_axis_off()
    ax[0].set_title("Predictions")

    ax[1].imshow(torch2np(mask, squeeze=True))
    ax[1].set_axis_off()
    ax[1].set_title("Ground Truth")

    plt.savefig(os.path.join(validation_folder, f"prediction_{i}.jpeg"), dpi=300)
    plt.close()
