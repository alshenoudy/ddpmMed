import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from Experiments.config import brats_64x64_config
from ddpmMed.data.datasets import SegmentationDataset, PixelDataset
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from ddpmMed.core.pixel_classifier import Ensemble
from ddpmMed.insights.metrics import SegmentationMetrics
from ddpmMed.utils.data import dump_brats_dataset, prepare_brats_pixels, scale_features
from ddpmMed.utils.helpers import get_feature_clusters, torch2np
from matplotlib.lines import Line2D

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
                          generator=torch.Generator().manual_seed(77))

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

metrics = SegmentationMetrics()
all_metrics = {
    'jaccard': list(),
    'dice': list(),
    'fp_error': list(),
    'fn_error': list()
}
with tqdm(enumerate(val), total=len(val)) as pbar:
    for i, (image, mask) in pbar:

        # predict on current image
        features = scale_features(feature_extractor(image), size=64)
        features = features.reshape(256, (64 * 64)).T
        pred = ensemble.predict(features.cpu()).reshape(64, 64)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch2np(pred))
        ax[0].set_axis_off()
        ax[0].set_title("Predictions")

        ax[1].imshow(torch2np(mask, squeeze=True))
        ax[1].set_axis_off()
        ax[1].set_title("Ground Truth")

        if len(torch.unique(mask)) == 1 or len(torch.unique(pred)) == 1:
            scores = metrics.calculate_metrics(torch2np(pred + 1), torch2np(mask.long() + 1, squeeze=True))
        else:
            scores = metrics.calculate_metrics(torch2np(pred), torch2np(mask.long(), squeeze=True))

        for key in scores.keys():
            s = scores[key]
            # change from infinity to nan
            if s == float('inf') or s == float('nan'):
                s = float('nan')
            all_metrics[key].append(s)

        fig.suptitle(f"Jaccard: {scores['jaccard']:.4f}\n"
                     f"Dice: {scores['dice']:.4f}\n"
                     f"FP Error: {scores['fp_error']:.4f}\n"
                     f"FN Error: {scores['fn_error']:.4f}")

        plt.savefig(os.path.join(validation_folder, f"prediction_{i}.jpeg"), dpi=300)
        plt.close()

# %%
print(f"Jaccard mean: {np.mean(all_metrics['jaccard'])}")
print(f"Dice mean: {np.mean(all_metrics['dice'])}")
print(f"FP error mean: {np.mean(all_metrics['fp_error'])}")
print(f"FN error mean: {np.mean(all_metrics['fn_error'])}")
