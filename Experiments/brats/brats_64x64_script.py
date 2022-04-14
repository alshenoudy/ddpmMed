import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from Experiments.config import brats_64x64_config, brats_128x128_config
from ddpmMed.data.datasets import SegmentationDataset, PixelDataset
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from ddpmMed.core.pixel_classifier import Ensemble
from ddpmMed.insights.metrics import SegmentationMetrics
from ddpmMed.utils.data import dump_brats_dataset, prepare_brats_pixels, scale_features
from ddpmMed.utils.helpers import get_feature_clusters, torch2np
from ddpmMed.utils.data import brats_labels
import warnings
warnings.filterwarnings("ignore")

# Setup dataset and feature extractor
config = brats_128x128_config()
config["cache_dir"] = r"F:\diffusion"
blocks = [18, 19, 20, 21]

dataset = SegmentationDataset(images_dir=r"E:\1. Datasets\BRATS\2D\Training\scans",
                              masks_dir=r"E:\1. Datasets\BRATS\2D\Training\masks",
                              image_size=128,
                              device='cuda')

feature_extractor = FeatureExtractorDDPM(steps=[100],
                                         blocks=blocks,
                                         model_path=r"C:\Users\ahmed\Desktop\model\model350000.pt",
                                         config=config)

train_pool, val, test = random_split(dataset=dataset, lengths=[257, 500, 8000],
                                     generator=torch.Generator().manual_seed(42))

seeds = [2048]
all_dice_scores = {
    'TC': list(),
    'ED': list(),
    'ET': list(),
    'mean_dice': list()
}

all_hd_distances = {
    'TC': list(),
    'ED': list(),
    'ET': list(),
    'mean_hd': list()
}

folder = r"F:\ddpmMed\Experiments\brats\128x128"
for seed in seeds:

    # create folder
    base_folder = os.path.join(folder, f"seed_{seed}")
    os.makedirs(base_folder, exist_ok=True)

    # validation results directory
    val_folder = os.path.join(base_folder, "validation")
    os.makedirs(val_folder, exist_ok=True)

    # data-split
    train, _ = random_split(dataset=train_pool, lengths=[80, 177],
                            generator=torch.Generator().manual_seed(seed))

    # Dump training data to visualize the content of the split
    dump_brats_dataset(dataset=train, dump_folder=base_folder)

    # train classifier    
    x_data, y_data = prepare_brats_pixels(data=train, feature_extractor=feature_extractor, image_size=128,
                                          num_features=256)
    pixel_dataset = PixelDataset(x_data=x_data, y_data=y_data)
    pixel_dataloader = DataLoader(dataset=pixel_dataset, batch_size=32, shuffle=True)
    ensemble = Ensemble(in_features=256, num_classes=4, size=10, init_weights="normal")
    if seed == 12:
        ensemble.load_ensemble(ensemble_folder=r"F:\ddpmMed\Experiments\brats\128x128\seed_12\ensemble")
    else:
        ensemble.train(epochs=8, data=pixel_dataloader, cache_folder=base_folder)

    # calculate and save on validation set
    metrics = SegmentationMetrics(num_classes=4, include_background=False)
    all_metrics = {
        'jaccard': list(), 'dice': list(), 'fp_error': list(), 'fn_error': list()
    }
    with tqdm(enumerate(test), total=len(test)) as pbar:
        for i, (image, mask) in pbar:
            if len(torch.unique(mask)) > 1:
                # predict on current image
                features = scale_features(feature_extractor(image), size=128)
                features = features.reshape(256, (128 * 128)).T
                pred = ensemble.predict(features.cpu()).reshape(128, 128)

                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(torch2np(pred))
                ax[0].set_axis_off()
                ax[0].set_title("Predictions")

                ax[1].imshow(torch2np(mask, squeeze=True))
                ax[1].set_axis_off()
                ax[1].set_title("Ground Truth")

                # compute all metrics
                mean_dice, dice_scores = metrics.dice_score(pred, mask)
                mean_hd, hd_distances = metrics.hausdorff_distance(pred, mask)

                all_dice_scores['mean_dice'].append(mean_dice)
                all_dice_scores['TC'].append(dice_scores[0][0])
                all_dice_scores['ED'].append(dice_scores[0][1])
                all_dice_scores['ET'].append(dice_scores[0][2])
                if mean_hd < 100:
                    all_hd_distances['mean_hd'].append(mean_hd)
                    all_hd_distances['TC'].append(hd_distances[0][0])
                    all_hd_distances['ED'].append(hd_distances[0][1])
                    all_hd_distances['ET'].append(hd_distances[0][2])
                    fig.suptitle(f"Mean Dice: {mean_dice:.4f}\n"
                                 f"Tumor Core: {dice_scores[0][0]:.4f}\n"
                                 f"Invaded Tissue: {dice_scores[0][1]:.4f}\n"
                                 f"Enhancing Tumor: {dice_scores[0][2]:.4f}\n"
                                 f"Mean HD: {mean_hd:.4f}")
                else:
                    fig.suptitle(f"Mean Dice: {mean_dice:.4f}\n"
                                 f"Tumor Core: {dice_scores[0][0]:.4f}\n"
                                 f"Invaded Tissue: {dice_scores[0][1]:.4f}\n"
                                 f"Enhancing Tumor: {dice_scores[0][2]:.4f}\n"
                                 f"Mean HD: Infinity")
                plt.savefig(os.path.join(val_folder, f"prediction_{i}.jpeg"), dpi=300)
                plt.close()
    print(f"Dice mean: {np.mean(all_dice_scores['mean_dice'])}")
    print(f"Dice mean for TC: {np.nanmean(all_dice_scores['TC'])}")
    print(f"Dice mean for ED: {np.nanmean(all_dice_scores['ED'])}")
    print(f"Dice mean for ET: {np.nanmean(all_dice_scores['ET'])}")

    print(f"HD mean: {np.nanmean(all_hd_distances['mean_hd'])}")
    print(f"HD mean for TC: {np.nanmean(all_hd_distances['TC'])}")
    print(f"HD mean for ED: {np.nanmean(all_hd_distances['ED'])}")
    print(f"HD mean for ET: {np.nanmean(all_hd_distances['ET'])}")
