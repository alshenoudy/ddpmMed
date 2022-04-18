import os
import torch
import json
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from ddpmMed.insights.plots import plot_result
from ddpmMed.core.pixel_classifier import Ensemble
from Experiments.config import brats_128x128_config
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from ddpmMed.data.datasets import SegmentationDataset, PixelDataset
from ddpmMed.insights.metrics import SegmentationMetrics
from ddpmMed.utils.palette import get_palette
from ddpmMed.utils.data import dump_brats_dataset, scale_features, prepare_brats_pixels, balance_labels

import warnings

warnings.filterwarnings('ignore')


def brats_experiment(config: dict,
                     model_dir: str,
                     images_dir: str,
                     masks_dir: str,
                     image_size: int,
                     time_steps: list,
                     blocks: list,
                     seeds: list,
                     train_size: int,
                     train_pool: int = 757,
                     test_size: int = 8000,
                     epochs: int = 8,
                     trained_ensemble: list = None,
                     device: str = 'cuda'
                     ):
    """
    Runs an experiment using Brats dataset in 2D using a trained diffusion
    model
    """
    # check cache directory
    results_folder = config.get("cache_dir")
    if results_folder is None:
        raise ValueError("expected 'cache_dir' to be a string but got None")

    results_folder = os.path.join(results_folder, "results")
    os.makedirs(results_folder, exist_ok=True)

    # palette for plotting
    p = get_palette('brats')

    for i, seed in enumerate(seeds):
        # create results folder
        seed_folder = os.path.join(results_folder, f"seed_{seed}")
        os.makedirs(results_folder, exist_ok=True)

        # create segmentation outputs folder
        seg_folder = os.path.join(seed_folder, "segmentation outputs")
        os.makedirs(seg_folder, exist_ok=True)

        # dataset and feature extractor
        dataset = SegmentationDataset(images_dir=images_dir,
                                      masks_dir=masks_dir,
                                      image_size=image_size,
                                      device=device)

        feature_extractor = FeatureExtractorDDPM(steps=time_steps,
                                                 blocks=blocks,
                                                 model_path=model_dir,
                                                 config=config)

        # dataset split to a training pool to sample training data from and a test set
        training_pool, test = random_split(dataset=dataset, lengths=[train_pool, test_size],
                                           generator=torch.Generator().manual_seed(42))

        train, _ = random_split(dataset=training_pool, lengths=[train_size, train_pool - train_size],
                                generator=torch.Generator().manual_seed(seed))

        # save training data for manual inspection
        dump_brats_dataset(dataset=train, dump_folder=seed_folder)

        # save a txt file for all file names used for training
        with open(os.path.join(seed_folder, "dataset indices.txt"), "w") as f:
            for idx in train.indices:
                img_name = os.path.basename(dataset.dataset[idx]["image"])
                msk_name = os.path.basename(dataset.dataset[idx]["mask"])
                f.write(f"index: {idx}\t\t\timage: {img_name},\t\tmask: {msk_name}\n")
            f.close()

        # get number of features
        image, mask = train[0]
        features = feature_extractor(image)
        features = scale_features(features, size=image_size)
        num_features = features.shape[0]

        x_data, y_data = prepare_brats_pixels(data=train, feature_extractor=feature_extractor,
                                              image_size=image_size, num_features=num_features)

        pixel_dataset = PixelDataset(x_data=x_data, y_data=y_data)
        pixel_dataloader = DataLoader(dataset=pixel_dataset, batch_size=32, shuffle=True)
        ensemble = Ensemble(in_features=num_features, num_classes=4, size=10, init_weights="normal")
        if trained_ensemble is not None:
            ensemble.load_ensemble(ensemble_folder=trained_ensemble[i])
        else:
            ensemble.train(epochs=epochs, data=pixel_dataloader, cache_folder=seed_folder)

        # compute metrics for this split on test data
        metrics = SegmentationMetrics(num_classes=4, include_background=False)

        # dictionary to save all metrics
        all_metrics = {}
        with tqdm(enumerate(test), total=test_size) as pbar:
            for j, (image, mask) in pbar:

                # test non empty masks
                if len(torch.unique(mask)) > 1:

                    # image name for further reference
                    image_name = os.path.basename(dataset.dataset[test.indices[j]]["image"])

                    # Predict on image
                    features = scale_features(feature_extractor(image), size=image_size)
                    features = features.reshape(num_features, (image_size * image_size)).T
                    pred = ensemble.predict(features.cpu()).reshape(image_size, image_size)

                    # calculate metrics
                    mean_scores, scores = metrics.get_all_metrics(prediction=pred, ground_truth=mask)

                    # Caption figure
                    caption = f"Dice Scores:\n" \
                              f"{'-' * 25}\n" \
                              f"mean: {mean_scores['mean_dice'].item():.3f} TC:{scores['dice'][0].item():.3f} " \
                              f"IT:{scores['dice'][1].item():.3f} and TC: {scores['dice'][0].item():.3f}\n\n" \
                              f"HD 95 Distances:\n" \
                              f"{'-' * 25}\n" \
                              f"mean: {mean_scores['mean_hd95'].item():.3f} TC:{scores['hd95'][0].item():.3f} " \
                              f"IT:{scores['hd95'][1].item():.3f} and TC: {scores['hd95'][0].item():.3f}\n\n" \
                              f"Jaccard Scores:\n" \
                              f"{'-' * 25}\n" \
                              f"mean: {mean_scores['mean_jaccard'].item():.3f} TC:{scores['jaccard'][0].item():.3f} " \
                              f"IT:{scores['jaccard'][1].item():.3f} and TC: {scores['jaccard'][0].item():.3f}"

                    plot_result(prediction=pred, ground_truth=mask, palette=p,
                                file_name=os.path.join(seg_folder, f"{j:5d}.jpeg"),
                                caption=caption,
                                fontsize=5)

                    all_metrics[image_name] = {
                        "dice": {
                            "TC": scores['dice'][0].item(),
                            "IT": scores['dice'][1].item(),
                            "ET": scores['dice'][2].item()

                        },

                        "hd95": {
                            "TC": scores['hd95'][0].item(),
                            "IT": scores['hd95'][1].item(),
                            "ET": scores['hd95'][2].item()
                        },

                        "jaccard": {
                            "TC": scores['jaccard'][0].item(),
                            "IT": scores['jaccard'][1].item(),
                            "ET": scores['jaccard'][2].item()
                        }
                    }

            # Save metrics to a JSON file
            with open(os.path.join(seed_folder, "calculated_metrics.json"), 'w') as jf:
                json.dump(all_metrics, jf)
            jf.close()


# run experiment
config = brats_128x128_config()
config["cache_dir"] = r"F:\diffusion\128x128 model"
brats_experiment(
    config=config,
    model_dir=r"F:\diffusion\128x128 model\model350000.pt",
    images_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\scans",
    masks_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\masks",
    image_size=128,
    time_steps=[100],
    blocks=[18, 19, 20, 21],
    seeds=[16],
    train_size=50,
    epochs=8,
)
