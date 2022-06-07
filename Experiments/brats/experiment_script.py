import os
from typing import Callable
import torch
import json
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from ddpmMed.insights.plots import plot_result, plot_debug
from ddpmMed.core.pixel_classifier import Ensemble
from Experiments.config import brats_128x128_config
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from ddpmMed.data.datasets import SegmentationDataset, PixelDataset
from ddpmMed.insights.metrics import SegmentationMetrics
from ddpmMed.utils.palette import get_palette
from ddpmMed.utils.data import dump_brats_dataset, scale_features, prepare_brats_pixels, balance_labels
from ddpmMed.utils.helpers import metrics2str
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
                     use_val: bool = False,
                     batch_size: int = 32,
                     num_classes: int = 4,
                     ensemble_size: int = 10,
                     initialization: str = "normal",
                     trained_ensemble: list = None,
                     label_names: list = None,
                     process_labels: Callable = None,
                     plot_results: bool = True,
                     device: str = 'cuda'):
    """
    Runs an experiment using Brats dataset in 2D using a trained diffusion
    model
    """
    # check cache directory
    if label_names is None:
        label_names = ["TC", "IT", "ET"]
    results_folder = config.get("cache_dir")
    if results_folder is None:
        raise ValueError("expected 'cache_dir' to be a string but got None")

    results_folder = os.path.join(results_folder, f"{train_size} samples - t{time_steps[0]}")
    os.makedirs(results_folder, exist_ok=True)

    # palette for plotting
    if len(label_names) > 1:
        p = get_palette('brats')
    else:
        p = get_palette('binary')

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
                                      device=device,
                                      process_labels=process_labels)

        feature_extractor = FeatureExtractorDDPM(steps=time_steps,
                                                 blocks=blocks,
                                                 model_path=model_dir,
                                                 config=config)

        model, diffusion = feature_extractor.get_model_and_diffusion()

        # dataset split to a training pool to sample training data from and a test set
        training_pool, test = random_split(dataset=dataset, lengths=[train_pool, test_size],
                                           generator=torch.Generator().manual_seed(42))

        train, val = random_split(dataset=training_pool, lengths=[train_size, train_pool - train_size],
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

        # create a pixel dataset/ dataloader
        pixel_dataset = PixelDataset(x_data=x_data, y_data=y_data, device='cpu')
        pixel_dataloader = DataLoader(dataset=pixel_dataset, batch_size=batch_size, shuffle=True)

        # create ensemble
        ensemble = Ensemble(in_features=num_features,
                            num_classes=num_classes,
                            size=ensemble_size,
                            init_weights=initialization,
                            device='cpu')

        # load trained weights if passed
        if trained_ensemble is not None:
            ensemble.load_ensemble(ensemble_folder=trained_ensemble[i])
        else:
            ensemble.train(epochs=epochs, data=pixel_dataloader, cache_folder=seed_folder)

        # compute metrics for this split on test data
        metrics = SegmentationMetrics(num_classes=num_classes, include_background=False)

        # dictionary to save all metrics
        all_metrics = {}
        with tqdm(enumerate(val if use_val else test), total=len(val) if use_val else test_size) as pbar:
            for j, (image, mask) in pbar:

                # test non empty masks
                if len(torch.unique(mask)) > 1:
                    # image name for further reference
                    image_name = os.path.basename(dataset.dataset[test.indices[j]]["image"])

                    # Predict on image
                    features = scale_features(feature_extractor(image), size=image_size)
                    features = features.reshape(num_features, (image_size * image_size)).T
                    pred = ensemble.predict(features.cpu()).reshape(image_size, image_size)

                    #  check model and x_start prediction from noisy input
                    time = torch.tensor(time_steps[0]).cuda()
                    time = time.unsqueeze(0)
                    image_noisy = diffusion.q_sample(x_start=image.unsqueeze(0), t=time, noise=None)
                    image_denoised = diffusion.p_sample(model=model, x=image_noisy, t=time)
                    image_denoised = image_denoised['pred_xstart']


                    # calculate metrics
                    mean_scores, scores = metrics.get_all_metrics(prediction=pred, ground_truth=mask)

                    if plot_results:
                        # Caption figure
                        caption = metrics2str(mean_scores=mean_scores,
                                              scores=scores,
                                              labels=label_names)
                        # print(f"prediction: {torch.unique(pred)}, gt: {torch.unique(mask)}\n\n")
                        # plot_result(prediction=pred, ground_truth=mask, palette=None,
                        #             file_name=os.path.join(seg_folder, f"{j:5d}.jpeg"),
                        #             caption=caption,
                        #             fontsize=5)

                        plot_debug(prediction=pred,
                                   mask=mask,
                                   images=(image,
                                           image_noisy.squeeze(0),
                                           image_denoised.squeeze(0)),
                                   caption=None,
                                   file_name=os.path.join(seg_folder, f"{j:5d}.jpeg"),
                                   fontsize=5
                                   )
                    if len(label_names) > 1:
                        all_metrics[image_name] = {m: {l: scores[m][i].item() for i, l in enumerate(label_names)}
                                                   for m in scores.keys()}
                    else:
                        all_metrics[image_name] = {m: {l: scores[m].item() for i, l in enumerate(label_names)}
                                                   for m in scores.keys()}

            # Save metrics to a JSON file
            with open(os.path.join(seed_folder, "calculated_metrics.json"), 'w') as jf:
                json.dump(all_metrics, jf)
            jf.close()
