import os
import json
import torch
from typing import Callable, Tuple
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from ddpmMed.core.mlp_ensemble import DiffusionMLPEnsemble
from ddpmMed.data.datasets import SegmentationDataset, PixelDataset
from Experiments.config import brats_128x128_config
from ddpmMed.insights.evaluator import NiftiEvaluator
from ddpmMed.utils.data import dump_brats_dataset, prepare_brats_pixels


def mlp_experiment(
        configuration: dict,
        output_dir: str,
        model_dir: str,
        images_dir: str,
        labels_dir: str,
        time_steps: list,
        layers: list,
        seeds: Tuple,
        train_size: int,
        validation_size: int,
        test_size: int,
        train_pool: int,
        epochs: int,
        validate_every: int,
        learning_rate: float,
        evaluate_val: bool,
        evaluate_test: bool,
        plot_predictions: bool,
        batch_size: int,
        num_classes: int,
        ensemble_size: int,
        architecture: str,
        initialization: str,
        evaluate_test_nifti_folder: str,
        load_trained_ensemble: str = None,
        label_names: dict = None,
        map_labels: Callable = None,
        device: str = 'cpu',
        dump_train: bool = True,
        dump_val: bool = True,
        dump_test: bool = False,
        label_weights=None) -> None:

    """
    Runs a Diffusion MLP Ensemble on a given configuration/data split and saves results to disk.

    Args:
        learning_rate:
        ensemble_size:
        configuration (dict):
            a dictionary with all settings for loading the diffusion model.
        output_dir (str):
            a string like a path pointing to a directory to store results.
        model_dir (str):
            a string pointing to a .pt trained diffusion model.
        images_dir (str):
            a string pointing to images directory.
        labels_dir (str):
            a string pointing to labels directory.
        time_steps (list):
            a list containing integers of time-steps that samples input images to using the diffusion model.
        layers (list):
            list of all intermediate layers from the denoise function to use to extract representations from.
        seeds (tuple):
            Two integers used as seeds for train/validation/test splits for reproducible results, first seed is used
            to create a training pool and a test set, where the second is used to extract train/validations sets.
        train_size (int):
            Integer representing the training data split size.
        validation_size (int):
            Integer representing the validation data split size.
        test_size (int):
            Integer representing the test data split size.
        train_pool (int):
            Integer representing the training data pool split size. Training images are later sampled from this pool.
        epochs (int):
            Integer indicating epochs to train on training set per classifier in an ensemble.
        evaluate_val (bool):
            Boolean indicating to evaluate the validation set or not.
        evaluate_test (bool):
            Boolean indicating to evaluate the test set or not.
        batch_size (int):
            Integer indicating the batch size for the data-loaders during training.
        num_classes (int):
            Integer representing number of classes the MLP/classifier has to predict.
        initialization (str):
            String to indicate the type of layer weight initialization.
        load_trained_ensemble (str):
            To load a pre-train ensemble or not, when a string is given it directly loads it if valid.
        label_names (dict):
            Optional dictionary to map label IDs to string names.
        map_labels (Callable):
            A function to map the dataset labels.
        device (str):
            String indicating to use cuda or cpu, defaults to 'cpu'.
        dump_test (bool):
            To save a copy of the test data or not for debugging/visual inspection, defaults to False.
        dump_val (bool):
            To save a copy of the validation data or not for debugging/visual inspection, defaults to True.
        dump_train (bool):
            To save a copy of the training data or not for debugging/visual inspection, defaults to True.
    """

    # check output directory
    if output_dir is not None:
        if not os.path.exists(output_dir):
            raise FileExistsError(f"output_dir {output_dir} does not exist!")
    else:
        print(f"output_dir is None, using current directory to save results ..")
        output_dir = os.getcwd()

    if label_weights is None and num_classes == 4:
        label_weights = torch.tensor([0.1, 0.3, 0.3, 0.3])

    # different seeds for data splits
    seed_0, seed_1 = seeds

    # define experiment name and create experiment folder
    experiment_name = f"MLP Experiment - {train_size} Samples"
    output_dir = os.path.join(output_dir, experiment_name, f"Experiment Seed {seed_1}")
    os.makedirs(output_dir, exist_ok=True)

    # create and write experiment configuration
    exp_config = {
        'output_dir': output_dir,
        'diffusion_model': model_dir,
        'images': images_dir,
        'labels': labels_dir,
        'time_steps': time_steps,
        'layers': layers,
        'seeds': seeds,
        'training_size': train_size,
        'testing_size': test_size,
        'validation_size': validation_size,
        'size_train_pool': train_pool,
        'mlp_epochs': epochs,
        'batch_size': batch_size,
        'classes': num_classes,
        'architecture': architecture,
        'initialization': initialization,
        'load_pretrained_ensemble': load_trained_ensemble,
        'evaluate_validation_set': evaluate_val,
        'evaluate_test_set': evaluate_test,
        'label_names': label_names,
        'map_labels': map_labels,
        'device': device
    }

    with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as jf:
        json.dump(exp_config, jf)
    jf.close()

    # create and split dataset
    dataset = SegmentationDataset(images_dir=images_dir,
                                  masks_dir=labels_dir,
                                  image_size=configuration.get('image_size'),
                                  device='cuda',
                                  process_labels=map_labels)
    total_data = len(dataset)
    if total_data != (train_pool + test_size):
        raise RuntimeError(f"Training pool size and test size should match total dataset size "
                           f"{total_data} != {train_pool} + {test_size}")

    training_pool_data, test_data = random_split(dataset=dataset, lengths=[train_pool, test_size],
                                                 generator=torch.Generator().manual_seed(seed_0))

    expected_val_size = train_pool - train_size
    resample_val = False
    if expected_val_size != validation_size:
        RuntimeWarning(f"Expected validation size is not the same as passed validation size ..")
        if validation_size > expected_val_size:
            validation_size = expected_val_size
            print(f"Validation size ({validation_size}) greater than expected validation size ({expected_val_size}),"
                  f" thresholding to available data ..")
        else:
            print(f"Validation size ({validation_size}) less than expected validation size ({expected_val_size}),"
                  f"resampling will be done from validation pool ..")
            resample_val = True

    training_data, validation_data = random_split(dataset=training_pool_data, lengths=[train_size, expected_val_size],
                                                  generator=torch.Generator().manual_seed(seed_1))
    if resample_val:
        validation_data, _ = random_split(dataset=validation_data, lengths=[validation_size,
                                                                            (expected_val_size - validation_size)],
                                          generator=torch.Generator().manual_seed(seed_1))

    print(f"\n"
          f"Dataset divided into:\n"
          f"{'=' * 30}\n"
          f"Training Pool Size: {len(training_pool_data)}\n"
          f"Training Dataset Size: {len(training_data)}\n"
          f"Test Dataset Size: {len(test_data)}\n"
          f"Validation Dataset Size: {len(validation_data)}\n\n")

    # Dump datasets for inspection
    if dump_train and len(training_data) > 0:
        print(f"Saving training data ..")
        dump_brats_dataset(dataset=training_data, dump_folder=os.path.join(output_dir, "Dataset", "Training Data"))
        print(f"Training data copy saved to {os.path.join(output_dir, f'Dataset', 'Training Data')}")

    if dump_val and len(validation_data) > 0:
        print(f"Saving validation data ..")
        dump_brats_dataset(dataset=validation_data, dump_folder=os.path.join(output_dir, "Dataset", "Validation Data"))
        print(f"Validation data copy saved to {os.path.join(output_dir, f'Dataset', 'Validation Data')}")

    if dump_test and len(test_data) > 0:
        print(f"Saving test data ..")
        dump_brats_dataset(dataset=test_data, dump_folder=os.path.join(output_dir, "Dataset", "Test Data"))
        print(f"Test data copy saved to {os.path.join(output_dir, f'Dataset', 'Test Data')}")

    # Save split IDs to a json file
    training_files, validation_files, test_files = [], [], []

    for index in training_data.indices:
        file_name = os.path.basename(dataset.dataset[index]["mask"])
        training_files.append(file_name)

    if len(validation_data) > 0:
        for index in validation_data.indices:
            file_name = os.path.basename(dataset.dataset[index]["mask"])
            validation_files.append(file_name)

    if len(test_data) > 0:
        for index in test_data.indices:
            file_name = os.path.basename(dataset.dataset[index]["mask"])
            test_files.append(file_name)

    # Create dictionary for split
    data_split_info = {
        'train':
            {
                'ids': training_data.indices,
                'files': training_files
            },
        'validation':
            {
                'ids': validation_data.indices,
                'files': validation_files
            },
        'test':
            {
                'ids': test_data.indices,
                'files': test_files
            }
    }

    # write dataset splits to a json file
    with open(os.path.join(output_dir, f"dataset.json"), 'w') as jf:
        json.dump(data_split_info, jf)
    jf.close()

    # Create an Ensemble
    mlp_ensemble = DiffusionMLPEnsemble(
        time_steps=time_steps,
        layers=layers,
        trained_diffusion_model=model_dir,
        configuration=configuration,
        num_classes=num_classes,
        ensemble_size=ensemble_size,
        init_weights=initialization,
        device=device,
        cache_dir=output_dir,
        architecture=architecture
    )

    # Create pixel dataset
    x_train_data, y_train_data = prepare_brats_pixels(data=training_data,
                                                      feature_extractor=mlp_ensemble.feature_extractor,
                                                      image_size=mlp_ensemble.image_size,
                                                      num_features=mlp_ensemble.in_features)

    # Create pixel dataset
    x_valid_data, y_valid_data = prepare_brats_pixels(data=validation_data,
                                                      feature_extractor=mlp_ensemble.feature_extractor,
                                                      image_size=mlp_ensemble.image_size,
                                                      num_features=mlp_ensemble.in_features)

    pixel_train_data = PixelDataset(x_data=x_train_data, y_data=y_train_data, device=device)
    pixel_train_dataloader = DataLoader(dataset=pixel_train_data, batch_size=batch_size, shuffle=True)
    pixel_valid_data = PixelDataset(x_data=x_valid_data, y_data=y_valid_data, device=device)
    pixel_valid_dataloader = DataLoader(dataset=pixel_valid_data, batch_size=batch_size, shuffle=True)

    # train ensemble
    if load_trained_ensemble:
        mlp_ensemble.load_pretrained(path=load_trained_ensemble)
    else:
        mlp_ensemble.train(
            train_data=pixel_train_dataloader,
            valid_data=pixel_valid_dataloader,
            epochs=epochs,
            lr=learning_rate,
            validate_every=validate_every,
            use_dice_loss=False,
            include_background=True,
            ce_weight=label_weights)

    # after training is complete remove unnecessary objects/params
    del x_valid_data, y_valid_data
    del pixel_train_data, pixel_train_dataloader
    del pixel_valid_data, pixel_valid_dataloader

    # change transforms to test time
    dataset.set_test_transforms()

    # evaluate ensemble
    if evaluate_val:
        print(f"\nEvaluating Validation Data ..")
        pred_val_dir = os.path.join(output_dir, "Predictions", "Validation Data")
        os.makedirs(name=pred_val_dir, exist_ok=True)
        mlp_ensemble.predict_and_export(data=dataset,
                                        indices=data_split_info["validation"]["ids"],
                                        plot_predictions=plot_predictions,
                                        save_to=pred_val_dir)

        print(f"Evaluated Validation Data\n\n")

    if evaluate_test:
        print(f"\nEvaluating Test Data ..")
        pred_test_dir = os.path.join(output_dir, "Predictions", "Test Data")
        os.makedirs(name=pred_test_dir, exist_ok=True)
        mlp_ensemble.predict_and_export(data=dataset,
                                        indices=data_split_info["test"]["ids"],
                                        plot_predictions=plot_predictions,
                                        save_to=pred_test_dir)
        if evaluate_test_nifti_folder is not None:
            if os.path.exists(evaluate_test_nifti_folder):
                nifti_evaluator = NiftiEvaluator(
                    predictions=os.path.join(pred_test_dir, 'nifti_predictions'),
                    references=evaluate_test_nifti_folder,
                    labels=label_names)
                nifti_evaluator.evaluate_folders(output_dir=pred_test_dir)
        print(f"Evaluated Test Data\n\n")




