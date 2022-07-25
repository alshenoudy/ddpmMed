import os
import json
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from ddpmMed.data.datasets import SegmentationDataset
from ddpmMed.insights.metrics_update import BraTSMetrics
from ddpmMed.insights.plots import plot_result
from ddpmMed.utils.data import scale_features, torch2np
from ddpmMed.core.pixel_classifier import Classifier
from monai.losses import DiceCELoss
import SimpleITK as sitk


class DiffusionMLPEnsemble:
    """
    A Diffusion based MLP ensemble


    Uses learned representations from a trained diffusion model scaled to the same input image size
    and learns a mapping between representations and target labels on an pixel level

    :param time_steps: a list with time-steps to sample an input image to
    :param layers: a list containing layer IDs to extract representations from
    :param trained_diffusion_model: a path to a Pytorch trained diffusion model
    :param configuration: a dictionary holding the diffusion model configuration for loading
    :param num_classes: integer with number of target classes
    :param ensemble_size: integer with total size of ensemble
    :param init_weights: a string indicating weight initialization method, can be ['random', 'kaiming', 'normal',
                        'xavier', 'orthogonal']
    :param device: device to move ensemble to
    :param cache_dir: a string pointing to a directory to store intermediate results
    """

    def __init__(self,
                 time_steps: list,
                 layers: list,
                 trained_diffusion_model: str,
                 configuration: dict,
                 num_classes: int,
                 ensemble_size: int,
                 init_weights: str,
                 device: str = 'cpu',
                 cache_dir: str = os.getcwd(),
                 architecture: str = 'simple') -> None:

        self.architecture = architecture
        self.time_steps = time_steps
        self.layers = layers
        self.diffusion_model_path = trained_diffusion_model
        self.configuration = configuration
        self.in_features = 0
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        self.init_weights = init_weights.lower()
        self.cache_dir = cache_dir
        self.device = device
        self.ensemble = OrderedDict()
        self.ensemble_metadata = OrderedDict()
        self.ensemble_path = None
        self.softmax = nn.Softmax(dim=1)

        # other constants/params
        _init_functions = ['normal', 'xavier', 'kaiming', 'orthogonal']

        # checks/validations
        if not os.path.exists(self.cache_dir):
            raise FileExistsError(f"cache directory [{self.cache_dir}] does not exist!")
        else:
            self.cache_dir = os.path.join(self.cache_dir, "Diffusion MLP Ensemble")
            os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Using {self.cache_dir} as a cache directory ..\n")

        # create an ensemble folder
        self.ensemble_path = os.path.join(self.cache_dir, f"Ensemble")
        os.makedirs(self.ensemble_path, exist_ok=True)

        # define and create feature extractor
        self.feature_extractor = FeatureExtractorDDPM(steps=self.time_steps,
                                                      blocks=self.layers,
                                                      model_path=self.diffusion_model_path,
                                                      config=self.configuration)

        print(f"Created Feature Extractor from Diffusion Model\n"
              f"Using layers: {self.layers} and time-steps: {self.time_steps}")

        print(f"\n{'==' * 25}"
              f"\nCalculating number of features from layers ..")

        # random image to acquire in_features based on layer channels
        self.in_channels = self.configuration.get("in_channels")
        self.image_size = self.configuration.get("image_size")
        _image = torch.randn(size=(1,
                                   self.in_channels,
                                   self.image_size,
                                   self.image_size)).to(self.feature_extractor.model.device)

        _features = self.feature_extractor(_image)
        _features = scale_features(_features, size=self.configuration.get("image_size"))
        self.in_features = _features.shape[0]

        # remove temp variables
        del _features, _image
        print(f"Calculated total number of features, num_features = {self.in_features}")

        print(f"\n{'==' * 25}"
              f"\nCreating MLP Ensemble with in_features = {self.in_features} and num_classes = {self.num_classes}")
        # define and create ensemble
        for i in range(1, self.ensemble_size + 1):
            # init weights
            if self.init_weights == "random":
                initialization = np.random.choice(_init_functions)
            else:
                initialization = self.init_weights

            # create classifier
            self.ensemble[i] = Classifier(in_features=self.in_features,
                                          num_classes=self.num_classes,
                                          architecture=self.architecture).to(self.device)

            # init weights and save metadata
            self.ensemble[i].init_weights(init_type=initialization)
            self.ensemble_metadata[i] = {
                'classifier_id': i,
                'model': self.ensemble[i].__module__,
                'initialization_method': initialization,
                'input_features': self.in_features,
                'num_classes': num_classes,
                'device': self.device,
                'trained': False,
                'epoch': 0
            }

        # save untrained ensemble
        torch.save(
            obj=self.ensemble,
            f=os.path.join(self.ensemble_path, f"ensemble.pt")
        )
        with open(os.path.join(self.ensemble_path, f"ensemble_metadata.json"), 'w') as jf:
            json.dump(self.ensemble_metadata, jf)
        jf.close()

        # save current classifier/ update ensemble
        torch.save(
            obj=self.ensemble,
            f=os.path.join(self.ensemble_path, f"ensemble.pt")
        )
        print(f"Created MLP Ensemble with {self.ensemble_size} MLPs")

    def load_pretrained(self, path: str):
        """
        Loads a pretrained ensemble
        Args:
            path (str): string pointing to ensemble folder

        Returns: None, loads trained ensemble from disk
        """
        self.ensemble_path = path
        ensemble_metadata = os.path.join(self.ensemble_path, 'ensemble_metadata.json')

        # load ensemble metadata
        with open(ensemble_metadata, 'r') as jf:
            ensemble_metadata = json.load(jf)
            self.ensemble_metadata = ensemble_metadata
        jf.close()

        classifier_paths = [os.path.join(self.ensemble_path, f"classifier_{i}.pt")
                            for i in range(1, self.ensemble_size + 1)]
        classifier_paths = iter(classifier_paths)
        for i in range(1, self.ensemble_size + 1):
            self.ensemble[i].load_state_dict(
                torch.load(f=next(classifier_paths),
                           map_location=self.device)['model'])
        print(f"Loaded pretrained ensemble located at {path}..\n\n")

    def train(self,
              train_data: DataLoader,
              valid_data: DataLoader = None,
              epochs: int = 8,
              lr: float = 0.0001,
              validate_every: int = 1,
              use_dice_loss: bool = True,
              ce_weight: torch.Tensor = None,
              lambda_dice: float = 1.,
              lambda_ce: float = 1.,
              use_softmax: bool = True,
              use_sigmoid: bool = False,
              include_background: bool = False
              ) -> None:
        """
        Trains the Diffusion MLP object
        """
        if use_dice_loss:
            criterion = DiceCELoss(
                include_background=include_background,
                to_onehot_y=True,
                sigmoid=use_sigmoid,
                softmax=use_softmax,
                ce_weight=ce_weight,
                lambda_dice=lambda_dice,
                lambda_ce=lambda_ce
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=ce_weight,
                ignore_index=-100 if include_background else 0)

        print(f"Training all Ensemble classifiers ..\n"
              f"{'==' * 25}\n")
        for i, classifier in self.ensemble.items():

            # define optimizer
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)

            # add losses to metadata dict
            self.ensemble_metadata[i]['train_losses'] = list()
            self.ensemble_metadata[i]['validation_losses'] = list()

            print(f"\n\nStarted training classifier [{i}] ..")
            with tqdm(range(0, epochs), postfix={"loss": "calculating .."}, leave=False) as pbar:

                # epoch train/val losses
                training_losses = list()
                running_losses_tr = list()
                validation_losses = list()
                running_losses_val = list()

                for e in pbar:
                    # one training step/epoch through entire data
                    pbar.set_description("Training")
                    for x, y in train_data:
                        optimizer.zero_grad()
                        predictions = classifier(x)
                        if use_dice_loss:
                            y = y.unsqueeze(-1)
                        loss = criterion(predictions, y)
                        loss.backward()
                        optimizer.step()

                        # log to console/progress bar
                        running_losses_tr.append(loss.item())

                        pbar.set_postfix(
                            {"loss": f"{running_losses_tr[-1]:.8f}"}
                        )
                    training_losses.append((e, np.mean(running_losses_tr)))

                    # one validation step based on validate_every
                    if validate_every > 0:
                        if valid_data is not None:
                            if e != 0 and e % validate_every == 0:
                                pbar.set_description("Validating")
                                with torch.no_grad():
                                    for x, y in valid_data:
                                        predictions = classifier(x)
                                        if use_dice_loss:
                                            y = y.unsqueeze(-1)
                                        loss = criterion(predictions, y)
                                        running_losses_val.append(loss.item())
                                        pbar.set_postfix(
                                            {"loss": f"{running_losses_val[-1]:.8f}"}
                                        )
                                    validation_losses.append((e, np.mean(running_losses_val)))

                    # append to metadata for debugging/ reloading
                    self.ensemble_metadata[i]['epoch'] = e + 1

            # add train/val losses to ensemble_metadata
            self.ensemble_metadata[i]['train_losses'].append(training_losses)
            self.ensemble_metadata[i]['validation_losses'].append(validation_losses)
            self.ensemble_metadata[i]['trained'] = True

            # write over ensemble metadata
            with open(os.path.join(self.ensemble_path, f"ensemble_metadata.json"), 'w') as jf:
                json.dump(self.ensemble_metadata, jf)
            jf.close()

            # save classifier/ensemble
            torch.save(
                obj={
                    'model': self.ensemble[i].state_dict()
                },
                f=os.path.join(self.ensemble_path, f"classifier_{i}.pt")
            )
        print(f"\n\nFinished training Ensemble \n\n")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict on an input image/tensor. First we extract features and do a voting
        over all the pixels predicted from all classifiers
        """

        # Extract features
        features = scale_features(self.feature_extractor(x), size=self.image_size)
        features = features.reshape(self.in_features, (self.image_size * self.image_size)).T
        features = features.to(self.device)

        # Predict from features
        x_pred = [torch.argmax(self.softmax(classifier(features)), dim=1) for i, classifier in self.ensemble.items()]
        x_pred = torch.stack(x_pred)
        x_pred = torch.mode(x_pred, dim=0)[0]
        x_pred = x_pred.reshape(self.image_size, self.image_size)

        # delete features tensor
        del features

        return x_pred

    def predict_and_export(self,
                           data: SegmentationDataset,
                           indices: list,
                           plot_predictions: bool = True,
                           export_predictions: bool = True,
                           save_to: str = None):
        """
        Predicts all items in a Dataset and exports them, optionally also plot them
        Args:
            export_predictions: Flag to export .nii.gz predictions
            plot_predictions: Flag to plot predictions or not, against ground truth
            data:
            indices:
            save_to:
        Returns:
        """

        total_items = len(indices)
        indices = iter(indices)
        metrics = BraTSMetrics()
        metrics_results = OrderedDict()
        running_metrics = None

        if export_predictions:
            os.makedirs(os.path.join(save_to, 'nifti_predictions'), exist_ok=True)

        with tqdm(enumerate(indices), total=total_items) as pbar:
            for i, idx in pbar:
                x, y = data[idx]
                filename = os.path.basename(data.dataset[idx]['mask'])
                filename = filename.split('.tif')[0]
                filename = filename.split('_')
                filename = f"BraTS_{int(filename[1]):05d}s{int(filename[2]):03d}"
                pbar.set_description(f"Evaluating {filename}")

                pred = self.predict(x)

                # plot/save predictions, not saving saves a lot of disk space
                if plot_predictions:
                    plot_result(
                        prediction=pred,
                        ground_truth=y,
                        palette=[
                            45, 0, 55,  # 0: Background
                            20, 90, 139,  # 1: Non Enhancing (BLUE)
                            22, 159, 91,  # 2: Tumor Core (GREEN)
                            255, 232, 9  # 3: Enhancing Tumor (YELLOW)
                        ],
                        file_name=os.path.join(save_to, f"{i:05d}.jpeg")
                    )
                if export_predictions:
                    image = torch2np(pred).astype(np.uint8)
                    image = sitk.JoinSeries(sitk.GetImageFromArray(image))
                    image.SetOrigin([0, 0, 0])
                    image.SetSpacing([1, 1, 999])
                    sitk.WriteImage(image=image, fileName=os.path.join(save_to,
                                                                       'nifti_predictions',
                                                                       f"{filename}.nii.gz"))

