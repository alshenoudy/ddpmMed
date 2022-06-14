import os
import sys
import json
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from ddpmMed.core.feature_extractor import FeatureExtractorDDPM
from ddpmMed.utils.data import scale_features
from ddpmMed.core.pixel_classifier import Classifier


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
                 cache_dir: str = os.getcwd()) -> None:

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
            self.cache_dir = os.path.join(self.cache_dir, "diffusion_ensemble")
            os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Using {self.cache_dir} as a cache directory ..\n")

        # create an ensemble folder
        self.ensemble_path = os.path.join(self.cache_dir, f"MLP Ensemble")
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
                                          num_classes=self.num_classes).to(self.device)

            # init weights and save metadata
            self.ensemble[i].init_weights(init_type=initialization)
            self.ensemble_metadata[i] = {
                'classifier_id': i,
                'model': self.ensemble[i].__module__,
                'initialization_method': initialization,
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
        print(f"Created MLP Ensemble with {self.ensemble_size} MLPs")

    def train(self,
              train_data: DataLoader,
              valid_data: DataLoader = None,
              epochs: int = 8,
              lr: float = 0.0001,
              validate_every: int = 1,
              use_dice_loss: bool = True,
              dice_loss_weight: float = 0.95,
              ignore_background: bool = True
              ) -> None:
        """
        Trains the Diffusion MLP object
        """
        if use_dice_loss:
            criterion = None
        elif ignore_background:
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            criterion = nn.CrossEntropyLoss()

        print(f"Training all Ensemble classifiers ..\n"
              f"{'==' * 25}\n")
        for i, classifier in self.ensemble.items():

            # define optimizer
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)

            # add losses to metadata dict
            self.ensemble_metadata[i]['train_losses'] = list()
            self.ensemble_metadata[i]['validation_losses'] = list()

            print(f"Started training classifier [{i}] ..\n")
            with tqdm(range(0, epochs), postfix={"loss": "calculating .."}) as pbar:

                # epoch train/val losses
                training_losses = list()
                validation_losses = list()

                for e in pbar:
                    # one training step/epoch through entire data
                    pbar.set_description("Training")
                    for x, y in train_data:
                        optimizer.zero_grad()
                        predictions = classifier(x)
                        loss = criterion(predictions, y)
                        loss.backward()
                        optimizer.step()

                        # log to console/progress bar
                        training_losses.append(loss.item())
                        pbar.set_postfix(
                            {"loss": f"{training_losses[-1]:.8f}"}
                        )

                    # one validation step based on validate_every
                    if valid_data is not None:
                        if e != 0 and e % validate_every == 0:
                            pbar.set_description("Validating")
                            with torch.no_grad():
                                for x, y in valid_data:
                                    predictions = classifier(x)
                                    loss = criterion(predictions, y)
                                    validation_losses.append(loss.item())

                    # append to metadata for debugging/ reloading
                    self.ensemble[i]['epoch'] = e

            # add train/val losses to ensemble_metadata
            self.ensemble_metadata[i]['train_losses'].append(training_losses)
            self.ensemble_metadata[i]['validation_losses'].append(validation_losses)
            self.ensemble[i]['trained'] = True

            # save current classifier
            torch.save(obj=self.ensemble,
                       f=os.path.join(self.ensemble_path, f"ensemble.pt"))
            print(f"Saved classifier [{i}] to ensemble file located at:"
                  f" [{os.path.join(self.ensemble_path, f'ensemble.pt')}]")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict on an input image/tensor
        """
        features = scale_features(self.feature_extractor(x), size=self.image_size)
        features = features.reshape(self.in_features, (self.image_size, self.image_size)).T
        x_pred = [torch.argmax(self.softmax(c(features))) for i, c in self.ensemble.items()]
        x_pred = torch.stack(x_pred)
        x_pred = torch.mode(x_pred, dim=0)[0]

        # delete features tensor
        del features

        return x_pred
