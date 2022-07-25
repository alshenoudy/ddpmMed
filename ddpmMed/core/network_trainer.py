"""
Mostly stolen and simplified from nnUNet base-class here:
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/network_training/network_trainer.py
"""

import os
import tqdm
import torch
import numpy as np
from time import time
from datetime import datetime
from abc import abstractmethod
import matplotlib.pyplot as plt


class NetworkTrainer:
    def __init__(self) -> None:
        """
        Generic class to train a Pytorch based network
        """

        # core parameters, objects and directories
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_function = None
        self.initialized_flag = False

        # dataset, dataloader(s), training and validation
        self.dataset = None
        self.dataset_directory = None
        self.data_train, self.data_valid = None, None

        # Training settings
        self.maximum_epochs = 500
        self.training_losses = []
        self.validation_losses = []
        self.evaluation_metrics = []
        self.epoch = 0
        self.log_file = None
        self.save_every = 25
        self.validate_every = 25

    @abstractmethod
    def initialize(self) -> bool:
        """
        Creates output folders, initializes networks and learning rate scheduler.

        Sets self.initialized_flag to True
        Returns (bool): True when successfully initiated network, lr_scheduler and optimizer
        """
        pass

    @abstractmethod
    def initialize_network(self) -> bool:
        """
        Initializes self.network and it's weights

        Returns (bool): True when successfully initiated network, False otherwise
        """
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self) -> bool:
        """
        Initializes the lr_scheduler and the optimizer

        Returns (bool): True when successfully initiated network, False otherwise
        """
        pass

    @abstractmethod
    def load_checkpoint(self):
        """
        Loads the network from a checkpoint file

        Returns:
        """
        pass

    def save_checkpoint(self, file_name: str, save_optimizer: bool = True) -> None:
        """

        Args:
            file_name: file name to save to self.output_folder
            save_optimizer: a boolean to save optimizer state dict or not

        Returns:

        """
        start_time = time()

        # get and move network state dict values to cpu
        network_state_dict = self.network.state_dict()
        network_state_dict = {key: value.cpu() for key, value in network_state_dict.items()}

        #












