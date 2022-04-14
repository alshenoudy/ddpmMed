import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, num_classes),
            nn.ReLU())

    def init_weights(self, init_type='normal', gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Ensemble:
    """
    An ensemble of classifiers
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 size: int = 10,
                 init_weights: str = "random") -> None:

        self.in_features = in_features
        self.num_classes = num_classes
        self.size = size
        self.init_weights = init_weights.lower()
        self.ensemble = []
        self.softmax = nn.Softmax(dim=1)
        init_functions = ['normal', 'xavier', 'kaiming', 'orthogonal']

        # create ensemble
        for i in range(0, self.size):

            # create nth classifier object
            classifier = Classifier(in_features=self.in_features,
                                    num_classes=self.num_classes)

            # initialize weights accordingly
            if self.init_weights == "random":
                classifier.init_weights(init_type=np.random.choice(init_functions))
            else:
                classifier.init_weights(init_type=self.init_weights)

            self.ensemble.append(classifier)
        print(f"Created ensemble with {self.size} classifiers\n")

    def load_ensemble(self, ensemble_folder: str):
        """ loads a pretrained ensemble from a directory """
        if not os.path.exists(ensemble_folder):
            raise FileExistsError(f" ensemble folder ({ensemble_folder}) does not exist")

        classifiers = os.listdir(ensemble_folder)
        classifiers = [os.path.join(ensemble_folder, c) for c in classifiers if c.split('.')[-1].lower() == 'pt']

        if len(classifiers) != self.size:
            raise RuntimeError(f"ensemble size and found classifiers do not match ({len(classifiers)} != {self.size})")
        for i in range(0, self.size):
            self.ensemble[i].load_state_dict(torch.load(classifiers[i]))

    def train(self,
              epochs: int,
              data: DataLoader,
              lr: float = 0.0001,
              cache_folder: str = os.getcwd()):
        """
        Trains each classifier in an ensemble, uses Adam as an optimizer
        """
        # define criterion and cache dir
        criterion = nn.CrossEntropyLoss()
        cache_folder = os.path.join(cache_folder, "ensemble")
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder, exist_ok=True)

        # Train each classifier in ensemble separately
        for i, classifier in enumerate(self.ensemble):

            # define optimizer for current classifier
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

            with tqdm(range(0, epochs), postfix={"batch_loss": "N/A"}) as pbar:
                pbar.set_description(f"Training Classifier [{i}]")
                for e in pbar:
                    for x, y in data:
                        optimizer.zero_grad()
                        predictions = classifier(x)
                        loss = criterion(predictions, y)
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix({
                            "batch_loss": "{:.6f}".format(loss.item())
                        })

                # save current model
                torch.save(obj=classifier.state_dict(), f=os.path.join(cache_folder, f"classifier_{i}.pt"))

    def predict(self, x: torch.Tensor):
        """
        Ensemble voting over pixels
        """
        x_pred = [torch.argmax(self.softmax(c(x)), dim=1) for c in self.ensemble]
        x_pred = torch.stack(x_pred)
        x_pred = torch.mode(x_pred, dim=0)[0]
        return x_pred
