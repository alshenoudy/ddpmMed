import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, architecture: str = 'simple') -> None:
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.architecture = architecture

        if architecture is None:
            self.architecture = 'simple'
        elif self.architecture.lower() not in ['simple', 'wide', 'deep', 'deep_wide']:
            raise ValueError(f"unknown architecture type: {self.architecture}, should be one of"
                             f" ['simple', 'wide', 'deep']")

        self.simple = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, num_classes))

        self.wide = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, num_classes))

        self.deep = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, num_classes)
        )

        self.deep_wide = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, self.num_classes)
        )

        if self.architecture == 'simple':
            self.model = self.simple
        elif self.architecture == 'wide':
            self.model = self.wide
        elif self.architecture == 'deep':
            self.model = self.deep
        elif self.architecture == 'deep_wide':
            self.model = self.deep_wide

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
