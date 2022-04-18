import torch
from torch import nn
from .feature_extractor import FeatureExtractorDDPM
from .pixel_classifier import Ensemble
from ..utils.data import scale_features


class DiffusionNet(nn.Module):
    def __init__(self,
                 image_size: int,
                 features: int,
                 feature_extractor: FeatureExtractorDDPM,
                 ensemble: Ensemble):
        super().__init__()
        self.ensemble = ensemble
        self.features = features
        self.image_size = image_size
        self.feature_extractor = feature_extractor

    def forward(self, x: torch.Tensor):
        """
        Takes an input image of shape [Batch_size, Channels, H, W]

        """
        batches = x.shape[1]
        features = self.feature_extractor(x)
        features = scale_features(activations=features, size=self.image_size)
        features = features.reshape(batches, self.features, (self.image_size * self.image_size))
        features = torch.permute(features, dims=(0, -1, 1))
        features = features.reshape((batches * self.image_size * self.image_size), self.features)
        predictions = self.ensemble.predict(features).reshape(batches, self.image_size, self.image_size)

        return predictions
