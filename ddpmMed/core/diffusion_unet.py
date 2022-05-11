import torch
from torch import nn
from ddpmMed.scripts.script_utils import load_model


class DiffusionUNet(nn.Module):
    """
    Uses a trained noise predictor network from a diffusion model
    to predict directly on labels, where the generative process is
    used as an initialization to the network's weights.
    """
    def __init__(self, model_path: str):
        super(DiffusionUNet, self).__init__()
        self.model_path = model_path

        # TODO: implement the sampling, and training
