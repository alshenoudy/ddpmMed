import torch
from torch import nn
from ddpmMed.diffusion.improved_ddpm import dist_util
from ddpmMed.scripts.script_utils import load_model
from ddpmMed.utils.helpers import save_inputs, save_outputs


class GenericFeatureExtractor(nn.Module):
    """ A generic base class for feature extractors """
    def __init__(self, model_path: str, input_activations: bool = False, **kwargs):
        super(GenericFeatureExtractor, self).__init__()
        self._load_pretrained_model(model_path, **kwargs)
        self.save_hook = save_inputs if input_activations else save_outputs
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        raise NotImplementedError


class FeatureExtractorDDPM(GenericFeatureExtractor):
    """ A feature extractor for ddpm pretrained models """

    def __init__(self, steps: list, blocks: list, **kwargs):
        super(FeatureExtractorDDPM, self).__init__(**kwargs)
        self.steps = steps
        self.blocks = blocks
        for idx, block in enumerate(self.model.output_blocks):
            if idx in self.blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path: str, **kwargs):
        self.model, self.diffusion = load_model(model_path=model_path, **kwargs)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        for t in self.steps:
            t = torch.tensor([t]).to(x.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            noisy_x = self.diffusion.q_sample(x_start=x, t=t, noise=noise)
            self.model(noisy_x, t)

            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        return activations

    def get_model_and_diffusion(self):
        return self.model, self.diffusion
