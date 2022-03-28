import torch
from torch import nn

from ddpmMed.diffusion.improved_ddpm import dist_util
from ddpmMed.scripts.script_utils import create_model_and_diffusion
from ddpmMed.utils.helpers import save_inputs, save_outputs


class GenericFeatureExtractor(nn.Module):
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
        dist_util.setup_dist()
        config = kwargs.get("config")
        self.model, self.diffusion = create_model_and_diffusion(
            image_size=config.get("image_size", 128),
            in_channels=config.get("in_channels", 3),
            class_cond=config.get("class_cond", False),
            learn_sigma=config.get("learn_sigma", False),
            num_channels=config.get("model_channels", False),
            num_res_blocks=config.get("num_resnet_blocks", 2),
            channel_mult=config.get("channel_mult", "1, 2, 3, 4"),
            num_heads=config.get("attention_heads", 4),
            num_head_channels=-1,
            num_heads_upsample=-1,
            attention_resolutions=config.get("attention_resolutions", None),
            dropout=config.get("dropout", 0),
            diffusion_steps=config.get("diffusion_steps", 1000),
            noise_schedule=config.get("noise_schedule", "cosine"),
            timestep_respacing=config.get("timestep_respacing", ""),
            use_kl=config.get("use_kl", False),
            predict_xstart=config.get("predict_xstart", False),
            rescale_timesteps=config.get("rescale_timesteps", False),
            rescale_learned_sigmas=config.get("rescale_learned_sigmas", False),
            use_checkpoint=config.get("use_checkpoint", False),
            use_scale_shift_norm=config.get("use_scale_shift_norm", False),
            resblock_updown=config.get("resblock_updown", False),
            use_fp16=config.get("use_fp16", False),
            use_new_attention_order=config.get("use_new_attention_order", False))

        self.model.to(dist_util.dev())
        self.model.load_state_dict(dist_util.load_state_dict(model_path, map_location=dist_util.dev()))
        self.model.eval()
        print("Loaded pretrained model...")

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
