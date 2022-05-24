import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from ddpmMed.scripts.script_utils import load_model


class DiffUNet(nn.Module):
    """
    Uses a trained noise predictor network from a diffusion model
    to predict directly on labels, where the generative process is
    used as an initialization to the network's weights.
    """
    def __init__(self, model_path: str, config: dict, out_channels: int = 4) -> None:
        super(DiffUNet, self).__init__()
        self.model_path = model_path
        self.config = config
        self.out_channels = out_channels

        # load model and diffusion
        self.model, self.diffusion = load_model(model_path=self.model_path, config=self.config)

        # freeze Encoder part
        self.model.freeze_encoder()

        # adjust final layer's output channels
        if self.model.out[-1].out_channels != self.out_channels:
            self.model.out[-1] = nn.Conv2d(
                in_channels=self.model.out[-1].in_channels,
                out_channels=self.out_channels,
                kernel_size=self.model.out[-1].kernel_size,
                stride=self.model.out[-1].stride,
                padding=self.model.out[-1].padding
            )

    # def freeze_encoder(self, middle_block: bool = False) -> None:
    #     """
    #     Freezes the weights of the encoder part of the Unet
    #     """
    #     for name, param in self.model.input_blocks.named_parameters():
    #         param.requires_grad = False
    #
    #     if middle_block:
    #         for name, param in self.model.middle_block.named_parameters():
    #             param.requires_grad = False

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples an input image to a given time-step, and passes
        it to the noise predictor network
        """

        x_noisy = self.diffusion.q_sample(x_start=x, t=t)
        x = self.model(x=x_noisy, timesteps=t)
        return x

    def load_pre_trained(self, path: str):
        self.load_state_dict(
            torch.load(f=path, map_location=self.model.device)
        )


class DiffUNetTrainer:
    def __init__(self,
                 model: DiffUNet,
                 time: int = 250,
                 lr: float = 0.0001,
                 epochs: int = 25,
                 dataloader: DataLoader = None,
                 cache_folder: str = None) -> None:

        self.model = model
        self.time = torch.tensor([time]).to(self.model.model.device)
        self.lr = lr
        self.epochs = epochs
        self.dataloader = dataloader
        self.cache_folder = cache_folder
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )

        if not os.path.exists(self.cache_folder) or self.cache_folder is None:
            self.cache_folder = os.getcwd()
            self.cache_folder = os.path.join(self.cache_folder, "DiffUNet")
            os.makedirs(self.cache_folder, exist_ok=True)

    def train(self):

        with tqdm(range(0, self.epochs), postfix={"epoch_loss": "n/a",  "batch_loss": "n/a"}) as pbar:
            epoch_losses = []
            for epoch in pbar:
                batch_losses = []
                for images, masks in self.dataloader:
                    self.optimizer.zero_grad()

                    predictions = self.model(x=images, t=self.time)
                    losses = self.loss_function(predictions, masks.squeeze(1))
                    losses.backward()

                    self.optimizer.step()

                    # append loss of current batch
                    batch_losses.append(losses.item())
                    pbar.set_postfix({"epoch_loss": f"{epoch_losses[-1]:.5f}" if len(epoch_losses) > 0 else "n/a",
                                      "batch_loss": f"{batch_losses[-1]:.5f}"})

                # add mean batch losses over current epoch
                epoch_losses.append(np.mean(batch_losses))
                # pbar.set_postfix({"epoch_loss": f"{epoch_losses[-1]:.5f}"})
                pbar.set_description(f"Epoch: {epoch + 1}")
            pbar.set_description("Saving Model")

            torch.save(self.model.state_dict(), f=os.path.join(self.cache_folder, f"model_{epoch + 1}.pt"))
            print(f"\n\nSaved model to: {self.cache_folder}")

            fig, ax = plt.subplots(1, 1, figsize=(15, 4))
            ax.plot(epoch_losses)
            ax.set_title("Diffusion UNet training losses")
            ax.set_xlabel("epochs")
            ax.set_ylabel("loss")

            plt.savefig(os.path.join(self.cache_folder, f"training_losses_{epoch + 1}.jpeg"))
            plt.close()














