import os
from typing import List, Optional, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from ddpmMed.utils.data import torch2np, normalize
from ddpmMed.utils.palette import colorize
from torch.nn.functional import interpolate


def plot_modal(image: np.ndarray, fname: str, suptitle: str, titles: Optional[List[str]]):
    """ plot a single modal image """

    if titles is None:
        # we assume we are using brats
        titles = ["T1", "T1ce", "T2", "Flair"]

    # image shape in H W C B
    if image.ndim > 3:
        h, w, c, b = image.shape
        if b == 1:
            image = image.squeeze(-1)
        else:
            raise RuntimeError(f"plot_modal() supports only batch size of 1, got batch = {b}")
    elif image.ndim == 2:
        raise RuntimeError(f"expected channels > 1")

    h, w, c = image.shape
    fig, ax = plt.subplots(1, c, figsize=(8, 8))
    for i in range(0, c):
        ax[i].imshow(normalize(image)[:, :, i], cmap="gray")
        ax[i].set_axis_off()
        ax[i].set_title(titles[i])
    plt.suptitle(suptitle)
    plt.savefig(fname=fname, dpi=600)
    plt.close()


def plot_result(prediction: torch.Tensor,
                ground_truth: torch.Tensor,
                palette: list = None,
                file_name: str = None,
                title: str = None,
                caption: str = None,
                fontsize: int = 14):
    """
    Plots a prediction vs mask and optionally
    saves it to storage
    """
    fig, ax = plt.subplots(1, 2)
    if palette is None:
        ax[0].imshow(torch2np(prediction, squeeze=True))
        ax[1].imshow(torch2np(ground_truth, squeeze=True))
    else:
        ax[0].imshow(colorize(prediction, palette))
        ax[1].imshow(colorize(ground_truth, palette))

    ax[0].set_axis_off()
    ax[1].set_axis_off()

    ax[0].set_title("Prediction")
    ax[1].set_title("Ground Truth")

    if caption is not None:
        fig.text(0.5, 0.05, caption, ha='left', fontsize=fontsize)

    if title is not None:
        fig.suptitle(title, fontsize=fontsize)

    if file_name is not None:
        plt.savefig(file_name, dpi=600)
    plt.close()
    return fig, ax

