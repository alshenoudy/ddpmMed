import os
from typing import List, Optional, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from ddpmMed.utils.data import torch2np, normalize


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

