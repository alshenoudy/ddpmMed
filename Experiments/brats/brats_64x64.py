import os
import numpy as np
from ddpmMed.insights.plots import plot_modal
from Experiments.config import brats_64x64_config

# get configuration dictionary
config = brats_64x64_config()
config["cache_dir"] = r"F:\diffusion"

# load samples
samples = np.load(r"F:\diffusion\samples_32x64x64x4.npz")['arr_0']
samples = iter(samples)

# plot and save all samples
folder = r"F:\ddpmMed\Experiments\brats\generated_samples_64x64"
for i, image in enumerate(samples):
    plot_modal(image, os.path.join(folder, f"image_{i}.jpeg"),
               suptitle=f"BRATS Sample {i}", titles=None)
