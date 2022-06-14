import os
from ddpmMed.utils.data import binary_brats, brats_labels, brats_tumor_core, brats_ET
from Experiments.config import brats_128x128_config
from Experiments.brats.experiment_script import brats_experiment

config = brats_128x128_config()
config["cache_dir"] = r"F:\diffusion\debugging"


if not os.path.exists(config["cache_dir"]):
    os.makedirs(config["cache_dir"], exist_ok=True)

# run experiment
brats_experiment(
    config=config,
    model_dir=r"F:\diffusion\128x128 model\model350000.pt",
    images_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\scans",
    masks_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\masks",
    image_size=128,
    time_steps=[50, 250, 350],
    blocks=[16, 17, 18, 19, 20],
    seeds=[42],
    train_size=30,
    epochs=8,
    use_val=True,
    batch_size=32,
    num_classes=4,
    ensemble_size=3,
    label_names=None,
    process_labels=None,
    plot_results=True,
    device='cuda'
)
