import os.path

from Experiments.config import brats_128x128_config
from Experiments.brats.experiment_script import brats_experiment

time_steps = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000]
layers = [21, 22, 23, 14, 13, 12, 11, 10, 8, 9]
seeds = [42]

config = brats_128x128_config()

for layer in layers:
    config["cache_dir"] = os.path.join(r"H:\brats 128x128\layers vs time", f"layer_{layer}")

    if not os.path.exists(config["cache_dir"]):
        os.makedirs(config["cache_dir"], exist_ok=True)

    for t in time_steps:
        # run experiment
        brats_experiment(
            config=config,
            model_dir=r"H:\brats 128x128\model360000.pt",
            images_dir=r"H:\BRATS\2D Stacked Images\scans",
            masks_dir=r"H:\BRATS\2D Stacked Images\masks",
            image_size=128,
            time_steps=[t],
            blocks=[layer],
            seeds=[16],
            train_size=50,
            epochs=8,
            use_val=True,
            device='cuda'
        )
