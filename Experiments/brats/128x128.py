from Experiments.config import brats_128x128_config
from Experiments.brats.experiment_script import brats_experiment

time_steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 999]
seeds = [42, 256, 88]


for step in time_steps:
    # run experiment
    config = brats_128x128_config()
    config["cache_dir"] = r"F:\diffusion\128x128 model\results"
    brats_experiment(
        config=config,
        model_dir=r"F:\diffusion\128x128 model\model350000.pt",
        images_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\scans",
        masks_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\masks",
        image_size=128,
        time_steps=[step],
        blocks=[18, 19, 20, 21],
        seeds=seeds,
        train_size=50,
        epochs=8,
        device='cuda'
    )
