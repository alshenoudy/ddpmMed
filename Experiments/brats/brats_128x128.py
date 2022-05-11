from ddpmMed.utils.data import binary_brats, brats_labels, brats_tumor_core, brats_ET
from Experiments.config import brats_128x128_config
from Experiments.brats.experiment_script import brats_experiment

time_steps = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000]
seeds = [42]
for t in time_steps:
    # run experiment
    config = brats_128x128_config()
    config["cache_dir"] = r"F:\diffusion\layers and time"
    brats_experiment(
        config=config,
        model_dir=r"F:\diffusion\128x128 model\model350000.pt",
        images_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\scans",
        masks_dir=r"E:\1. Datasets\1. BRATS 2021\2D\Training\masks",
        image_size=128,
        time_steps=[t],
        blocks=[20],
        seeds=[16],
        train_size=50,
        epochs=8,
        use_val=True,
        batch_size=64,
        num_classes=4,
        ensemble_size=10,
        label_names=None,
        process_labels=None,
        plot_results=True,
        device='cuda'
    )
