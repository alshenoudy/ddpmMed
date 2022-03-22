import os
from typing import Any
from torch.utils.data import DataLoader
from ddpmMed.data.datasets import ImageDataset
from ddpmMed.diffusion.improved_ddpm import logger, dist_util
from ddpmMed.diffusion.improved_ddpm.resample import create_named_schedule_sampler
from ddpmMed.diffusion.improved_ddpm.train import TrainLoop
from ddpmMed.scripts.script_utils import create_model_and_diffusion


def load_data(data_dir: str, image_size: int, transforms: Any, batch_size: int, deterministic: bool = False,
              seed: int = 42, device: str = 'cuda'):
    """
    Creates a generator over a dataset that cycles continuously over all data entries

    :param data_dir: dataset directory, folder containing images
    :param image_size: image size for transformed images, ignore if transforms is specified
    :param transforms: a transforms object, a list of transformations for input images
    :param batch_size: data loader's batch size
    :param deterministic: either to shuffle or avoid shuffling the dataloader
    :param seed: random seed
    :param device: device to move images to
    :return: a cycling generator over the entire dataset entries
    """
    # create dataset object
    dataset = ImageDataset(images_dir=data_dir, image_size=image_size,
                           transforms=transforms, seed=seed, device=device)

    # create dataloader
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(not deterministic))

    # cycle over all data
    while True:
        yield from loader


def train_on_images(config: dict):
    dist_util.setup_dist()
    logger.configure(config.get("cache_dir"))
    logger.log("creating model and diffusion...")

    # write configuration file
    file = os.path.join(config.get("cache_dir"), "config.txt")
    with open(file, 'w') as f:
        for k, v in config.items():
            f.write(f"{k}:{v}\n")
    f.close()

    # create model and diffusion
    model, diffusion = create_model_and_diffusion(
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

    model.cuda()
    schedule_sampler = create_named_schedule_sampler(config.get("sampler", "uniform"), diffusion)
    logger.log("creating data loader...")
    data = load_data(
        data_dir=config.get("data"),
        image_size=config.get("image_size", 128),
        transforms=None,
        batch_size=config.get("batch_size"),
        deterministic=False,
        seed=42,
        device='cuda')

    # start training
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=config.get("batch_size"),
        microbatch=config.get("micro_batch"),
        lr=config.get("lr"),
        ema_rate=config.get("ema_rate"),
        log_interval=config.get("log_interval"),
        save_interval=config.get("save_interval"),
        resume_checkpoint=config.get("resume_train_checkpoint"),
        use_fp16=config.get("use_fp16"),
        schedule_sampler=schedule_sampler,
        weight_decay=config.get("weight_decay"),
        lr_anneal_steps=config.get("total_steps"),
    ).run_loop()


