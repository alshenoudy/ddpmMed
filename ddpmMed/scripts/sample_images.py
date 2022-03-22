import os
import numpy as np
import torch.distributed as dist
import torch
from ddpmMed.diffusion.improved_ddpm import dist_util, logger
from ddpmMed.diffusion.improved_ddpm.resample import create_named_schedule_sampler
from ddpmMed.scripts.script_utils import create_model_and_diffusion


def sample_images(config: dict, model_dir: str):

    dist_util.setup_dist()
    logger.configure(config.get("cache_dir"))
    logger.log("creating model and diffusion...")

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

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(config.get("sampler", "uniform"), diffusion)
    model.load_state_dict(
        dist_util.load_state_dict(model_dir, map_location='cuda')
    )

    model.eval()
    use_ddim = False
    class_cond = False
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * 4 < 32:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (4, model.in_channels, model.image_size, model.image_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * 4} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: 32]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")
    return arr

