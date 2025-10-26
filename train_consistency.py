import argparse
import copy
from copy import deepcopy
import logging
import os
import glob
from PIL import Image
import cv2

from pathlib import Path
from collections import OrderedDict
import json
from omegaconf import OmegaConf
import matplotlib

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import make_grid

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from ppd.models.dit import DiT
from ppd.models.depth_anything_v2.dpt import DepthAnythingV2

from ppd.utils.timesteps import Timesteps
from ppd.utils.schedule import LinearSchedule
from ppd.utils.sampler import EulerSampler
from ppd.utils.transform import image2tensor, resize_keep_aspect

from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.util.alignment import align_depth_least_square

from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.config_util import recursive_load_config
from src.util.loss import get_loss

import math

logger = get_logger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

#################################################################################
#                                Model inference                                #
#################################################################################
@torch.no_grad()
def infer_model_euler_method(
    args,
    model, 
    semantic_encoder, 
    image: torch.tensor,    # ~[0, 1] 
    device, 
    sampling_steps=4, 
    use_fp16: bool = True, 
):
    schedule = LinearSchedule(T=args.T)
    sampling_timesteps = Timesteps(
        T=args.T,
        steps=sampling_steps,
        device=device,
    )
    sampler = EulerSampler(
        schedule=schedule,
        timesteps=sampling_timesteps,
        prediction_type='velocity',
    )

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
        semantics = semantic_encoder(image)
        cond = image - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]], device=device)

        for timestep in sampling_timesteps:
            model_input = torch.cat([latent, cond], dim=1)
            pred = model(x=model_input, semantics=semantics, timestep=timestep)
            latent = sampler.step(pred=pred, x_t=latent, t=timestep)
    
    return latent + 0.5

@torch.no_grad()
def infer_model_consistency_sampling(
    args,
    model,
    semantic_encoder, 
    image: torch.tensor,    # ~[0, 1] 
    train_timesteps,    # NOTE: this includes 0, len(train_timesteps) = 51
    device, 
    sampling_steps=4, 
    use_fp16: bool = True,
):
    idx = torch.linspace(0, len(train_timesteps) - 1, sampling_steps + 1).int()
    timesteps = train_timesteps[idx]
    sampling_timesteps = timesteps[:-1]
    sampling_timesteps_next = timesteps[1:]

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
        semantics = semantic_encoder(image)
        cond = image - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]], device=device)

        for timestep, timestep_next in zip(sampling_timesteps, sampling_timesteps_next):
            model_input = torch.cat([latent, cond], dim=1)
            pred = model(x=model_input, semantics=semantics, timestep=timestep)
            pred_x_0 = latent - (timestep / args.T) * pred

            latent = (1 - timestep_next / args.T) * pred_x_0 + (timestep_next / args.T) * torch.randn_like(pred_x_0)
    
    return latent + 0.5

#################################################################################
#                                  Sanity check                                 #
#################################################################################
@torch.no_grad()
def sanity_check(
        args,
        model,
        semantic_encoder,
        device,
        sampling_steps=4,
        pred_only=True,
        save_dir="output/sanity_check",
        example_dir="assets/examples",
    ):
    os.makedirs(save_dir, exist_ok=True)

    image_paths = glob.glob(f"{example_dir}/*.jpg") + glob.glob(f"{example_dir}/*.jpeg") + glob.glob(f"{example_dir}/*.png")
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for image_path in image_paths:
        orig_image = cv2.imread(image_path)
        H, W = orig_image.shape[:2]

        resize_image = resize_keep_aspect(orig_image)
        image = image2tensor(resize_image)
        image = image.to(device)

        depth = infer_model_euler_method(
            args=args,
            model=model,
            semantic_encoder=semantic_encoder,
            image=image,
            device=device,
            sampling_steps=sampling_steps,
        )
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.squeeze().cpu().numpy()

        vis_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.
        vis_depth = vis_depth.astype(np.uint8)
        vis_depth_color = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if pred_only:
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + '_grey.png'), vis_depth)
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + '_color.png'), vis_depth_color)
        else:
            split_region = np.ones((orig_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([orig_image, split_region, np.repeat(vis_depth[:, :, None], 3, axis=2)])
            combined_result_color = cv2.hconcat([orig_image, split_region, vis_depth_color])
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + '_grey.png'), combined_result)
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + '_color.png'), combined_result_color)

    return

#################################################################################
#                                 Visualization                                 #
#################################################################################
@torch.no_grad()
def visualize(args, model, semantics_encoder, vis_loaders, save_dir, train_timesteps, device, sampling_steps=1, pred_only=True):
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    model.eval()
    for data_loader in vis_loaders:
        for batch in data_loader:
            assert 1 == data_loader.batch_size
            # Read input image
            image = batch["rgb_int"].to(device) / 255.0 # check image size
            
            depth = infer_model_consistency_sampling(
                args=args,
                model=model,
                semantic_encoder=semantics_encoder,
                image=image,
                train_timesteps=train_timesteps,
                device=device,
                sampling_steps=sampling_steps,
            )

            depth = depth[0, 0] # [h, w]
            depth = depth.squeeze().cpu().numpy()

            vis_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.
            vis_depth = vis_depth.astype(np.uint8)
            vis_depth_color = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            image_name = batch["rgb_relative_path"][0].replace("/", "_").split('.')[0]

            cv2.imwrite(os.path.join(save_dir, image_name + '_grey.png'), vis_depth)
            cv2.imwrite(os.path.join(save_dir, image_name + '_color.png'), vis_depth_color)
    
    return

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    args = recursive_load_config(args)

    # Set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        # Root output folder
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)

        # Visualization folder
        visualization_dir = f"{save_dir}/visualization"
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Checkpoint folder
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        
        # Save configs
        config_output = os.path.join(save_dir, "config.yaml")
        with open(config_output, "w+") as f:
            OmegaConf.save(config=args, f=f)

        args_dict = OmegaConf.to_container(args, resolve=True)
        
        # Log all args for reference
        logger.info("Training arguments:")
        for arg, value in sorted(args_dict.items()):
            logger.info(f"  {arg}: {value}")
        
    device = accelerator.device

    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Load DiT
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'ppd.pth'), map_location='cpu')
    _checkpoint = {}
    for k, v in checkpoint.items():
        _k = k.replace('dit.', '')
        _checkpoint[_k] = v
    
    # 1) Student model: average velocity prediction model
    model = DiT()
    missing_keys, unexpected_keys = model.load_state_dict(_checkpoint, strict=False)
    if accelerator.is_main_process:
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")

    model = model.to(device)
    requires_grad(model, True)

    if accelerator.is_main_process:
        logger.info("*** Student model initialized!")
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2) Teacher model: instantaneous velocity prediction model
    teacher_model = copy.deepcopy(model)
    requires_grad(teacher_model, False)
    if accelerator.is_main_process:
        logger.info("*** Teacher model initialized!")

    # 3) Create an EMA of the model for use after training
    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        if accelerator.is_main_process:
            logger.info("*** EMA model initialized!")
    else:
        ema = None
    
    # 4) Load semantic encoder
    semantics_encoder = DepthAnythingV2(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024]
    )
    semantics_encoder.load_state_dict(
        torch.load(os.path.join(args.checkpoint_dir, 'depth_anything_v2_vitl.pth'), map_location='cpu'),
        strict=False
    )
    
    semantics_encoder = semantics_encoder.to(device).eval()
    requires_grad(semantics_encoder, False)
    if accelerator.is_main_process:
        logger.info("*** Semantics encoder initialized!")
    
    # Sanity check
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and args.do_sanity_check:
        sanity_check_dir = os.path.join(save_dir, "sanity_check")
        sanity_check(
            args=args,
            model=model,
            semantic_encoder=semantics_encoder,
            device=device,
            sampling_steps=4,
            pred_only=True,
            save_dir=sanity_check_dir,
        )
        logger.info(f"Sanity check completed. ---> '{sanity_check_dir}'")
    accelerator.wait_for_everyone()

    
    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    loss_func = get_loss(loss_name=args.loss.name, **args.loss.kwargs)
    
    # Setup data:
    local_batch_size = int(args.batch_size // accelerator.num_processes)

    # Training dataset & dataloader
    depth_transform = get_depth_normalizer(
        cfg_normalizer=args.depth_normalization
    )
    train_dataset = get_dataset(
        args.dataset.train,
        base_data_dir=args.data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=args.augmentation,
        depth_transform=depth_transform,
        target_type=args.depth_target_type
    )
    if "mixed" == args.dataset.train.name:
        dataset_ls = train_dataset
        assert len(args.dataset.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=local_batch_size,
            drop_last=args.drop_last,
            prob=args.dataset.train.prob_ls,
            shuffle=args.shuffle,
        )
        train_dataloader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=args.num_workers,
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=local_batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            drop_last=args.drop_last,
        )

    # Visualization dataset & dataloader
    vis_loaders = []
    for _vis_dict in args.dataset.vis:
        _vis_dataset = get_dataset(
            _vis_dict,
            base_data_dir=args.data_dir,
            mode=DatasetMode.EVAL,
        )
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        vis_loaders.append(_vis_loader)

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(concat_dataset):,} images in '{args.data_dir}'")
    
    # Depth variables
    gt_depth_type = args.gt_depth_type
    gt_mask_type = args.gt_mask_type

    # Prepare models for training:
    model.train()
    if args.ema:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        if args.ema:
            ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Timesteps & Time Scheduling for training
    T = args.T
    num_train_timesteps = args.num_train_timesteps

    schedule = LinearSchedule(T=T)
    sampling_timesteps = Timesteps(
        T=T,
        steps=num_train_timesteps,
        device=device,
    )
    sampler = EulerSampler(
        schedule=schedule,
        timesteps=sampling_timesteps,
        prediction_type='velocity'
    )
    timesteps = torch.cat([sampling_timesteps.timesteps, torch.zeros(1, device=device)])
    train_timesteps_t = timesteps[:-1]
    train_timesteps_s = timesteps[1:]

    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(concat_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {local_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            model.train()

            with accelerator.accumulate(model):
                image = batch["rgb_int"].to(device) / 255.0     # ~  [0, 1], torch.tensor([b, 3, h, w])
                depth_gt = batch[gt_depth_type].to(device)      # ~ [-0.5, 0.5], torch.tensor([b, 1, h, w])
                valid_depth_mask = batch[gt_mask_type].to(device) if gt_mask_type is not None else None
                
                batch_size, _, height, width = image.shape
                
                # obtain semantic information
                with torch.no_grad():
                    semantics = semantics_encoder(image)
                    cond = image - 0.5

                # sample timesteps t & s
                idx = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
                t = train_timesteps_t[idx]
                s = train_timesteps_s[idx]

                # sample noise
                noise = torch.randn_like(depth_gt)

                # prepare model input
                x_t = schedule.forward(x_0=depth_gt, x_T=noise, t=t)
                model_input = torch.cat([x_t, cond], dim=1)

                # model predicted 0-pointing velocity
                pred_u = model(x=model_input, semantics=semantics, timestep=t)
                pred_x0_student, _ = schedule.convert_from_pred(pred_u, 'velocity', x_t, t)

                with torch.no_grad():
                    # Teacher pred x_s: from x_t -> x_s
                    v_pred_t = teacher_model(x=model_input, semantics=semantics, timestep=t)
                    x_s = sampler.step(pred=v_pred_t, x_t=x_t, t=t)

                    model_input_target = torch.cat([x_s, cond], dim=1)
                    pred_v = model(x=model_input_target, semantics=semantics, timestep=s)
                    x_0_target, _ = schedule.convert_from_pred(pred_v, 'velocity', x_s, s)

                    mask = (s <= 0).view(-1, 1, 1, 1)
                    x_0_target = torch.where(mask, depth_gt, x_0_target)
                
                if args.interpolate_gt:
                    _t = t.view(-1, 1, 1, 1)
                    x_0_target = (1 - _t) * x_0_target + _t * depth_gt

                # loss_mean_ref = torch.mean((error ** 2))
                if valid_depth_mask is not None:
                    pred_x0_student = pred_x0_student[valid_depth_mask]
                    x_0_target = x_0_target[valid_depth_mask]
                
                loss = loss_func(pred_x0_student.float(), x_0_target.float())

                ## optimization
                accelerator.backward(loss)
                grad_norm = 0.0
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients and args.ema:
                    update_ema(ema, model)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0 or global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if args.ema:
                        checkpoint.update({"ema": ema.state_dict()})

                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if global_step % args.visualization_steps == 0 and global_step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    for vis_sampling_step in args.vis_sampling_steps:
                        cur_step_vis_dir = os.path.join(
                            visualization_dir, f"iter_{global_step}", f"{vis_sampling_step}_steps"
                        )
                        os.makedirs(cur_step_vis_dir, exist_ok=True)
                        visualize(
                            args=args,
                            model=model,
                            semantics_encoder=semantics_encoder,
                            vis_loaders=vis_loaders,
                            save_dir=cur_step_vis_dir,
                            train_timesteps=timesteps,
                            device=device,
                            sampling_steps=vis_sampling_step
                        )
                accelerator.wait_for_everyone()
            
            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
                "loss_scaled": accelerator.gather(loss).mean().detach().item() * 100,
                # "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            
            # Log to file periodically
            if accelerator.is_main_process and global_step % args.logging_steps == 0:
                # logger.info(f"Step {global_step}: loss = {logs['loss']:.4f}, loss_scaled(x100) = {logs['loss_scaled']:.4f}, grad_norm = {logs['grad_norm']:.4f}")
                logger.info(f"Step {global_step}: loss = {logs['loss']:.8f}, loss_scaled(x100) = {logs['loss_scaled']:.8f}")

            if global_step >= args.max_train_steps:
                break
        
        # Log epoch completion
        if accelerator.is_main_process:
            logger.info(f"Completed epoch {epoch+1}/{num_train_epochs}")
            
        if global_step >= args.max_train_steps:
            break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed!")
    accelerator.end_training()

def parse_args():
    parser = argparse.ArgumentParser(description="MeanFlow Training")
    parser.add_argument("--config", type=str, default="config/train_ppd_depth.yaml")
    
    args = parser.parse_args()
    
    return args.config


    # # logging:
    # parser.add_argument("--output-dir", type=str, default="exps")
    # parser.add_argument("--exp-name", type=str, required=True)
    # parser.add_argument("--logging-dir", type=str, default="logs")
    # parser.add_argument("--resume-step", type=int, default=0)

    # # model
    # parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # parser.add_argument("--image_encoder", type=str, choices=["dino_v1, dino_v2, depth_anything_v1, depth_anything_v2"], default="depth_anything_v2")
    # parser.add_argument("--time_ebmedding", type=str, choices=["single", "dual_v1", "dual_v2"], default="single")

    # # dataset
    # parser.add_argument("--data-dir", type=str, default="/data/train_sdvae_latents_lmdb")
    # parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    # parser.add_argument("--batch-size", type=int, default=256)

    # # precision
    # parser.add_argument("--allow-tf32", action="store_true")
    # parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # # optimization
    # parser.add_argument("--epochs", type=int, default=240)
    # parser.add_argument("--max-train-steps", type=int, default=None)
    # parser.add_argument("--checkpointing-steps", type=int, default=50000)
    # parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # parser.add_argument("--learning-rate", type=float, default=1e-4)
    # parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    # parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    # parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    # parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    # parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # # seed
    # parser.add_argument("--seed", type=int, default=0)

    # # cpu
    # parser.add_argument("--num-workers", type=int, default=4)

    # # basic loss
    # parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    # parser.add_argument("--cfg-prob", type=float, default=0.1)
    # parser.add_argument("--weighting", default="adaptive", type=str, choices=["uniform", "adaptive"], help="Loss weighting type")
    
    # # MeanFlow specific parameters
    # parser.add_argument("--time-sampler", type=str, default="logit_normal", choices=["uniform", "logit_normal"], 
    #                    help="Time sampling strategy")
    # parser.add_argument("--time-mu", type=float, default=-0.4, help="Mean parameter for logit_normal distribution")
    # parser.add_argument("--time-sigma", type=float, default=1.0, help="Std parameter for logit_normal distribution")
    # parser.add_argument("--ratio-r-not-equal-t", type=float, default=0.75, help="Ratio of samples where r≠t")
    # parser.add_argument("--adaptive-p", type=float, default=1.0, help="Power param for adaptive weighting")
    # parser.add_argument("--cfg-omega", type=float, default=1.0, help="CFG omega param, default 1.0 means no CFG")
    # parser.add_argument("--cfg-kappa", type=float, default=0.0, help="CFG kappa param for mixing")
    # parser.add_argument("--cfg-min-t", type=float, default=0.0, help="Minum time for cfg trigger")
    # parser.add_argument("--cfg-max-t", type=float, default=1.0, help="Maxium time for cfg trigger")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)