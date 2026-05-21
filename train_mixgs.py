#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import time
import yaml
import os
import torch
import torchvision
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render_mix
import sys
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from scene import LargeScene, MixGSModel
from scene.datasets import GSDataset, CacheDataLoader
from utils.camera_utils import loadCam
from utils.general_utils import safe_state, parse_cfg
from tqdm import tqdm
from os import makedirs
from utils.image_utils import psnr
from utils.log_utils import TrainingResourceLogger, tensorboard_log_image, wandb_log_image
from argparse import ArgumentParser, Namespace
from lpipsPyTorch import lpips
from fused_ssim import fused_ssim


class StageBudgetController:
    def __init__(self, pipe, joint_start_iter):
        self.enabled = bool(getattr(pipe, "stage_budget_adjustment", False))
        self.base_budget = int(getattr(pipe, "render_gaussian_budget", 0) or 0)
        self.joint_start_iter = int(joint_start_iter)
        self.scale = self._clamp(float(getattr(pipe, "stage_budget_initial_scale", 0.75) or 0.75), 0.0, 1.0)
        self.min_scale = self._clamp(float(getattr(pipe, "stage_budget_min_scale", 0.10) or 0.10), 0.0, 1.0)
        self.safety_scale = self._clamp(float(getattr(pipe, "stage_budget_safety_scale", 0.95) or 0.95), 0.0, 1.0)
        self.grow_factor = max(1.0, float(getattr(pipe, "stage_budget_grow_factor", 1.02) or 1.02))
        self.reference_vram_mb = 0.0
        self.observed_vram_mb = 0.0

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(value, upper))

    def is_active(self, iteration):
        return self.enabled and self.base_budget > 0 and iteration >= self.joint_start_iter

    def current_scale(self, iteration):
        if not self.is_active(iteration):
            return 1.0
        return self.scale

    def budget_for(self, iteration):
        if self.base_budget <= 0:
            return 0
        if not self.is_active(iteration):
            return self.base_budget
        return max(1, int(self.base_budget * self.scale))

    def observe(self, iteration, observed_vram_mb):
        if not self.enabled:
            return
        self.observed_vram_mb = float(observed_vram_mb or 0.0)
        if self.observed_vram_mb <= 0:
            return

        if iteration < self.joint_start_iter:
            if self.reference_vram_mb <= 0:
                self.reference_vram_mb = self.observed_vram_mb
            else:
                self.reference_vram_mb = 0.1 * self.observed_vram_mb + 0.9 * self.reference_vram_mb
            return

        if self.reference_vram_mb <= 0:
            return

        target_vram_mb = self.reference_vram_mb * self.safety_scale
        if self.observed_vram_mb > target_vram_mb:
            shrink = self._clamp(target_vram_mb / max(self.observed_vram_mb, 1.0), self.min_scale, 1.0)
            self.scale = self._clamp(self.scale * shrink, self.min_scale, 1.0)
        elif self.observed_vram_mb < target_vram_mb * 0.95:
            self.scale = self._clamp(self.scale * self.grow_factor, self.min_scale, 1.0)

    def stats(self, effective_budget, used_scale):
        if not self.enabled:
            return {}
        return {
            "base_render_gaussian_budget": self.base_budget,
            "effective_render_gaussian_budget": int(effective_budget),
            "stage_budget_scale": float(used_scale),
            "stage_budget_reference_vram_mb": float(self.reference_vram_mb),
            "stage_budget_observed_vram_mb": float(self.observed_vram_mb),
        }


def training(dataset, opt, pipe, testing_iterations, saving_iterations, refilter_iterations, checkpoint_iterations,
             checkpoint, max_cache_num, debug_from):
    first_iter = 0
    log_writer, image_logger = prepare_output_and_logger(dataset)
    resource_log_interval = int(getattr(pipe, "resource_log_interval", 0) or 0)
    resource_plot_on_complete = bool(getattr(pipe, "resource_plot_on_complete", False))
    resource_logger = None
    if resource_log_interval > 0 and log_writer:
        resource_logger = TrainingResourceLogger(
            getattr(log_writer, "log_dir", None) or dataset.model_path,
            getattr(dataset, "detail_max_slots", 1),
        )
    stage_budget_controller = StageBudgetController(pipe, getattr(opt, "joint_start_iter", opt.iterations + 1))

    modules = __import__('scene')
    model_config = dataset.model_config
    model_kwargs = dict(model_config['kwargs'])
    model_kwargs.setdefault("max_detail_slots", getattr(dataset, "detail_max_slots", 1))
    gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_kwargs)

    mixgs = MixGSModel(
        hash_args=dataset.hash_args,
        net_args=dataset.network_args,
        max_detail_slots=getattr(dataset, "detail_max_slots", 1),
    )
    mixgs.train_setting(opt)

    scene = LargeScene(dataset, gaussians)
    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe)
    if len(gs_dataset) > 0:
        print(f"Using maximum cache size of {max_cache_num} for {len(gs_dataset)} training images")
        data_loader = CacheDataLoader(gs_dataset, max_cache_num=max_cache_num, seed=42, batch_size=1, shuffle=True, num_workers=8)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_time_render = 0.0
    ema_time_loss = 0.0
    ema_time_densify = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    while iteration <= opt.iterations:
        if len(gs_dataset) == 0:
            print("No training data found")
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration, dataset)
            break

        for dataset_index, (cam_info, gt_image) in enumerate(data_loader):
            should_log_resources = (
                resource_logger is not None
                and iteration % resource_log_interval == 0
            )
            if (should_log_resources or stage_budget_controller.enabled) and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            iter_start.record()

            # Render
            start = time.time()
            if (iteration - 1) == debug_from:
                pipe.debug = True

            if iteration == opt.joint_start_iter:
                gaussians.gaussian_training()

            effective_render_gaussian_budget = stage_budget_controller.budget_for(iteration)
            stage_budget_scale = stage_budget_controller.current_scale(iteration)
            vis_mask = prefilter_voxel(cam_info, gaussians, pipe, background)
            hash_input = [gaussians.get_xyz[vis_mask].detach(), gaussians.get_scaling[vis_mask].detach(), gaussians.get_rotation[vis_mask].detach()]
            decoded_data = mixgs.step(
                hash_input,
                cam_info['world_view_transform'][-1, :-1],
                render_gaussian_budget=effective_render_gaussian_budget,
                scale_min=getattr(pipe, "scale_min", 0.0),
            )

            render_pkg = render_mix(cam_info, gaussians, pipe, background, vis_mask, decoded_data)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            end = time.time()
            ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render

            # Loss
            start = time.time()
            gt_image = gt_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

            loss.backward()
            end = time.time()
            ema_time_loss = 0.4 * (end - start) + 0.6 * ema_time_loss

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                grads = gaussians.xyz_gradient_accum / gaussians.denom
                grads[grads.isnan()] = 0.0
                budget_stats = render_pkg.get("budget_stats", {})
                observed_vram_mb = 0.0
                if torch.cuda.is_available():
                    observed_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
                stage_budget_controller.observe(iteration, observed_vram_mb)
                budget_stats.update(stage_budget_controller.stats(effective_render_gaussian_budget, stage_budget_scale))
                if should_log_resources:
                    resource_logger.log(iteration, budget_stats, decoded_data.get("detail_counts"))

                ema_time = {
                    "render": ema_time_render,
                    "loss": ema_time_loss,
                    "densify": ema_time_densify,
                    "num_points": radii.shape[0],
                    "mean_grad": grads.mean().item(),
                    "visible_anchor_count": budget_stats.get("visible_anchor_count", 0),
                    "detail_budget": budget_stats.get("detail_budget", 0),
                    "selected_detail_count": budget_stats.get("selected_detail_count", 0),
                    "render_gaussian_budget": budget_stats.get("render_gaussian_budget", 0),
                    "budget_overflow": int(budget_stats.get("budget_overflow", False)),
                    "base_render_gaussian_budget": budget_stats.get("base_render_gaussian_budget", 0),
                    "effective_render_gaussian_budget": budget_stats.get("effective_render_gaussian_budget", 0),
                    "stage_budget_scale": budget_stats.get("stage_budget_scale", 1.0),
                    "stage_budget_reference_vram_mb": budget_stats.get("stage_budget_reference_vram_mb", 0),
                    "stage_budget_observed_vram_mb": budget_stats.get("stage_budget_observed_vram_mb", 0),
                }

                lr = {}
                for param_group in gaussians.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                for param_group in mixgs.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                # Log and save
                training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr,
                                iter_start.elapsed_time(iter_end), testing_iterations, scene, mixgs, (pipe, background))

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    # log_writer.log_dir
                    scene.save(iteration, log_writer.log_dir)
                    mixgs.save_weights(log_writer.log_dir, iteration)

                if (iteration in refilter_iterations):
                    print("\n[ITER {}] Refiltering Training Data".format(iteration))
                    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe)

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                    mixgs.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    mixgs.optimizer.zero_grad()
                    mixgs.update_learning_rate(iteration)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            iteration += 1
            if iteration >= opt.iterations:
                break

    if resource_logger is not None and resource_plot_on_complete:
        resource_logger.plot()


def prepare_output_and_logger(args):
    if not args.model_path:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        args.model_path = os.path.join("./output/", config_name)
        if args.block_id >= 0:
            if args.block_id < args.block_dim[0] * args.block_dim[1] * args.block_dim[2]:
                args.model_path = f"{args.model_path}/cells/cell{args.block_id}"
                if args.logger_config is not None:
                    args.logger_config['name'] = f"{args.logger_config['name']}_cell{args.block_id}"
            else:
                raise ValueError("Invalid block_id: {}".format(args.block_id))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # build logger
    log_writer = None
    image_logger = None
    logger_args = {
        "save_dir": args.model_path
    }
    if args.logger_config is None or args.logger_config['logger'] == "tensorboard":
        log_writer = TensorBoardLogger(**logger_args)
        image_logger = tensorboard_log_image
    elif args.logger_config['logger'] == "wandb":
        logger_args.update(name=args.logger_config['name'])
        logger_args.update(project=args.logger_config['project'])
        log_writer = WandbLogger(**logger_args)
        image_logger = wandb_log_image
    else:
        raise ValueError("Unknown logger: {}".format(args.logger_config['logger']))

    return log_writer, image_logger


def training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr, elapsed,
                    testing_iterations, scene: LargeScene, mixgs, renderArgs):
    if log_writer:
        metrics_to_log = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_time/render": ema_time["render"],
            "train_time/loss": ema_time["loss"],
            "train_time/densify": ema_time["densify"],
            "train_time/num_points": ema_time["num_points"],
            "train_time/mean_grad": ema_time["mean_grad"],
            "iter_time": elapsed,
        }
        for key in (
            "visible_anchor_count",
            "detail_budget",
            "selected_detail_count",
            "render_gaussian_budget",
            "budget_overflow",
            "base_render_gaussian_budget",
            "effective_render_gaussian_budget",
            "stage_budget_scale",
            "stage_budget_reference_vram_mb",
            "stage_budget_observed_vram_mb",
        ):
            if key in ema_time:
                metrics_to_log["train_budget/" + key] = ema_time[key]
        for key, value in lr.items():
            metrics_to_log["trainer/" + key] = value
        log_writer.log_metrics(metrics_to_log, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims_test = 0.0
                lpips_test = 0.0
                for idx, camera in enumerate(config['cameras']):
                    viewpoint_cam = loadCam(dataset, id, camera, 1)
                    viewpoint = {
                        "FoVx": viewpoint_cam.FoVx,
                        "FoVy": viewpoint_cam.FoVy,
                        "image_name": viewpoint_cam.image_name,
                        "image_height": viewpoint_cam.image_height,
                        "image_width": viewpoint_cam.image_width,
                        "camera_center": viewpoint_cam.camera_center,
                        "world_view_transform": viewpoint_cam.world_view_transform,
                        "full_proj_transform": viewpoint_cam.full_proj_transform,
                    }
                    org_img = viewpoint_cam.original_image

                    vis_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)

                    hash_input = [scene.gaussians.get_xyz[vis_mask].detach(), scene.gaussians.get_scaling[vis_mask].detach(),
                                  scene.gaussians.get_rotation[vis_mask].detach()]
                    decoded_data = mixgs.step(
                        hash_input,
                        viewpoint['world_view_transform'][-1, :-1],
                        render_gaussian_budget=getattr(renderArgs[0], "render_gaussian_budget", 0),
                        scale_min=getattr(renderArgs[0], "scale_min", 0.0),
                    )

                    render_pkg = render_mix(viewpoint, scene.gaussians, *renderArgs, vis_mask, decoded_data)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(org_img.to("cuda"), 0.0, 1.0)

                    if log_writer and (idx < 5):
                        grid = torchvision.utils.make_grid(torch.concat([image, gt_image], dim=-1))
                        image_logger(
                            log_writer=log_writer,
                            tag=config['name'] + "_view_{}".format(viewpoint["image_name"]),
                            image_tensor=grid,
                            step=iteration,
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssims_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssims_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssims_test, lpips_test))
                if log_writer:
                    metrics_to_log = {
                        config['name'] + '/loss_viewpoint/l1_loss': l1_test,
                        config['name'] + '/loss_viewpoint/psnr': psnr_test,
                        config['name'] + '/loss_viewpoint/ssim': ssims_test,
                        config['name'] + '/loss_viewpoint/lpips': lpips_test,
                    }
                    log_writer.log_metrics(metrics_to_log, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[200_000, 250_000, 300_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[200_000, 250_000, 300_000])

    parser.add_argument("--refilter_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--max_cache_num", type=int, default=32)
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)
        args.save_iterations.append(op.iterations)

    print("Optimizing " + lp.model_path)

    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, [11264, 65535])

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, args.test_iterations, args.save_iterations, args.refilter_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.max_cache_num, args.debug_from)

    # All done
    print("\nTraining complete.")
