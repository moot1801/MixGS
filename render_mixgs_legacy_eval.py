#
# Legacy MixGS evaluation renderer.
#
# This script keeps the original MixGS runtime path local to this file so that
# old checkpoints can be evaluated without depending on the current research
# MixGSModel/allocation implementation.
#

import json
import math
import os
import resource
import sys
import time
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as tf
import yaml
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import prefilter_voxel
from hashencoder.hashgrid import HashEncoder
from lpipsPyTorch import lpips
from scene import LargeScene
from utils.camera_utils import loadCam
from utils.general_utils import colorize, get_expon_lr_func, parse_cfg, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import eval_sh
from utils.system_utils import searchForMaxIteration


class LegacyGSEncoder(nn.Module):
    def __init__(
            self,
            canonical_num_levels=16,
            canonical_level_dim=2,
            canonical_base_resolution=16,
            canonical_desired_resolution=2048,
            canonical_log2_hashmap_size=19,
            decoder_num_levels=32,
            decoder_level_dim=2,
            bound=256.0,
    ):
        super().__init__()
        self.out_dim = canonical_num_levels * canonical_level_dim + decoder_num_levels * decoder_level_dim * 3
        self.canonical_num_levels = canonical_num_levels
        self.canonical_level_dim = canonical_level_dim
        self.bound = bound
        self.xyz_encoding = HashEncoder(
            input_dim=3,
            num_levels=canonical_num_levels,
            level_dim=canonical_level_dim,
            per_level_scale=2,
            base_resolution=canonical_base_resolution,
            log2_hashmap_size=canonical_log2_hashmap_size,
            desired_resolution=canonical_desired_resolution,
        )

    def forward(self, coords, pose):
        return self.xyz_encoding(coords, size=self.bound), pose.unsqueeze(0).repeat(coords.size()[0], 1)


class LegacyGSDecoder(nn.Module):
    def __init__(self, spatial_in_dim, mlp_in_dim, depth=1, width=256):
        super().__init__()
        self.depth = depth
        self.width = width
        self.spatial_mlp = nn.Sequential(nn.Linear(spatial_in_dim, width))
        self.mlp = nn.Sequential(nn.Linear(mlp_in_dim, width), nn.ReLU())

        mlp = []
        for _ in range(depth):
            mlp.append(nn.Linear(width, width))
            mlp.append(nn.ReLU())
        self.tiny_mlp = nn.Sequential(*mlp)

        self.gaussian_color = nn.Linear(width, 3)
        self.gaussian_rotation = nn.Linear(width, 4)
        self.gaussian_scaling = nn.Linear(width, 3)
        self.gaussian_opacity = nn.Linear(width, 1)

    def forward(self, spatial_h, pose_input, scale_input, rotate_input):
        spatial_h = self.spatial_mlp(spatial_h)
        cat_feat = torch.cat([pose_input, scale_input, rotate_input], dim=1)
        cat_feat = self.mlp(cat_feat)
        h = spatial_h * (2 * torch.sigmoid(cat_feat) - 1)
        h = self.tiny_mlp(h)

        color = self.gaussian_color(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)
        opacity = torch.sigmoid(self.gaussian_opacity(h))
        return color, rotation, scaling, opacity


class LegacyMixGSModel:
    def __init__(self, hash_args, net_args):
        self.encoder = LegacyGSEncoder(**hash_args).cuda()
        self.spatial_dim = self.encoder.canonical_level_dim * self.encoder.canonical_num_levels
        self.mlp_dim = 10
        net_args = {key: value for key, value in dict(net_args).items() if key in {"depth", "width"}}
        self.decoder = LegacyGSDecoder(
            spatial_in_dim=self.spatial_dim,
            mlp_in_dim=self.mlp_dim,
            **net_args,
        ).cuda()
        self.decoder_lr_scale = 50.0
        self.encoder_lr_scale = 100.0

    def step(self, data, pose):
        coords = data[0]
        scale_input = data[1]
        rotate_input = data[2]
        spatial_h, temporal_h = self.encoder(coords, pose)
        color, rotation, scaling, opacity = self.decoder(spatial_h, temporal_h, scale_input, rotate_input)
        return {
            "d_color": color,
            "d_rotation": rotation,
            "d_scaling": scaling,
            "d_opacity": opacity,
        }

    def train_setting(self, training_args):
        self.decoder_lr_scale = training_args.decoder_lr_scale
        self.encoder_lr_scale = training_args.encoder_lr_scale
        l = [
            {
                "params": list(self.decoder.parameters()),
                "lr": training_args.position_lr_init * self.decoder_lr_scale,
                "name": "decoder",
            },
            {
                "params": list(self.encoder.parameters()),
                "lr": training_args.position_lr_init * self.encoder_lr_scale,
                "name": "encoder",
            },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.decoder_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.decoder_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.decoder_lr_max_steps,
        )
        self.encoder_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.encoder_lr_scale,
            lr_final=training_args.position_lr_final * self.encoder_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.decoder_lr_max_steps,
        )

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "decoder"))
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(loaded_iter))
        else:
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(iteration))

        print("Load weight:", weights_path)
        grid_weight, network_weight = torch.load(weights_path, map_location="cuda")
        try:
            self.decoder.load_state_dict(network_weight)
            self.encoder.load_state_dict(grid_weight)
        except RuntimeError as exc:
            raise RuntimeError(
                "Legacy evaluator can only load original MixGS-compatible checkpoints. "
                "Use render_mixgs_research_eval.py for research checkpoints."
            ) from exc


def _camera_pose(viewpoint):
    transform = viewpoint["world_view_transform"]
    if isinstance(transform, torch.Tensor):
        while transform.dim() > 2:
            transform = transform[0]
    return transform[-1, :-1]


def _viewpoint_from_camera(dataset, cam_info, idx):
    viewpoint_cam = loadCam(dataset, idx, cam_info, 1)
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
    gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0)
    return viewpoint, gt_image


def _training_report_train_cameras(scene):
    train_cameras = scene.getTrainCameras()
    if len(train_cameras) == 0:
        return []
    return [train_cameras[idx % len(train_cameras)] for idx in range(5, 30, 5)]


def _save_render_outputs(base_path, idx, image, gt_image, depth):
    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "gt")
    depth_path = os.path.join(base_path, "depth")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    depth = colorize(depth.cpu().squeeze(0), cmap_name="jet")
    torchvision.utils.save_image(image, os.path.join(render_path, f"{idx:05d}.png"))
    torchvision.utils.save_image(gt_image, os.path.join(gts_path, f"{idx:05d}.png"))
    torchvision.utils.save_image(depth.permute(2, 0, 1).unsqueeze(0), os.path.join(depth_path, f"{idx:05d}.png"))


def _metric_values(image, gt_image):
    return {
        "l1_loss": float(l1_loss(image, gt_image).mean().item()),
        "psnr": float(psnr(image, gt_image).mean().item()),
        "ssim": float(ssim(image, gt_image).mean().item()),
        "lpips": float(lpips(image, gt_image, net_type="vgg").mean().item()),
    }


def _write_metrics(path, metrics, filename):
    with open(os.path.join(path, filename), "w") as fp:
        json.dump(metrics, fp, indent=2)


def _compute_saved_metrics(base_dir):
    render_path = Path(base_dir) / "renders"
    gt_path = Path(base_dir) / "gt"
    image_names = sorted(os.listdir(render_path))
    totals = {"ssim": 0.0, "psnr": 0.0, "lpips": 0.0}

    for fname in tqdm(image_names, desc=f"Saved metrics {base_dir}"):
        render = Image.open(render_path / fname)
        gt = Image.open(gt_path / fname)
        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        render.close()
        gt.close()
        totals["ssim"] += float(ssim(render_tensor, gt_tensor).mean().item())
        totals["psnr"] += float(psnr(render_tensor, gt_tensor).mean().item())
        totals["lpips"] += float(lpips(render_tensor, gt_tensor, net_type="vgg").mean().item())

    count = max(len(image_names), 1)
    metrics = {key: value / count for key, value in totals.items()}
    _write_metrics(base_dir, metrics, "metrics_saved.json")
    return metrics


def legacy_render_mix(viewpoint_camera, pc, pipe, bg_color, vis_mask, decoded_data, scaling_modifier=1.0):
    ori_xyz = pc.get_xyz[vis_mask].detach()
    ori_rot = pc.get_rotation[vis_mask].detach()

    d_scaling = decoded_data["d_scaling"].to(torch.float32)
    d_rotation = decoded_data["d_rotation"].to(torch.float32)
    d_sh = decoded_data["d_color"].to(torch.float32)
    d_opacity = decoded_data["d_opacity"].to(torch.float32)

    num = len(d_scaling)
    if num > 0:
        offsets = pc.get_offset[vis_mask]
        if offsets.dim() == 3:
            offsets = offsets[:, 0, :]
        means3D = ori_xyz + offsets.reshape(num, -1)[:, :3]
        rotations = pc.rotation_activation(ori_rot + d_rotation)
    else:
        means3D = ori_xyz.new_empty((0, 3))
        rotations = ori_rot.new_empty((0, 4))

    pc_features = pc.get_features[vis_mask].transpose(1, 2)
    shs_view = pc_features.view(pc_features.shape[0], -1, (pc.max_sh_degree + 1) ** 2)
    dir_pp = pc.get_xyz[vis_mask] - viewpoint_camera["camera_center"].repeat(pc_features.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    colors_precomp = torch.cat([torch.sigmoid(d_sh), colors_precomp], dim=0)

    opacity = d_opacity
    res_scales = torch.clamp_min(d_scaling, pipe.scale_min)

    ori_means3D = pc.get_xyz[vis_mask]
    ori_opacity = pc.get_opacity[vis_mask]
    ori_scales = pc.get_scaling[vis_mask]
    ori_rotations = pc.get_rotation[vis_mask]

    means3D = torch.cat([means3D, ori_means3D], dim=0)
    opacity = torch.cat([opacity, ori_opacity], dim=0)
    scales = torch.cat([res_scales, ori_scales], dim=0)
    rotations = torch.cat([rotations, ori_rotations], dim=0)

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, depth_image = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth_image,
        "scale": res_scales,
    }


def render_legacy_eval_set(
        dataset,
        name,
        iteration,
        cameras,
        scene,
        mixgs,
        pipeline,
        background,
        output_suffix,
        run_metrics=False,
        run_saved_metrics=False):
    if not cameras:
        return None

    split_name = f"{name}_{output_suffix}" if output_suffix else name
    base_dir = os.path.join(dataset.model_path, split_name, f"ours_{iteration}")
    avg_render_time = 0.0
    max_render_time = 0.0
    avg_memory = 0.0
    max_memory = 0.0
    metric_totals = {"l1_loss": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

    for idx, cam_info in enumerate(tqdm(cameras, desc=f"Rendering {split_name}")):
        viewpoint, gt_image = _viewpoint_from_camera(dataset, cam_info, idx)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        vis_mask = prefilter_voxel(viewpoint, scene.gaussians, pipeline, background)
        hash_input = [
            scene.gaussians.get_xyz[vis_mask].detach(),
            scene.gaussians.get_scaling[vis_mask].detach(),
            scene.gaussians.get_rotation[vis_mask].detach(),
        ]
        decoded_data = mixgs.step(hash_input, _camera_pose(viewpoint))
        render_pkg = legacy_render_mix(viewpoint, scene.gaussians, pipeline, background, vis_mask, decoded_data)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_render_time += elapsed
        max_render_time = max(max_render_time, elapsed)
        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)

        if run_metrics:
            values = _metric_values(image, gt_image)
            for key, value in values.items():
                metric_totals[key] += value

        _save_render_outputs(base_dir, idx, image, gt_image, render_pkg["depth"])

    count = len(cameras)
    summary = {
        "Split": split_name,
        "Mode": "legacy",
        "Iteration": int(iteration),
        "View Count": count,
        "Average FPS": count / avg_render_time if avg_render_time > 0 else 0.0,
        "Min FPS": 1.0 / max_render_time if max_render_time > 0 else 0.0,
        "Average Memory(M)": avg_memory / count,
        "Max Memory(M)": max_memory,
        "Number of Gaussians": int(scene.gaussians.get_xyz.shape[0]),
    }
    if run_metrics:
        metrics = {key: value / count for key, value in metric_totals.items()}
        summary["Metrics"] = metrics
        _write_metrics(base_dir, metrics, "metrics.json")
        print(f"[{split_name}] L1    : {metrics['l1_loss']:>12.7f}")
        print(f"[{split_name}] PSNR  : {metrics['psnr']:>12.7f}")
        print(f"[{split_name}] SSIM  : {metrics['ssim']:>12.7f}")
        print(f"[{split_name}] LPIPS : {metrics['lpips']:>12.7f}")
    if run_saved_metrics:
        summary["Saved Metrics"] = _compute_saved_metrics(base_dir)

    with open(os.path.join(dataset.model_path, "costs.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[{split_name}] Average FPS: {summary['Average FPS']:.4f}")
    print(f"[{split_name}] Min FPS: {summary['Min FPS']:.4f}")
    print(f"[{split_name}] Average Memory: {summary['Average Memory(M)']:.4f} M")
    print(f"[{split_name}] Max Memory: {summary['Max Memory(M)']:.4f} M")
    print(f"[{split_name}] Output: {base_dir}")
    return summary


def render_sets(dataset: ModelParams, opt, iteration: int, pipeline: PipelineParams,
                load_vq: bool, skip_train: bool, skip_test: bool, output_suffix: str,
                run_metrics: bool, run_saved_metrics: bool):
    with torch.no_grad():
        modules = __import__("scene")
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config["name"])(dataset.sh_degree, **model_config["kwargs"])
        scene = LargeScene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)

        mixgs = LegacyMixGSModel(hash_args=dataset.hash_args, net_args=dataset.network_args)
        mixgs.load_weights(dataset.model_path, iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        summaries = []
        if not skip_train:
            summaries.append(render_legacy_eval_set(
                dataset,
                "train",
                iteration,
                _training_report_train_cameras(scene),
                scene,
                mixgs,
                pipeline,
                background,
                output_suffix,
                run_metrics=run_metrics,
                run_saved_metrics=run_saved_metrics,
            ))
        if not skip_test:
            summaries.append(render_legacy_eval_set(
                dataset,
                "test",
                iteration,
                scene.getTestCameras(),
                scene,
                mixgs,
                pipeline,
                background,
                output_suffix,
                run_metrics=run_metrics,
                run_saved_metrics=run_saved_metrics,
            ))
        return summaries


if __name__ == "__main__":
    parser = ArgumentParser(description="Render and evaluate original MixGS checkpoints")
    parser.add_argument("--config", type=str, required=True, help="train config file path")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--iteration", default=300000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="legacy_eval")
    parser.add_argument("--run_metrics", action="store_true")
    parser.add_argument("--run_saved_metrics", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.model_path is None:
        args.model_path = os.path.join("output", os.path.basename(args.config).split(".")[0])

    resource.setrlimit(resource.RLIMIT_NOFILE, [11264, 65535])
    safe_state(args.quiet)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)

    render_sets(
        lp,
        op,
        args.iteration,
        pp,
        args.load_vq,
        args.skip_train,
        args.skip_test,
        args.output_suffix,
        args.run_metrics,
        args.run_saved_metrics,
    )
