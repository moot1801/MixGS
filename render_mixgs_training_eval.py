#
# Training-eval compatible renderer for MixGS.
#

import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as tf
import yaml
from PIL import Image
from argparse import ArgumentParser
from os import makedirs
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import prefilter_voxel
from lpipsPyTorch import lpips
from scene import LargeScene, MixGSModel
from scene.allocation_score import build_allocation_scorer
from train_mixgs import (
    StageBudgetDecaySchedule,
    _render_with_allocation,
    _resolve_iteration_render_budget,
)
from utils.camera_utils import loadCam
from utils.general_utils import colorize, parse_cfg, safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim


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


def _compute_saved_metrics(base_dir):
    render_path = Path(base_dir) / "renders"
    gt_path = Path(base_dir) / "gt"
    image_names = sorted(os.listdir(render_path))
    ssims = []
    psnrs = []
    lpipss = []

    for fname in tqdm(image_names, desc=f"Metrics {base_dir}"):
        render = Image.open(render_path / fname)
        gt = Image.open(gt_path / fname)
        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        render.close()
        gt.close()
        ssims.append(ssim(render_tensor, gt_tensor))
        psnrs.append(psnr(render_tensor, gt_tensor))
        lpipss.append(lpips(render_tensor, gt_tensor, net_type="vgg"))

    metrics = {
        "SSIM": float(torch.tensor(ssims).mean().item()) if ssims else 0.0,
        "PSNR": float(torch.tensor(psnrs).mean().item()) if psnrs else 0.0,
        "LPIPS": float(torch.tensor(lpipss).mean().item()) if lpipss else 0.0,
    }
    with open(Path(base_dir) / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    print("[{}] SSIM : {:>12.7f}".format(base_dir, metrics["SSIM"]))
    print("[{}] PSNR : {:>12.7f}".format(base_dir, metrics["PSNR"]))
    print("[{}] LPIPS: {:>12.7f}".format(base_dir, metrics["LPIPS"]))
    return metrics


def render_training_eval_set(
        dataset,
        name,
        iteration,
        cameras,
        scene,
        mixgs,
        pipeline,
        background,
        allocation_scorer,
        budget_decay_schedule,
        joint_start_iter,
        output_suffix,
        run_metrics=False):
    if not cameras:
        return None

    split_name = f"{name}_{output_suffix}" if output_suffix else name
    base_dir = os.path.join(dataset.model_path, split_name, f"ours_{iteration}")
    avg_render_time = 0.0
    max_render_time = 0.0
    avg_memory = 0.0
    max_memory = 0.0
    budget_rows = []

    for idx, cam_info in enumerate(tqdm(cameras, desc=f"Rendering {split_name}")):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        viewpoint, gt_image = _viewpoint_from_camera(dataset, cam_info, idx)
        vis_mask = prefilter_voxel(viewpoint, scene.gaussians, pipeline, background)
        visible_anchor_count = int(vis_mask.sum().item())
        render_gaussian_budget = _resolve_iteration_render_budget(
            pipeline,
            visible_anchor_count,
            iteration,
            getattr(pipeline, "render_gaussian_budget", 0),
            budget_decay_schedule,
        )
        visible_anchor_indices = torch.nonzero(vis_mask, as_tuple=False).flatten()
        render_pkg, decoded_data, _ = _render_with_allocation(
            viewpoint,
            gt_image,
            scene.gaussians,
            mixgs,
            pipeline,
            background,
            vis_mask,
            render_gaussian_budget,
            allocation_scorer,
            bool(getattr(pipeline, "allocation_score_eval_uses_gt", False)),
            iteration=iteration,
            joint_start_iter=joint_start_iter,
            anchor_indices=visible_anchor_indices,
        )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        _save_render_outputs(base_dir, idx, image, gt_image, render_pkg["depth"])

        torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_render_time += elapsed
        max_render_time = max(max_render_time, elapsed)
        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)
        budget_stats = dict(render_pkg.get("budget_stats", {}))
        budget_stats.update({
            "view_index": idx,
            "visible_anchor_count": visible_anchor_count,
            "render_gaussian_budget": int(render_gaussian_budget),
        })
        budget_rows.append(budget_stats)

    count = len(cameras)
    summary = {
        "Split": split_name,
        "Iteration": int(iteration),
        "View Count": count,
        "Average FPS": count / avg_render_time if avg_render_time > 0 else 0.0,
        "Min FPS": 1.0 / max_render_time if max_render_time > 0 else 0.0,
        "Average Memory(M)": avg_memory / count,
        "Max Memory(M)": max_memory,
        "Number of Gaussians": int(scene.gaussians.get_xyz.shape[0]),
        "Stage Budget Decay": bool(budget_decay_schedule.enabled),
        "Effective Joint Start Iter": int(joint_start_iter),
        "Budget Stats": budget_rows,
    }
    with open(os.path.join(base_dir, "costs.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    avg_fps = summary["Average FPS"]
    min_fps = summary["Min FPS"]
    avg_memory_mb = summary["Average Memory(M)"]
    max_memory_mb = summary["Max Memory(M)"]
    print(f"[{split_name}] Average FPS: {avg_fps:.4f}")
    print(f"[{split_name}] Min FPS: {min_fps:.4f}")
    print(f"[{split_name}] Average Memory: {avg_memory_mb:.4f} M")
    print(f"[{split_name}] Max Memory: {max_memory_mb:.4f} M")
    print(f"[{split_name}] Output: {base_dir}")
    if run_metrics:
        summary["Metrics"] = _compute_saved_metrics(base_dir)
        with open(os.path.join(base_dir, "costs.json"), "w") as fp:
            json.dump(summary, fp, indent=2)
    return summary


def render_sets(dataset: ModelParams, opt, iteration: int, pipeline: PipelineParams,
                load_vq: bool, skip_train: bool, skip_test: bool, output_suffix: str, run_metrics: bool):
    with torch.no_grad():
        modules = __import__("scene")
        model_config = dataset.model_config
        model_kwargs = dict(model_config["kwargs"])
        detail_count_choices = getattr(dataset, "detail_count_choices", None)
        model_kwargs.setdefault("max_detail_slots", getattr(dataset, "detail_max_slots", 1))
        model_kwargs.setdefault("detail_count_choices", detail_count_choices)
        gaussians = getattr(modules, model_config["name"])(dataset.sh_degree, **model_kwargs)
        scene = LargeScene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)

        mixgs = MixGSModel(
            hash_args=dataset.hash_args,
            net_args=dataset.network_args,
            max_detail_slots=getattr(dataset, "detail_max_slots", 1),
            detail_count_choices=detail_count_choices,
        )
        mixgs.load_weights(dataset.model_path, iteration)
        allocation_scorer = build_allocation_scorer(getattr(pipeline, "allocation_score", None))
        budget_decay_schedule = StageBudgetDecaySchedule(
            pipeline,
            mixgs.max_detail_slots,
            getattr(opt, "joint_start_iter", getattr(opt, "iterations", 0) + 1),
        )
        joint_start_iter = budget_decay_schedule.effective_joint_start_iter

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        summaries = []
        if not skip_test:
            summaries.append(render_training_eval_set(
                dataset,
                "test",
                iteration,
                scene.getTestCameras(),
                scene,
                mixgs,
                pipeline,
                background,
                allocation_scorer,
                budget_decay_schedule,
                joint_start_iter,
                output_suffix,
                run_metrics=run_metrics,
            ))
        if not skip_train:
            summaries.append(render_training_eval_set(
                dataset,
                "train",
                iteration,
                _training_report_train_cameras(scene),
                scene,
                mixgs,
                pipeline,
                background,
                allocation_scorer,
                budget_decay_schedule,
                joint_start_iter,
                output_suffix,
                run_metrics=run_metrics,
            ))
        return summaries


if __name__ == "__main__":
    parser = ArgumentParser(description="Render with the same path as train_mixgs.py metric evaluation")
    parser.add_argument("--config", type=str, required=True, help="train config file path")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--iteration", default=300000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="training_eval")
    parser.add_argument("--run_metrics", action="store_true")
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
    )
