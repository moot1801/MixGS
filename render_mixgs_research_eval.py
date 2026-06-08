#
# Research MixGS evaluation renderer.
#
# This script uses the current research MixGS runtime path and mirrors the
# train_mixgs.py evaluation render path as closely as possible.
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
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as tf
import yaml
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import prefilter_voxel
from lpipsPyTorch import lpips
from scene import LargeScene, MixGSModel
from train_mixgs import (
    StageBudgetDecaySchedule,
    _is_projected_area_mode,
    _render_with_proposal_budget,
    _resolve_iteration_render_budget,
)
from utils.camera_utils import loadCam
from utils.general_utils import colorize, parse_cfg, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


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


def _rasterize_detail_count_map(viewpoint, pc, pipe, visible_cache, decoded_data, max_detail_slots):
    detail_counts = decoded_data.get("detail_counts") if decoded_data is not None else None
    if detail_counts is None or visible_cache is None:
        return None, None

    visible_xyz = visible_cache.get("xyz")
    visible_rotation = visible_cache.get("rotation")
    visible_offset_slots = visible_cache.get("offset_slots")
    visible_opacity = visible_cache.get("opacity")
    visible_scaling = visible_cache.get("scaling")
    if visible_xyz is None or visible_xyz.numel() == 0:
        return None, None

    device = visible_xyz.device
    dtype = visible_xyz.dtype
    detail_counts = detail_counts.detach().to(device=device, dtype=torch.float32).flatten()
    visible_count = min(detail_counts.shape[0], visible_xyz.shape[0])
    if visible_count <= 0:
        return None, None

    visible_xyz = visible_xyz[:visible_count]
    visible_rotation = visible_rotation[:visible_count]
    visible_offset_slots = visible_offset_slots[:visible_count]
    visible_opacity = visible_opacity[:visible_count]
    visible_scaling = visible_scaling[:visible_count]
    detail_counts = detail_counts[:visible_count]

    d_scaling = decoded_data["d_scaling"].to(device=device, dtype=torch.float32)
    d_rotation = decoded_data["d_rotation"].to(device=device, dtype=torch.float32)
    d_opacity = decoded_data["d_opacity"].to(device=device, dtype=torch.float32)
    detail_anchor_idx = decoded_data.get("detail_anchor_idx")
    detail_slot_idx = decoded_data.get("detail_slot_idx")
    if detail_anchor_idx is None:
        detail_anchor_idx = torch.arange(d_scaling.shape[0], device=device, dtype=torch.long)
    else:
        detail_anchor_idx = detail_anchor_idx.to(device=device, dtype=torch.long)
    if detail_slot_idx is None:
        detail_slot_idx = torch.zeros(d_scaling.shape[0], device=device, dtype=torch.long)
    else:
        detail_slot_idx = detail_slot_idx.to(device=device, dtype=torch.long)

    valid_detail = (detail_anchor_idx >= 0) & (detail_anchor_idx < visible_count)
    detail_anchor_idx = detail_anchor_idx[valid_detail]
    detail_slot_idx = detail_slot_idx[valid_detail]
    d_scaling = d_scaling[valid_detail]
    d_rotation = d_rotation[valid_detail]
    d_opacity = d_opacity[valid_detail]

    if d_scaling.shape[0] > 0:
        detail_means3D = visible_xyz[detail_anchor_idx] + visible_offset_slots[detail_anchor_idx, detail_slot_idx]
        detail_rotations = pc.rotation_activation(visible_rotation[detail_anchor_idx] + d_rotation)
        detail_scales = torch.clamp_min(d_scaling, pipe.scale_min)
        detail_values = detail_counts[detail_anchor_idx]
    else:
        detail_means3D = visible_xyz.new_empty((0, 3))
        detail_rotations = visible_rotation.new_empty((0, 4))
        detail_scales = visible_scaling.new_empty((0, 3))
        detail_values = detail_counts.new_empty((0,))

    means3D = torch.cat([detail_means3D, visible_xyz], dim=0)
    scales = torch.cat([detail_scales, visible_scaling], dim=0)
    rotations = torch.cat([detail_rotations, visible_rotation], dim=0)
    opacities = torch.cat([d_opacity, visible_opacity], dim=0)
    values = torch.cat([detail_values, detail_counts], dim=0).to(dtype=torch.float32)
    max_detail = float(max(1, int(max_detail_slots)))
    normalized_values = torch.clamp(values / max_detail, 0.0, 1.0).unsqueeze(1).repeat(1, 3)

    screenspace_points = torch.zeros_like(means3D, dtype=dtype, device=device)
    tanfovx = math.tan(viewpoint["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint["FoVy"] * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint["image_height"]),
        image_width=int(viewpoint["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=viewpoint["world_view_transform"],
        projmatrix=viewpoint["full_proj_transform"],
        sh_degree=pc.active_sh_degree,
        campos=viewpoint["camera_center"],
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    detail_render, _, _ = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=normalized_values,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    coverage_render, _, _ = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=torch.ones_like(normalized_values),
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    coverage = coverage_render.mean(dim=0).clamp_min(0.0)
    detail_map = detail_render.mean(dim=0) / coverage.clamp_min(1e-6) * max_detail
    detail_map = torch.clamp(detail_map, 0.0, max_detail)
    return detail_map, coverage


def _resize_panel_height(panel, height):
    if panel.shape[-2] == height:
        return panel
    width = max(1, int(round(panel.shape[-1] * float(height) / float(panel.shape[-2]))))
    return F.interpolate(panel.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]


def _save_detail_map_outputs(
        base_path,
        idx,
        image,
        viewpoint,
        pc,
        pipe,
        visible_cache,
        decoded_data,
        max_detail_slots,
        overlay_alpha=0.32,
        cmap_name="turbo"):
    detail_path = os.path.join(base_path, "detail_map")
    heat_path = os.path.join(base_path, "detail_map_heat")
    panel_path = os.path.join(base_path, "detail_map_panel")
    makedirs(detail_path, exist_ok=True)
    makedirs(heat_path, exist_ok=True)
    makedirs(panel_path, exist_ok=True)

    detail_map, coverage_map = _rasterize_detail_count_map(
        viewpoint,
        pc,
        pipe,
        visible_cache,
        decoded_data,
        max_detail_slots,
    )
    if detail_map is None:
        return False

    max_detail = float(max(1, int(max_detail_slots)))
    density_mask = coverage_map > 1e-4
    heat = colorize(detail_map.cpu(), cmap_name=cmap_name, range=(0.0, max_detail))
    heat = heat.to(device=image.device, dtype=image.dtype).permute(2, 0, 1)
    heat_with_cbar = colorize(
        detail_map.cpu(),
        cmap_name=cmap_name,
        range=(0.0, max_detail),
        append_cbar=True,
        cbar_in_image=False,
    ).to(dtype=image.dtype).permute(2, 0, 1)

    detail_strength = torch.clamp(detail_map.to(device=image.device, dtype=image.dtype) / max_detail, 0.0, 1.0)
    density_alpha = density_mask.to(device=image.device, dtype=image.dtype)
    alpha_value = max(0.0, min(1.0, float(overlay_alpha)))
    alpha = (alpha_value * density_alpha * (0.25 + 0.75 * detail_strength)).unsqueeze(0)
    overlay = torch.clamp(image.detach() * (1.0 - alpha) + heat * alpha, 0.0, 1.0)

    torchvision.utils.save_image(overlay, os.path.join(detail_path, f"{idx:05d}.png"))
    torchvision.utils.save_image(heat_with_cbar, os.path.join(heat_path, f"{idx:05d}.png"))

    panel_height = image.shape[-2]
    heat_panel = _resize_panel_height(heat_with_cbar.to(device=image.device), panel_height)
    separator = torch.ones((3, panel_height, 8), dtype=image.dtype, device=image.device)
    panel = torch.cat([image.detach(), separator, heat_panel], dim=2)
    torchvision.utils.save_image(panel, os.path.join(panel_path, f"{idx:05d}.png"))
    return True


def _metric_values(image, gt_image):
    return {
        "l1_loss": float(l1_loss(image, gt_image).mean().item()),
        "psnr": float(psnr(image, gt_image).mean().item()),
        "ssim": float(ssim(image, gt_image).mean().item()),
        "lpips": float(lpips(image, gt_image, net_type="vgg").mean().item()),
    }


DETAIL_STAGE_NAMES = (
    "view_frustum",
    "importance_estimation",
    "allocation",
    "decoding",
    "rendering",
)


class CudaStageTimer:
    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._active = {}
        self._events = {name: [] for name in DETAIL_STAGE_NAMES}

    def start(self, name):
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._active[name] = event

    def stop(self, name):
        if not self.enabled:
            return
        start_event = self._active.pop(name, None)
        if start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._events.setdefault(name, []).append((start_event, end_event))

    def elapsed_ms(self):
        if not self.enabled:
            return {name: 0.0 for name in DETAIL_STAGE_NAMES}
        return {
            name: sum(start.elapsed_time(end) for start, end in self._events.get(name, []))
            for name in DETAIL_STAGE_NAMES
        }


def _new_detail_totals():
    return {name: {"sum_ms": 0.0, "max_ms": 0.0, "count": 0} for name in DETAIL_STAGE_NAMES}


def _accumulate_detail_timing(totals, elapsed_ms):
    for name in DETAIL_STAGE_NAMES:
        value = float(elapsed_ms.get(name, 0.0))
        totals[name]["sum_ms"] += value
        totals[name]["max_ms"] = max(totals[name]["max_ms"], value)
        totals[name]["count"] += 1


def _print_detail_timing(split_name, totals):
    print(f"[{split_name}] Detail timing (CUDA ms)")
    for name in DETAIL_STAGE_NAMES:
        row = totals[name]
        count = max(row["count"], 1)
        print(f"[{split_name}]   {name:<22} avg={row['sum_ms'] / count:>9.3f} max={row['max_ms']:>9.3f}")


def _visible_cache_from_mask(gaussians, vis_mask):
    return {
        "xyz": gaussians.get_xyz[vis_mask],
        "scaling": gaussians.get_scaling[vis_mask],
        "rotation": gaussians.get_rotation[vis_mask],
        "offset_slots": gaussians.get_offset_slots[vis_mask],
        "features": gaussians.get_features[vis_mask],
        "opacity": gaussians.get_opacity[vis_mask],
    }


def _fill_deferred_projected_stats(mixgs, decoded_data, budget_stats):
    projected_scores = decoded_data.pop("_projected_area_scores", None)
    if projected_scores is not None and "projected_area_score_mean" not in budget_stats:
        budget_stats.update(mixgs._projected_area_score_stats(projected_scores))


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


def render_research_eval_set(
        dataset,
        name,
        iteration,
        cameras,
        scene,
        mixgs,
        pipeline,
        background,
        budget_decay_schedule,
        joint_start_iter,
        output_suffix,
        run_metrics=False,
        run_saved_metrics=False,
        detail=False,
        detail_map=False,
        detail_map_alpha=0.32,
        detail_map_cmap="turbo"):
    if not cameras:
        return None

    split_name = f"{name}_{output_suffix}" if output_suffix else name
    base_dir = os.path.join(dataset.model_path, split_name, f"ours_{iteration}")
    avg_render_time = 0.0
    max_render_time = 0.0
    avg_memory = 0.0
    max_memory = 0.0
    budget_rows = []
    metric_totals = {"l1_loss": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    projected_area_mode = _is_projected_area_mode(pipeline)
    detail_enabled = bool(detail and projected_area_mode)
    detail_totals = _new_detail_totals() if detail_enabled else None
    detail_map_saved_count = 0
    if detail and not projected_area_mode:
        print(f"[{split_name}] --detail is only available for projected_area allocation mode; using aggregate timing only.")

    for idx, cam_info in enumerate(tqdm(cameras, desc=f"Rendering {split_name}")):
        viewpoint, gt_image = _viewpoint_from_camera(dataset, cam_info, idx)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        stage_timer = CudaStageTimer(detail_enabled)

        if detail_enabled:
            stage_timer.start("view_frustum")
        vis_mask = prefilter_voxel(viewpoint, scene.gaussians, pipeline, background)
        visible_anchor_count_tensor = vis_mask.sum()
        visible_cache = _visible_cache_from_mask(scene.gaussians, vis_mask) if (projected_area_mode or detail_map) else None
        if detail_enabled:
            stage_timer.stop("view_frustum")
        visible_anchor_count = int(visible_anchor_count_tensor.item())
        render_gaussian_budget = _resolve_iteration_render_budget(
            pipeline,
            visible_anchor_count,
            iteration,
            getattr(pipeline, "render_gaussian_budget", 0),
            budget_decay_schedule,
        )
        render_pkg, decoded_data = _render_with_proposal_budget(
            viewpoint,
            scene.gaussians,
            mixgs,
            pipeline,
            background,
            vis_mask,
            render_gaussian_budget,
            stage_timer=stage_timer if detail_enabled else None,
            visible_cache=visible_cache,
            include_projected_area_stats=not projected_area_mode,
            use_projected_area_optimizations=projected_area_mode,
        )
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        if detail_enabled:
            _accumulate_detail_timing(detail_totals, stage_timer.elapsed_ms())
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
        if detail_map:
            if _save_detail_map_outputs(
                    base_dir,
                    idx,
                    image,
                    viewpoint,
                    scene.gaussians,
                    pipeline,
                    visible_cache,
                    decoded_data,
                    mixgs.max_detail_slots,
                    overlay_alpha=detail_map_alpha,
                    cmap_name=detail_map_cmap):
                detail_map_saved_count += 1
        budget_stats = dict(render_pkg.get("budget_stats", {}))
        _fill_deferred_projected_stats(mixgs, decoded_data, budget_stats)
        budget_stats.update({
            "view_index": idx,
            "visible_anchor_count": visible_anchor_count,
            "render_gaussian_budget": int(render_gaussian_budget),
        })
        budget_rows.append(budget_stats)

    count = len(cameras)
    summary = {
        "Split": split_name,
        "Mode": "research",
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
        "Detail Map Saved Count": int(detail_map_saved_count),
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
    if detail_enabled:
        _print_detail_timing(split_name, detail_totals)
    if detail_map:
        print(f"[{split_name}] Detail maps saved: {detail_map_saved_count}")
    print(f"[{split_name}] Output: {base_dir}")
    return summary


def render_sets(dataset: ModelParams, opt, iteration: int, pipeline: PipelineParams,
                load_vq: bool, skip_train: bool, skip_test: bool, output_suffix: str,
                run_metrics: bool, run_saved_metrics: bool, detail: bool, detail_map: bool,
                detail_map_alpha: float, detail_map_cmap: str):
    with torch.no_grad():
        modules = __import__("scene")
        model_config = dataset.model_config
        model_kwargs = dict(model_config["kwargs"])
        detail_count_choices = getattr(dataset, "detail_count_choices", None)
        model_kwargs.setdefault("max_detail_slots", getattr(dataset, "detail_max_slots", 1))
        model_kwargs.setdefault("detail_count_choices", detail_count_choices)
        gaussians = getattr(modules, model_config["name"])(dataset.sh_degree, **model_kwargs)
        scene = LargeScene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)

        use_slot_embedding = bool(getattr(pipeline, "use_slot_embedding", False))
        mixgs = MixGSModel(
            hash_args=dataset.hash_args,
            net_args=dataset.network_args,
            max_detail_slots=getattr(dataset, "detail_max_slots", 1),
            detail_count_choices=detail_count_choices,
            gate_feature_mode=getattr(pipeline, "gate_feature_mode", "detail_view"),
            gate_view_context_dim=getattr(pipeline, "gate_view_context_dim", 32),
            use_slot_embedding=use_slot_embedding,
        )
        mixgs.load_weights(dataset.model_path, iteration)
        budget_decay_schedule = StageBudgetDecaySchedule(
            pipeline,
            mixgs.max_detail_slots,
            getattr(opt, "joint_start_iter", getattr(opt, "iterations", 0) + 1),
        )
        joint_start_iter = budget_decay_schedule.effective_joint_start_iter

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        summaries = []
        if not skip_train:
            summaries.append(render_research_eval_set(
                dataset,
                "train",
                iteration,
                _training_report_train_cameras(scene),
                scene,
                mixgs,
                pipeline,
                background,
                budget_decay_schedule,
                joint_start_iter,
                output_suffix,
                run_metrics=run_metrics,
                run_saved_metrics=run_saved_metrics,
                detail=detail,
                detail_map=detail_map,
                detail_map_alpha=detail_map_alpha,
                detail_map_cmap=detail_map_cmap,
            ))
        if not skip_test:
            summaries.append(render_research_eval_set(
                dataset,
                "test",
                iteration,
                scene.getTestCameras(),
                scene,
                mixgs,
                pipeline,
                background,
                budget_decay_schedule,
                joint_start_iter,
                output_suffix,
                run_metrics=run_metrics,
                run_saved_metrics=run_saved_metrics,
                detail=detail,
                detail_map=detail_map,
                detail_map_alpha=detail_map_alpha,
                detail_map_cmap=detail_map_cmap,
            ))
        return summaries


if __name__ == "__main__":
    parser = ArgumentParser(description="Render and evaluate research MixGS checkpoints")
    parser.add_argument("--config", type=str, required=True, help="train config file path")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--iteration", default=300000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--output_suffix", type=str, default="research_eval")
    parser.add_argument("--run_metrics", action="store_true")
    parser.add_argument("--run_saved_metrics", action="store_true")
    parser.add_argument("--detail", action="store_true", help="print per-stage CUDA timing for projected_area rendering")
    parser.add_argument("--detail_map", action="store_true", help="save rendered-image overlays of per-anchor detail count distribution")
    parser.add_argument("--detail_map_alpha", type=float, default=0.32, help="overlay opacity for --detail_map outputs")
    parser.add_argument("--detail_map_cmap", type=str, default="turbo", help="matplotlib colormap name for --detail_map outputs")
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
        args.detail,
        args.detail_map,
        args.detail_map_alpha,
        args.detail_map_cmap,
    )
