import json
import math
import os
import resource
import sys
from argparse import ArgumentParser
from os import makedirs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
import torchvision
import yaml
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import prefilter_voxel
from scene import LargeScene
from utils.camera_utils import loadCam
from utils.general_utils import colorize, parse_cfg, safe_state


DEFAULT_METHODS = (
    "projected_area",
    "tile_complexity",
    "image_edge",
    "harris_corner",
    "laplacian_blob",
    "entropy",
    "depth_boundary",
    "curvature",
    "density_penalized_edge",
    "coarse_residual",
    "hybrid_structure",
    "projection_image_edge",
    "projection_density_penalized_edge",
    "projection_entropy",
    "projection_harris_corner",
    "projection_laplacian_blob",
    "projection_hybrid_structure",
    "projection_knn_laplacian",
    "projection_plane_residual",
    "projection_multiscale_plane_residual",
    "projection_depth_layer_boundary",
    "projection_occlusion_support",
    "projection_support_gated_complexity",
    "projection_density_suppressed_curvature",
    "projection_gain_proxy",
    "projection_structure_hybrid",
    "projection_flat_surface_suppressed_boundary",
    "projection_component_penalized_structure",
    "projection_object_boundary_gain",
    "projection_learnable_detail_gain",
    "projected_area_flat_suppressed",
    "projection_inverse_common_bias",
    "projection_hard_reject_flat_dense",
    "projection_mid_support_anti_flat",
    "projection_anti_road_water_shadow",
    "projection_inverse_area_with_visibility_gate",
)


def _clamp01(tensor):
    return torch.clamp(tensor, 0.0, 1.0)


def _normalize_positive(values, eps=1e-6, quantile=0.95):
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if values.numel() == 0:
        return values
    positive = values[values > eps]
    if positive.numel() == 0:
        return torch.zeros_like(values)
    scale = torch.quantile(positive, float(quantile)).clamp_min(eps)
    return torch.clamp(values / scale, 0.0, 1.0)


def _camera_center(viewpoint, device, dtype):
    center = viewpoint["camera_center"]
    if isinstance(center, torch.Tensor):
        center = center.detach().to(device=device, dtype=dtype)
    else:
        center = torch.as_tensor(center, device=device, dtype=dtype)
    while center.dim() > 1:
        center = center[0]
    return center


def _viewpoint_tensor(viewpoint, key, device, dtype):
    value = viewpoint[key]
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


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
    gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda")[:3], 0.0, 1.0)
    return viewpoint, gt_image


def _visible_cache_from_mask(gaussians, vis_mask):
    return {
        "xyz": gaussians.get_xyz[vis_mask],
        "scaling": gaussians.get_scaling[vis_mask],
        "rotation": gaussians.get_rotation[vis_mask],
        "features": gaussians.get_features[vis_mask],
        "color_dc": gaussians.get_features_dc[vis_mask, 0, :],
        "opacity": gaussians.get_opacity[vis_mask],
    }


def _project_to_pixels(viewpoint, coords, eps=1e-6):
    device = coords.device
    dtype = torch.float32
    image_height = int(viewpoint["image_height"])
    image_width = int(viewpoint["image_width"])
    full_proj = _viewpoint_tensor(viewpoint, "full_proj_transform", device, dtype)
    coords_f = coords.detach().to(dtype=dtype)
    ones = torch.ones((coords_f.shape[0], 1), device=device, dtype=dtype)
    hom = torch.cat([coords_f, ones], dim=-1)
    clip = hom @ full_proj
    w = clip[:, 3].clamp_min(float(eps))
    ndc = torch.nan_to_num(clip[:, :3] / w.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
    pixel_x = torch.floor((ndc[:, 0] + 1.0) * 0.5 * float(image_width)).long()
    pixel_y = torch.floor((ndc[:, 1] + 1.0) * 0.5 * float(image_height)).long()
    pixel_x = pixel_x.clamp_(0, image_width - 1)
    pixel_y = pixel_y.clamp_(0, image_height - 1)
    return pixel_x, pixel_y, ndc


def _tile_indices(pixel_x, pixel_y, image_width, image_height, tile_size):
    tile_size = max(1, int(tile_size))
    tile_w = max(1, (int(image_width) + tile_size - 1) // tile_size)
    tile_h = max(1, (int(image_height) + tile_size - 1) // tile_size)
    tile_idx = (pixel_y // tile_size) * tile_w + (pixel_x // tile_size)
    tile_idx = tile_idx.clamp_(0, tile_w * tile_h - 1)
    return tile_idx, tile_w, tile_h


def _tile_counts(tile_idx, tile_count, dtype, device):
    counts = torch.zeros(tile_count, dtype=dtype, device=device)
    counts.index_add_(0, tile_idx, torch.ones_like(tile_idx, dtype=dtype))
    return counts


def _tile_mean(values, tile_idx, tile_count, fill_value=None):
    dtype = values.dtype
    device = values.device
    sums = torch.zeros(tile_count, dtype=dtype, device=device)
    counts = _tile_counts(tile_idx, tile_count, dtype, device)
    sums.index_add_(0, tile_idx, values)
    means = sums / counts.clamp_min(1.0)
    if fill_value is not None:
        fill = torch.as_tensor(fill_value, dtype=dtype, device=device)
        means = torch.where(counts > 0.0, means, fill)
    return means, counts


def _tile_var(values, tile_idx, tile_count, fill_value=0.0):
    means, counts = _tile_mean(values, tile_idx, tile_count, fill_value=fill_value)
    sums2 = torch.zeros(tile_count, dtype=values.dtype, device=values.device)
    sums2.index_add_(0, tile_idx, values * values)
    var = sums2 / counts.clamp_min(1.0) - means * means
    return torch.clamp_min(var, 0.0), means, counts


def _sobel_response(map_2d):
    device = map_2d.device
    dtype = map_2d.dtype
    gx_kernel = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    gy_kernel = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    tensor = map_2d.view(1, 1, map_2d.shape[-2], map_2d.shape[-1])
    tensor = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(tensor, gx_kernel)[0, 0]
    gy = F.conv2d(tensor, gy_kernel)[0, 0]
    return torch.sqrt(gx * gx + gy * gy).clamp_min(0.0), gx, gy


def _harris_response(map_2d):
    _, gx, gy = _sobel_response(map_2d)
    ixx = F.avg_pool2d((gx * gx).view(1, 1, *gx.shape), 3, stride=1, padding=1)[0, 0]
    iyy = F.avg_pool2d((gy * gy).view(1, 1, *gy.shape), 3, stride=1, padding=1)[0, 0]
    ixy = F.avg_pool2d((gx * gy).view(1, 1, *gx.shape), 3, stride=1, padding=1)[0, 0]
    return _normalize_positive(torch.clamp(ixx * iyy - ixy * ixy - 0.04 * (ixx + iyy) * (ixx + iyy), min=0.0))


def _laplacian_abs(map_2d):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=map_2d.device,
        dtype=map_2d.dtype,
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(F.pad(map_2d.view(1, 1, *map_2d.shape), (1, 1, 1, 1), mode="replicate"), kernel)[0, 0]
    return _normalize_positive(torch.abs(laplacian))


def _image_feature_maps(image):
    image = image.float()
    gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    edge, gx, gy = _sobel_response(gray)
    edge = _normalize_positive(edge)

    ixx = F.avg_pool2d((gx * gx).view(1, 1, *gx.shape), 5, stride=1, padding=2)[0, 0]
    iyy = F.avg_pool2d((gy * gy).view(1, 1, *gy.shape), 5, stride=1, padding=2)[0, 0]
    ixy = F.avg_pool2d((gx * gy).view(1, 1, *gx.shape), 5, stride=1, padding=2)[0, 0]
    harris = torch.clamp(ixx * iyy - ixy * ixy - 0.04 * (ixx + iyy) * (ixx + iyy), min=0.0)
    harris = _normalize_positive(harris)

    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(F.pad(gray.view(1, 1, *gray.shape), (1, 1, 1, 1), mode="replicate"), laplacian_kernel)[0, 0]
    blob = _normalize_positive(torch.abs(laplacian))
    entropy = _local_entropy(gray)
    return {
        "edge": edge,
        "harris": harris,
        "blob": blob,
        "entropy": entropy,
    }


def _local_entropy(gray, bins=16, window_size=11, eps=1e-6):
    bins = max(2, int(bins))
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    gray = torch.clamp(gray.float(), 0.0, 1.0)
    quantized = torch.clamp((gray * bins).long(), 0, bins - 1)
    one_hot = F.one_hot(quantized, num_classes=bins).permute(2, 0, 1).unsqueeze(0).float()
    kernel = torch.ones(
        (bins, 1, window_size, window_size),
        dtype=one_hot.dtype,
        device=one_hot.device,
    )
    counts = F.conv2d(
        F.pad(one_hot, (window_size // 2,) * 4, mode="reflect"),
        kernel,
        groups=bins,
    )
    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(float(eps))
    entropy = -(probs * torch.log2(probs.clamp_min(float(eps)))).sum(dim=1)[0]
    return _normalize_positive(entropy, eps=eps)


def _sample_map(feature_map, pixel_x, pixel_y):
    return feature_map[pixel_y, pixel_x]


def _raster_settings(viewpoint, pc, pipe, device):
    tanfovx = math.tan(viewpoint["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint["FoVy"] * 0.5)
    return GaussianRasterizationSettings(
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


def _rasterize_anchor_values(viewpoint, pc, pipe, cache, values, value_range):
    coords = cache["xyz"]
    if coords.numel() == 0:
        return None, None
    device = coords.device
    values = values.detach().to(device=device, dtype=torch.float32).flatten()
    colors = torch.clamp(values / max(float(value_range), 1e-6), 0.0, 1.0).unsqueeze(1).repeat(1, 3)
    screenspace_points = torch.zeros_like(coords, dtype=coords.dtype, device=device)
    rasterizer = GaussianRasterizer(raster_settings=_raster_settings(viewpoint, pc, pipe, device))

    value_render, _, _ = rasterizer(
        means3D=coords,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors,
        opacities=cache["opacity"],
        scales=cache["scaling"],
        rotations=cache["rotation"],
        cov3D_precomp=None,
    )
    coverage_render, _, _ = rasterizer(
        means3D=coords,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=torch.ones_like(colors),
        opacities=cache["opacity"],
        scales=cache["scaling"],
        rotations=cache["rotation"],
        cov3D_precomp=None,
    )
    coverage = coverage_render.mean(dim=0).clamp_min(0.0)
    value_map = value_render.mean(dim=0) / coverage.clamp_min(1e-6) * float(value_range)
    return torch.clamp(value_map, 0.0, float(value_range)), coverage


def _render_coarse_residual_map(viewpoint, pc, pipe, cache, image):
    coords = cache["xyz"]
    if coords.numel() == 0:
        return None
    device = coords.device
    color = torch.clamp(cache["color_dc"].detach().to(device=device, dtype=torch.float32) + 0.5, 0.0, 1.0)
    if color.dim() == 3:
        color = color[:, 0, :]
    screenspace_points = torch.zeros_like(coords, dtype=coords.dtype, device=device)
    rasterizer = GaussianRasterizer(raster_settings=_raster_settings(viewpoint, pc, pipe, device))
    rendered, _, _ = rasterizer(
        means3D=coords,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=cache["opacity"],
        scales=cache["scaling"],
        rotations=cache["rotation"],
        cov3D_precomp=None,
    )
    return torch.abs(torch.clamp(rendered, 0.0, 1.0) - image).mean(dim=0)


def _build_context(viewpoint, image, cache, tile_size, residual_map=None, build_image_features=True, eps=1e-6):
    coords = cache["xyz"]
    device = coords.device
    dtype = torch.float32
    pixel_x, pixel_y, _ = _project_to_pixels(viewpoint, coords, eps=eps)
    tile_idx, tile_w, tile_h = _tile_indices(
        pixel_x,
        pixel_y,
        int(viewpoint["image_width"]),
        int(viewpoint["image_height"]),
        tile_size,
    )
    tile_count = tile_w * tile_h

    center = _camera_center(viewpoint, device, dtype)
    coords_f = coords.detach().to(dtype=dtype)
    depth = (coords_f - center.unsqueeze(0)).norm(dim=-1).clamp_min(eps)
    scaling = cache["scaling"].detach().to(dtype=dtype)
    scale_mean = torch.abs(scaling[:, :3]).mean(dim=-1).clamp_min(eps)
    scale_std = torch.std(torch.abs(scaling[:, :3]), dim=-1, unbiased=False)
    scale_anisotropy = scale_std / scale_mean
    projected_area = scale_mean * scale_mean / (depth * depth).clamp_min(eps)

    color = cache["color_dc"].detach().to(dtype=dtype)
    if color.dim() == 3:
        color = color[:, 0, :]
    color = torch.clamp(color + 0.5, 0.0, 1.0)
    opacity = cache["opacity"].detach().to(dtype=dtype).reshape(-1).clamp(0.0, 1.0)
    features = _image_feature_maps(image) if build_image_features else None
    counts = _tile_counts(tile_idx, tile_count, dtype, device)

    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "tile_idx": tile_idx,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "tile_count": tile_count,
        "tile_counts": counts,
        "image_width": int(viewpoint["image_width"]),
        "image_height": int(viewpoint["image_height"]),
        "xyz": coords_f,
        "depth": depth,
        "scale_mean": scale_mean,
        "scale_anisotropy": scale_anisotropy,
        "projected_area": projected_area,
        "color": color,
        "opacity": opacity,
        "features": features,
        "residual_map": residual_map,
        "projection_proxy": None,
        "neighbor_proxy": {},
    }


def _score_projected_area(ctx):
    return _normalize_positive(ctx["projected_area"])


def _score_tile_complexity(ctx, eps=1e-6):
    tile_idx = ctx["tile_idx"]
    tile_count = ctx["tile_count"]
    color = ctx["color"]
    depth = ctx["depth"]
    scale_mean = ctx["scale_mean"]

    counts = ctx["tile_counts"].clamp_min(1.0)
    sum_color = torch.zeros((tile_count, 3), device=color.device, dtype=color.dtype)
    sum_color.index_add_(0, tile_idx, color)
    mean_color = sum_color[tile_idx] / counts[tile_idx].unsqueeze(-1)
    color_dev = torch.linalg.vector_norm(color - mean_color, dim=-1) / 1.7320508075688772

    depth_var, depth_mean, _ = _tile_var(depth, tile_idx, tile_count, fill_value=depth.mean())
    depth_scale = depth.mean().clamp_min(eps)
    depth_dev = torch.abs(depth - depth_mean[tile_idx]) / depth_scale
    depth_var = torch.sqrt(depth_var[tile_idx]) / depth_scale

    scale_var, scale_tile_mean, _ = _tile_var(scale_mean, tile_idx, tile_count, fill_value=scale_mean.mean())
    scale_scale = scale_mean.mean().clamp_min(eps)
    shape_dev = 0.5 * torch.abs(scale_mean - scale_tile_mean[tile_idx]) / scale_scale
    shape_dev = shape_dev + 0.5 * torch.sqrt(scale_var[tile_idx]) / scale_scale

    area = ctx["projected_area"].clamp_min(0.0)
    tau = area.mean().clamp_min(eps)
    saturated_area = 1.0 - torch.exp(-area / tau)
    large_area_penalty = torch.rsqrt(1.0 + area / (tau * 4.0).clamp_min(eps))
    score = ctx["opacity"] * saturated_area * large_area_penalty * (
        0.40 * color_dev + 0.30 * depth_dev + 0.20 * depth_var + 0.10 * shape_dev
    )
    return _normalize_positive(score)


def _score_image_edge(ctx):
    edge = _sample_map(ctx["features"]["edge"], ctx["pixel_x"], ctx["pixel_y"])
    area = torch.log1p(_normalize_positive(ctx["projected_area"]))
    return _normalize_positive(edge * (0.5 + area))


def _score_harris_corner(ctx):
    corner = _sample_map(ctx["features"]["harris"], ctx["pixel_x"], ctx["pixel_y"])
    area = torch.log1p(_normalize_positive(ctx["projected_area"]))
    return _normalize_positive(corner * (0.5 + area))


def _score_laplacian_blob(ctx):
    blob = _sample_map(ctx["features"]["blob"], ctx["pixel_x"], ctx["pixel_y"])
    area = torch.log1p(_normalize_positive(ctx["projected_area"]))
    return _normalize_positive(blob * (0.5 + area))


def _score_entropy(ctx):
    entropy = _sample_map(ctx["features"]["entropy"], ctx["pixel_x"], ctx["pixel_y"])
    area = torch.log1p(_normalize_positive(ctx["projected_area"]))
    density = ctx["tile_counts"][ctx["tile_idx"]].clamp_min(1.0)
    density_penalty = torch.rsqrt(density)
    return _normalize_positive(entropy * (0.5 + area) * (0.5 + density_penalty))


def _score_depth_boundary(ctx, eps=1e-6):
    tile_idx = ctx["tile_idx"]
    tile_count = ctx["tile_count"]
    depth = ctx["depth"]
    depth_scale = depth.mean().clamp_min(eps)
    depth_var, depth_mean, counts = _tile_var(depth, tile_idx, tile_count, fill_value=depth.mean())
    depth_dev = torch.abs(depth - depth_mean[tile_idx]) / depth_scale
    depth_std = torch.sqrt(depth_var[tile_idx]) / depth_scale

    depth_grid = depth_mean.view(ctx["tile_h"], ctx["tile_w"])
    grid_edge, _, _ = _sobel_response(depth_grid)
    grid_edge = grid_edge.reshape(-1)[tile_idx] / depth_scale
    density_weight = torch.rsqrt(counts[tile_idx].clamp_min(1.0))
    return _normalize_positive((0.45 * depth_dev + 0.35 * depth_std + 0.20 * grid_edge) * (0.5 + density_weight))


def _score_curvature(ctx, eps=1e-6):
    tile_idx = ctx["tile_idx"]
    tile_count = ctx["tile_count"]
    depth = ctx["depth"]
    scale_mean = ctx["scale_mean"]
    scale_var, _, _ = _tile_var(scale_mean, tile_idx, tile_count, fill_value=scale_mean.mean())
    depth_var, _, _ = _tile_var(depth, tile_idx, tile_count, fill_value=depth.mean())
    scale_component = torch.sqrt(scale_var[tile_idx]) / scale_mean.mean().clamp_min(eps)
    depth_component = torch.sqrt(depth_var[tile_idx]) / depth.mean().clamp_min(eps)
    score = 0.45 * ctx["scale_anisotropy"] + 0.35 * scale_component + 0.20 * depth_component
    return _normalize_positive(score)


def _score_density_penalized_edge(ctx):
    edge = _sample_map(ctx["features"]["edge"], ctx["pixel_x"], ctx["pixel_y"])
    tile_density = ctx["tile_counts"][ctx["tile_idx"]].clamp_min(1.0)
    density_penalty = torch.rsqrt(tile_density)
    return _normalize_positive(edge * density_penalty)


def _score_coarse_residual(ctx):
    residual_map = ctx["residual_map"]
    if residual_map is None:
        return torch.zeros_like(ctx["depth"])
    residual = _sample_map(_normalize_positive(residual_map), ctx["pixel_x"], ctx["pixel_y"])
    return _normalize_positive(residual)


def _score_hybrid_structure(ctx):
    return _normalize_positive(
        0.25 * _score_image_edge(ctx)
        + 0.15 * _score_harris_corner(ctx)
        + 0.10 * _score_laplacian_blob(ctx)
        + 0.10 * _score_entropy(ctx)
        + 0.25 * _score_depth_boundary(ctx)
        + 0.15 * _score_curvature(ctx)
        + 0.05 * _score_coarse_residual(ctx)
    )


def _projection_proxy(ctx, eps=1e-6):
    cached = ctx.get("projection_proxy")
    if cached is not None:
        return cached

    tile_idx = ctx["tile_idx"]
    tile_count = ctx["tile_count"]
    tile_h = ctx["tile_h"]
    tile_w = ctx["tile_w"]
    raw_counts = ctx["tile_counts"]
    counts = raw_counts.clamp_min(1.0)
    depth = ctx["depth"]
    color = ctx["color"]
    scale_mean = ctx["scale_mean"]
    projected_area = ctx["projected_area"]
    device = depth.device
    dtype = depth.dtype

    depth_var, depth_mean, _ = _tile_var(depth, tile_idx, tile_count, fill_value=depth.mean())
    scale_var, scale_mean_tile, _ = _tile_var(scale_mean, tile_idx, tile_count, fill_value=scale_mean.mean())
    area_var, area_mean, _ = _tile_var(projected_area, tile_idx, tile_count, fill_value=projected_area.mean())

    color_sum = torch.zeros((tile_count, 3), device=device, dtype=dtype)
    color_sum.index_add_(0, tile_idx, color)
    color_mean = color_sum / counts.unsqueeze(-1)
    empty_tiles = raw_counts <= 0.0
    if empty_tiles.any():
        color_mean = torch.where(empty_tiles.unsqueeze(-1), color.mean(dim=0, keepdim=True), color_mean)
    color_sq_sum = torch.zeros((tile_count, 3), device=device, dtype=dtype)
    color_sq_sum.index_add_(0, tile_idx, color * color)
    color_var = torch.clamp_min(color_sq_sum / counts.unsqueeze(-1) - color_mean * color_mean, 0.0)
    color_std = torch.sqrt(color_var.sum(dim=-1).clamp_min(0.0)) / 1.7320508075688772

    depth_grid = (depth_mean / depth.mean().clamp_min(eps)).view(tile_h, tile_w)
    color_grid = (0.299 * color_mean[:, 0] + 0.587 * color_mean[:, 1] + 0.114 * color_mean[:, 2]).view(tile_h, tile_w)
    density_grid = _normalize_positive(raw_counts).view(tile_h, tile_w)
    area_grid = _normalize_positive(area_mean).view(tile_h, tile_w)
    depth_std_grid = _normalize_positive(torch.sqrt(depth_var) / depth.mean().clamp_min(eps)).view(tile_h, tile_w)
    color_std_grid = _normalize_positive(color_std).view(tile_h, tile_w)
    scale_std_grid = _normalize_positive(torch.sqrt(scale_var) / scale_mean.mean().clamp_min(eps)).view(tile_h, tile_w)
    area_std_grid = _normalize_positive(torch.sqrt(area_var) / projected_area.mean().clamp_min(eps)).view(tile_h, tile_w)

    depth_edge = _normalize_positive(_sobel_response(depth_grid)[0])
    color_edge = _normalize_positive(_sobel_response(color_grid)[0])
    density_edge = _normalize_positive(_sobel_response(density_grid)[0])
    area_edge = _normalize_positive(_sobel_response(area_grid)[0])
    edge_tile = _normalize_positive(0.40 * depth_edge + 0.25 * color_edge + 0.20 * density_edge + 0.15 * area_edge)

    structural_map = _normalize_positive(
        0.35 * _normalize_positive(depth_grid)
        + 0.25 * density_grid
        + 0.20 * area_grid
        + 0.20 * depth_std_grid
    )
    harris_tile = _harris_response(structural_map)
    diversity_grid = 0.35 * depth_std_grid + 0.25 * color_std_grid + 0.20 * scale_std_grid + 0.20 * area_std_grid
    entropy_tile = _normalize_positive(F.avg_pool2d(diversity_grid.view(1, 1, tile_h, tile_w), 3, stride=1, padding=1)[0, 0])
    laplacian_tile = _normalize_positive(
        0.35 * _laplacian_abs(density_grid)
        + 0.30 * _laplacian_abs(area_grid)
        + 0.20 * _laplacian_abs(depth_std_grid)
        + 0.15 * _laplacian_abs(color_std_grid)
    )

    depth_dev = torch.abs(depth - depth_mean[tile_idx]) / depth.mean().clamp_min(eps)
    color_dev = torch.linalg.vector_norm(color - color_mean[tile_idx], dim=-1) / 1.7320508075688772
    scale_dev = torch.abs(scale_mean - scale_mean_tile[tile_idx]) / scale_mean.mean().clamp_min(eps)
    area_dev = torch.abs(projected_area - area_mean[tile_idx]) / projected_area.mean().clamp_min(eps)
    anchor_contrast = _normalize_positive(
        0.35 * depth_dev
        + 0.25 * color_dev
        + 0.20 * scale_dev
        + 0.20 * area_dev
        + 0.20 * _normalize_positive(ctx["scale_anisotropy"])
    )

    proxy = {
        "edge": edge_tile.reshape(-1)[tile_idx],
        "harris": harris_tile.reshape(-1)[tile_idx],
        "entropy": entropy_tile.reshape(-1)[tile_idx],
        "blob": laplacian_tile.reshape(-1)[tile_idx],
        "density_penalty": torch.rsqrt(counts[tile_idx]),
        "anchor_contrast": anchor_contrast,
        "area": torch.log1p(_normalize_positive(projected_area)),
    }
    ctx["projection_proxy"] = proxy
    return proxy


def _score_projection_image_edge(ctx):
    proxy = _projection_proxy(ctx)
    return _normalize_positive(proxy["edge"] * (0.5 + proxy["area"]) * (0.75 + 0.50 * proxy["anchor_contrast"]))


def _score_projection_density_penalized_edge(ctx):
    proxy = _projection_proxy(ctx)
    return _normalize_positive(proxy["edge"] * (0.5 + proxy["density_penalty"]) * (0.75 + 0.50 * proxy["anchor_contrast"]))


def _score_projection_entropy(ctx):
    proxy = _projection_proxy(ctx)
    return _normalize_positive(
        proxy["entropy"] * (0.5 + proxy["area"]) * (0.5 + proxy["density_penalty"]) * (0.75 + proxy["anchor_contrast"])
    )


def _score_projection_harris_corner(ctx):
    proxy = _projection_proxy(ctx)
    return _normalize_positive(proxy["harris"] * (0.5 + proxy["area"]) * (0.75 + proxy["anchor_contrast"]))


def _score_projection_laplacian_blob(ctx):
    proxy = _projection_proxy(ctx)
    return _normalize_positive(proxy["blob"] * (0.5 + proxy["area"]) * (0.75 + proxy["anchor_contrast"]))


def _score_projection_hybrid_structure(ctx):
    return _normalize_positive(
        0.30 * _score_projection_image_edge(ctx)
        + 0.20 * _score_projection_density_penalized_edge(ctx)
        + 0.15 * _score_projection_entropy(ctx)
        + 0.15 * _score_projection_harris_corner(ctx)
        + 0.10 * _score_projection_laplacian_blob(ctx)
        + 0.10 * _score_curvature(ctx)
    )


def _projection_neighbor_proxy(ctx, k=24, radius=24.0, eps=1e-6):
    cache = ctx.setdefault("neighbor_proxy", {})
    cache_key = (int(k), float(radius))
    if cache_key in cache:
        return cache[cache_key]

    depth = ctx["depth"].float()
    visible_count = int(depth.shape[0])
    device = depth.device
    dtype = torch.float32
    eps = max(float(eps), 1e-12)
    k = max(1, int(k))
    radius = max(1.0, float(radius))
    zeros = torch.zeros_like(depth, dtype=dtype)
    if visible_count <= 1:
        proxy = {
            "valid_ratio": zeros,
            "density_penalty": torch.ones_like(zeros),
            "support": zeros,
            "plane_residual": zeros,
            "plane_curvature": zeros,
            "depth_gap": zeros,
            "depth_range": zeros,
            "depth_contrast": zeros,
            "color_contrast": zeros,
            "scale_contrast": zeros,
            "area_contrast": zeros,
            "anisotropy": zeros,
            "area_weight": torch.ones_like(zeros),
        }
        cache[cache_key] = proxy
        return proxy

    points = torch.stack([ctx["pixel_x"].to(device=device, dtype=dtype), ctx["pixel_y"].to(device=device, dtype=dtype)], dim=-1)
    xyz = ctx["xyz"].to(device=device, dtype=dtype)
    color = ctx["color"].to(device=device, dtype=dtype)
    scale_mean = ctx["scale_mean"].to(device=device, dtype=dtype)
    projected_area = ctx["projected_area"].to(device=device, dtype=dtype)
    anisotropy = ctx["scale_anisotropy"].to(device=device, dtype=dtype)
    opacity = ctx["opacity"].to(device=device, dtype=dtype)

    valid_count = torch.zeros(visible_count, device=device, dtype=dtype)
    neighbor_radius = torch.full((visible_count,), radius, device=device, dtype=dtype)
    plane_residual = torch.zeros(visible_count, device=device, dtype=dtype)
    plane_curvature = torch.zeros_like(plane_residual)
    depth_gap = torch.zeros_like(plane_residual)
    depth_range = torch.zeros_like(plane_residual)
    depth_contrast = torch.zeros_like(plane_residual)
    color_contrast = torch.zeros_like(plane_residual)
    scale_contrast = torch.zeros_like(plane_residual)
    area_contrast = torch.zeros_like(plane_residual)

    image_width = max(1, int(ctx.get("image_width", int(points[:, 0].max().item()) + 1)))
    image_height = max(1, int(ctx.get("image_height", int(points[:, 1].max().item()) + 1)))
    cell_size = radius
    cell_w = max(1, int(math.ceil(float(image_width) / cell_size)))
    cell_h = max(1, int(math.ceil(float(image_height) / cell_size)))
    cell_x = torch.floor(points[:, 0] / cell_size).long().clamp_(0, cell_w - 1)
    cell_y = torch.floor(points[:, 1] / cell_size).long().clamp_(0, cell_h - 1)
    cell_ids = cell_y * cell_w + cell_x

    order = torch.argsort(cell_ids)
    sorted_ids = cell_ids[order]
    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    cell_to_indices = {}
    start = 0
    for cell_id, count in zip(unique_ids.tolist(), counts.tolist()):
        end = start + int(count)
        cell_to_indices[int(cell_id)] = order[start:end]
        start = end

    depth_mean = depth.mean().clamp_min(eps)
    scale_mean_global = scale_mean.mean().clamp_min(eps)
    area_mean_global = projected_area.mean().clamp_min(eps)

    for cell_id, anchor_idx in cell_to_indices.items():
        cx = cell_id % cell_w
        cy = cell_id // cell_w
        candidate_parts = []
        for dy in (-1, 0, 1):
            ny = cy + dy
            if ny < 0 or ny >= cell_h:
                continue
            for dx in (-1, 0, 1):
                nx = cx + dx
                if nx < 0 or nx >= cell_w:
                    continue
                part = cell_to_indices.get(int(ny * cell_w + nx))
                if part is not None:
                    candidate_parts.append(part)
        if not candidate_parts:
            continue
        candidate_idx = torch.cat(candidate_parts, dim=0)
        if candidate_idx.numel() <= 1:
            continue

        dist = torch.cdist(points[anchor_idx], points[candidate_idx])
        self_mask = candidate_idx.unsqueeze(0) == anchor_idx.unsqueeze(1)
        dist = dist.masked_fill(self_mask, float("inf"))
        dist = dist.masked_fill(dist > radius, float("inf"))
        take = min(k, int(candidate_idx.numel()) - 1)
        if take <= 0:
            continue
        knn_dist, knn_pos = torch.topk(dist, k=take, largest=False)
        valid = torch.isfinite(knn_dist)
        neighbor_idx = candidate_idx[knn_pos]
        valid_f = valid.to(dtype=dtype)
        local_valid_count = valid_f.sum(dim=1)
        has_neighbor = local_valid_count > 0.0
        valid_count[anchor_idx] = local_valid_count

        own_depth = depth[anchor_idx].unsqueeze(1)
        neighbor_depth_raw = depth[neighbor_idx]
        neighbor_depth = torch.where(valid, neighbor_depth_raw, own_depth)
        median_depth = neighbor_depth.median(dim=1).values
        depth_contrast[anchor_idx] = torch.abs(depth[anchor_idx] - median_depth) / depth_mean

        depth_min_raw = torch.where(valid, neighbor_depth_raw, neighbor_depth_raw.new_full((), float("inf"))).min(dim=1).values
        depth_max_raw = torch.where(valid, neighbor_depth_raw, neighbor_depth_raw.new_full((), float("-inf"))).max(dim=1).values
        depth_min = torch.where(has_neighbor, depth_min_raw, depth[anchor_idx])
        depth_max = torch.where(has_neighbor, depth_max_raw, depth[anchor_idx])
        depth_range_v = (depth_max - depth_min).clamp_min(0.0) / depth_mean
        if take > 1:
            sorted_depth = torch.sort(neighbor_depth, dim=1).values
            gap = torch.diff(sorted_depth, dim=1).clamp_min(0.0).max(dim=1).values / depth_mean
        else:
            gap = torch.zeros_like(depth_range_v)
        depth_range[anchor_idx] = depth_range_v
        depth_gap[anchor_idx] = torch.where(has_neighbor, gap, torch.zeros_like(gap))

        own_color = color[anchor_idx].unsqueeze(1).expand(-1, take, -1)
        neighbor_color = torch.where(valid.unsqueeze(-1), color[neighbor_idx], own_color)
        median_color = neighbor_color.median(dim=1).values
        color_contrast[anchor_idx] = torch.linalg.vector_norm(color[anchor_idx] - median_color, dim=-1) / 1.7320508075688772

        own_scale = scale_mean[anchor_idx].unsqueeze(1)
        neighbor_scale = torch.where(valid, scale_mean[neighbor_idx], own_scale)
        median_scale = neighbor_scale.median(dim=1).values
        scale_contrast[anchor_idx] = torch.abs(scale_mean[anchor_idx] - median_scale) / scale_mean_global

        own_area = projected_area[anchor_idx].unsqueeze(1)
        neighbor_area = torch.where(valid, projected_area[neighbor_idx], own_area)
        median_area = neighbor_area.median(dim=1).values
        area_contrast[anchor_idx] = torch.abs(projected_area[anchor_idx] - median_area) / area_mean_global

        neighbor_xyz = xyz[neighbor_idx]
        own_xyz = xyz[anchor_idx].unsqueeze(1).expand(-1, take, -1)
        neighbor_xyz = torch.where(valid.unsqueeze(-1), neighbor_xyz, own_xyz)
        weights = valid_f.unsqueeze(-1)
        safe_count = local_valid_count.clamp_min(1.0).unsqueeze(-1)
        neighbor_mean = (neighbor_xyz * weights).sum(dim=1) / safe_count
        neighbor_mean = torch.where(has_neighbor.unsqueeze(-1), neighbor_mean, xyz[anchor_idx])
        centered = (neighbor_xyz - neighbor_mean.unsqueeze(1)) * weights
        cov = torch.matmul(centered.transpose(1, 2), centered) / safe_count.unsqueeze(-1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp_min(eigvals, 0.0)
        normal = eigvecs[:, :, 0]
        local_scale = torch.sqrt(eigvals[:, -1]).clamp_min(eps)
        offset = xyz[anchor_idx] - neighbor_mean
        residual = torch.abs((offset * normal).sum(dim=-1)) / local_scale
        curvature = eigvals[:, 0] / eigvals.sum(dim=-1).clamp_min(eps)
        plane_residual[anchor_idx] = torch.where(has_neighbor, residual, torch.zeros_like(residual))
        plane_curvature[anchor_idx] = torch.where(has_neighbor, curvature, torch.zeros_like(curvature))

        finite_dist = torch.where(valid, knn_dist, knn_dist.new_full((), radius))
        neighbor_radius[anchor_idx] = finite_dist.max(dim=1).values.clamp_min(eps)

    valid_ratio = (valid_count / float(k)).clamp(0.0, 1.0)
    density_penalty = torch.sqrt(neighbor_radius / neighbor_radius.mean().clamp_min(eps)).clamp(0.35, 1.50)
    support = torch.sqrt(valid_ratio) * torch.clamp(opacity, 0.05, 1.0)
    area_weight = 0.35 + 0.65 * torch.log1p(_normalize_positive(projected_area))
    proxy = {
        "valid_ratio": valid_ratio,
        "density_penalty": density_penalty,
        "support": support,
        "plane_residual": _normalize_positive(plane_residual),
        "plane_curvature": _normalize_positive(plane_curvature),
        "depth_gap": _normalize_positive(depth_gap),
        "depth_range": _normalize_positive(depth_range),
        "depth_contrast": _normalize_positive(depth_contrast),
        "color_contrast": _normalize_positive(color_contrast),
        "scale_contrast": _normalize_positive(scale_contrast),
        "area_contrast": _normalize_positive(area_contrast),
        "anisotropy": _normalize_positive(anisotropy),
        "area_weight": area_weight,
    }
    cache[cache_key] = proxy
    return proxy


def _score_projection_knn_laplacian(ctx, k=16, radius=16.0, gamma=1.5, eps=1e-6):
    proxy = _projection_neighbor_proxy(ctx, k=k, radius=radius, eps=eps)
    local_contrast = (
        0.45 * proxy["depth_contrast"]
        + 0.15 * proxy["color_contrast"]
        + 0.25 * proxy["scale_contrast"]
        + 0.15 * proxy["area_contrast"]
        + 0.10 * proxy["anisotropy"]
    )
    score = local_contrast * proxy["density_penalty"] * proxy["area_weight"] * proxy["support"]
    return _normalize_positive(torch.pow(torch.clamp_min(score, 0.0), float(gamma)))


def _score_projection_plane_residual(ctx):
    proxy = _projection_neighbor_proxy(ctx, k=24, radius=24.0)
    score = proxy["plane_residual"] * proxy["support"] * proxy["density_penalty"] * proxy["area_weight"]
    return _normalize_positive(score)


def _score_projection_multiscale_plane_residual(ctx):
    small = _projection_neighbor_proxy(ctx, k=16, radius=16.0)
    large = _projection_neighbor_proxy(ctx, k=32, radius=40.0)
    agreement = torch.sqrt((small["plane_residual"] * large["plane_residual"]).clamp_min(0.0))
    score = agreement * small["support"] * small["density_penalty"] * small["area_weight"]
    return _normalize_positive(score)


def _score_projection_depth_layer_boundary(ctx):
    proxy = _projection_neighbor_proxy(ctx, k=24, radius=24.0)
    layer = 0.70 * proxy["depth_gap"] + 0.30 * proxy["depth_contrast"]
    score = layer * proxy["support"] * proxy["density_penalty"] * (0.5 + 0.5 * proxy["plane_residual"])
    return _normalize_positive(score)


def _score_projection_occlusion_support(ctx):
    proxy = _projection_neighbor_proxy(ctx, k=32, radius=32.0)
    boundary = 0.65 * proxy["depth_gap"] + 0.20 * proxy["plane_residual"] + 0.15 * proxy["area_contrast"]
    support_gate = proxy["support"] * proxy["valid_ratio"].clamp(0.0, 1.0)
    score = boundary * support_gate * proxy["density_penalty"]
    return _normalize_positive(score)


def _score_projection_support_gated_complexity(ctx):
    proxy = _projection_neighbor_proxy(ctx, k=24, radius=24.0)
    need = (
        0.40 * proxy["plane_residual"]
        + 0.25 * proxy["depth_gap"]
        + 0.20 * proxy["scale_contrast"]
        + 0.10 * proxy["anisotropy"]
        + 0.05 * proxy["color_contrast"]
    )
    score = need * proxy["support"] * proxy["density_penalty"] * proxy["area_weight"]
    return _normalize_positive(score)


def _score_projection_density_suppressed_curvature(ctx):
    proxy = _projection_neighbor_proxy(ctx, k=24, radius=24.0)
    curvature = (
        0.45 * proxy["plane_curvature"]
        + 0.25 * proxy["scale_contrast"]
        + 0.20 * proxy["plane_residual"]
        + 0.10 * proxy["area_contrast"]
    )
    score = curvature * proxy["support"] * proxy["density_penalty"]
    return _normalize_positive(score)


def _score_projection_gain_proxy(ctx):
    proxy = _projection_neighbor_proxy(ctx, k=24, radius=24.0)
    need = 0.50 * proxy["plane_residual"] + 0.30 * proxy["depth_gap"] + 0.20 * proxy["scale_contrast"]
    gain_support = proxy["support"] * proxy["density_penalty"] * proxy["area_weight"]
    score = torch.pow((need * gain_support).clamp_min(0.0), 1.35)
    return _normalize_positive(score)


def _score_projection_structure_hybrid(ctx):
    return _normalize_positive(
        0.25 * _score_projection_gain_proxy(ctx)
        + 0.20 * _score_projection_plane_residual(ctx)
        + 0.20 * _score_projection_depth_layer_boundary(ctx)
        + 0.15 * _score_projection_occlusion_support(ctx)
        + 0.10 * _score_projection_density_suppressed_curvature(ctx)
        + 0.10 * _score_projection_multiscale_plane_residual(ctx)
    )


def _projection_large_uniform_surface_confidence(ctx, eps=1e-6):
    cache = ctx.setdefault("surface_gate", {})
    if "large_uniform_surface_confidence" in cache:
        return cache["large_uniform_surface_confidence"]

    tile_idx = ctx["tile_idx"]
    tile_count = ctx["tile_count"]
    counts = ctx["tile_counts"].clamp_min(1.0)
    depth = ctx["depth"].float()
    scale_mean = ctx["scale_mean"].float()
    projected_area = ctx["projected_area"].float()
    color = ctx["color"].float()
    device = depth.device
    dtype = depth.dtype
    eps = max(float(eps), 1e-12)

    depth_var, _, _ = _tile_var(depth, tile_idx, tile_count, fill_value=depth.mean())
    scale_var, _, _ = _tile_var(scale_mean, tile_idx, tile_count, fill_value=scale_mean.mean())
    area_var, _, _ = _tile_var(projected_area, tile_idx, tile_count, fill_value=projected_area.mean())

    sum_color = torch.zeros((tile_count, 3), device=device, dtype=dtype)
    sum_color2 = torch.zeros((tile_count, 3), device=device, dtype=dtype)
    sum_color.index_add_(0, tile_idx, color)
    sum_color2.index_add_(0, tile_idx, color * color)
    mean_color = sum_color / counts.unsqueeze(-1)
    color_var = (sum_color2 / counts.unsqueeze(-1) - mean_color * mean_color).clamp_min(0.0)

    depth_std = torch.sqrt(depth_var[tile_idx].clamp_min(0.0)) / depth.mean().clamp_min(eps)
    scale_std = torch.sqrt(scale_var[tile_idx].clamp_min(0.0)) / scale_mean.mean().clamp_min(eps)
    area_std = torch.sqrt(area_var[tile_idx].clamp_min(0.0)) / projected_area.mean().clamp_min(eps)
    color_std = torch.sqrt(color_var[tile_idx].mean(dim=-1).clamp_min(0.0))

    local_variation = _normalize_positive(
        0.30 * depth_std
        + 0.25 * scale_std
        + 0.25 * area_std
        + 0.20 * color_std
    )
    high_population = _normalize_positive(torch.log1p(counts))[tile_idx]
    low_variation = (1.0 - local_variation).clamp(0.0, 1.0)
    confidence = (high_population * low_variation).clamp(0.0, 1.0)
    cache["large_uniform_surface_confidence"] = confidence
    return confidence


def _projection_surface_gates(ctx, eps=1e-6):
    cache = ctx.setdefault("surface_gate", {})
    if "base_gates" in cache:
        return cache["base_gates"]

    proxy = _projection_neighbor_proxy(ctx, k=24, radius=24.0, eps=eps)
    valid_ratio = proxy["valid_ratio"].clamp(0.0, 1.0)
    support = proxy["support"].clamp(0.0, 1.0)
    large_uniform = _projection_large_uniform_surface_confidence(ctx, eps=eps)

    attribute_transition = _normalize_positive(
        0.35 * proxy["depth_gap"]
        + 0.20 * proxy["scale_contrast"]
        + 0.20 * proxy["area_contrast"]
        + 0.15 * proxy["color_contrast"]
        + 0.10 * proxy["anisotropy"]
    )
    boundary_structure = _normalize_positive(
        0.35 * proxy["depth_gap"]
        + 0.25 * proxy["plane_residual"]
        + 0.15 * proxy["plane_curvature"]
        + 0.15 * proxy["scale_contrast"]
        + 0.10 * proxy["area_contrast"]
    )

    planar = (1.0 - proxy["plane_residual"]).clamp(0.0, 1.0) * (1.0 - proxy["plane_curvature"]).clamp(0.0, 1.0)
    low_transition = (1.0 - attribute_transition).clamp(0.0, 1.0)
    dense_support = (valid_ratio * torch.clamp(2.0 * support, 0.0, 1.0)).clamp(0.0, 1.0)
    flat_surface = (planar * low_transition * dense_support).clamp(0.0, 1.0)

    flat_suppression = torch.pow((1.0 - flat_surface).clamp(0.0, 1.0), 3.0)
    large_suppression = torch.pow((1.0 - large_uniform).clamp(0.0, 1.0), 2.5)
    sparse_gate = torch.clamp(valid_ratio / 0.25, 0.0, 1.0)
    mid_support = torch.exp(-torch.square((valid_ratio - 0.55) / 0.35))
    usable_support = (sparse_gate * (0.35 + 0.65 * mid_support)).clamp(0.0, 1.0)

    gates = {
        "proxy": proxy,
        "attribute_transition": attribute_transition,
        "boundary_structure": boundary_structure,
        "flat_surface": flat_surface,
        "large_uniform": large_uniform,
        "flat_suppression": flat_suppression,
        "large_suppression": large_suppression,
        "usable_support": usable_support,
    }
    cache["base_gates"] = gates
    return gates


def _score_projection_flat_surface_suppressed_boundary(ctx):
    gates = _projection_surface_gates(ctx)
    proxy = gates["proxy"]
    boundary = _normalize_positive(
        0.45 * proxy["depth_gap"]
        + 0.20 * proxy["depth_contrast"]
        + 0.15 * proxy["scale_contrast"]
        + 0.10 * proxy["area_contrast"]
        + 0.10 * proxy["plane_residual"]
    )
    score = boundary * gates["flat_suppression"] * (0.25 + 0.75 * gates["usable_support"])
    return _normalize_positive(score)


def _score_projection_component_penalized_structure(ctx):
    gates = _projection_surface_gates(ctx)
    proxy = gates["proxy"]
    structure = _normalize_positive(
        0.30 * proxy["plane_residual"]
        + 0.25 * proxy["depth_gap"]
        + 0.20 * proxy["scale_contrast"]
        + 0.15 * proxy["anisotropy"]
        + 0.10 * proxy["color_contrast"]
    )
    score = structure * gates["flat_suppression"] * gates["large_suppression"] * gates["usable_support"]
    return _normalize_positive(score)


def _score_projection_object_boundary_gain(ctx):
    gates = _projection_surface_gates(ctx)
    proxy = gates["proxy"]
    depth_boundary = _normalize_positive(0.70 * proxy["depth_gap"] + 0.30 * proxy["depth_contrast"])
    attribute_boundary = gates["attribute_transition"]
    objectness = _normalize_positive(
        0.45 * depth_boundary
        + 0.25 * attribute_boundary
        + 0.20 * proxy["scale_contrast"]
        + 0.10 * proxy["area_contrast"]
    )
    score = objectness * gates["flat_suppression"] * torch.sqrt(gates["large_suppression"].clamp_min(0.0))
    score = score * (0.20 + 0.80 * gates["usable_support"])
    return _normalize_positive(torch.pow(score.clamp_min(0.0), 1.25))


def _score_projection_learnable_detail_gain(ctx):
    gates = _projection_surface_gates(ctx)
    proxy = gates["proxy"]
    learnable_need_proxy = _normalize_positive(
        0.30 * gates["boundary_structure"]
        + 0.25 * gates["attribute_transition"]
        + 0.20 * proxy["plane_curvature"]
        + 0.15 * proxy["anisotropy"]
        + 0.10 * proxy["area_contrast"]
    )
    gain_gate = gates["usable_support"] * gates["large_suppression"] * torch.pow(gates["flat_suppression"], 1.25)
    score = torch.pow(learnable_need_proxy.clamp_min(0.0), 1.4) * gain_gate
    return _normalize_positive(score)


def _score_projected_area_flat_suppressed(ctx):
    gates = _projection_surface_gates(ctx)
    area = torch.log1p(_normalize_positive(ctx["projected_area"]))
    transition_gate = 0.15 + 0.85 * gates["attribute_transition"]
    score = area * torch.pow(gates["flat_suppression"], 1.5) * torch.pow(gates["large_suppression"], 1.25) * transition_gate
    return _normalize_positive(score)


def _projection_inverse_bias_terms(ctx, eps=1e-6):
    cache = ctx.setdefault("inverse_bias", {})
    if "terms" in cache:
        return cache["terms"]

    gates = _projection_surface_gates(ctx, eps=eps)
    proxy = gates["proxy"]
    color = ctx["color"].float().clamp(0.0, 1.0)
    opacity = ctx["opacity"].float().clamp(0.0, 1.0)
    eps = max(float(eps), 1e-12)

    area_score = _normalize_positive(torch.log1p(_normalize_positive(ctx["projected_area"])))
    density_score = proxy["valid_ratio"].clamp(0.0, 1.0)
    support_score = _normalize_positive(proxy["support"])
    flat_score = gates["flat_surface"].clamp(0.0, 1.0)
    large_uniform = gates["large_uniform"].clamp(0.0, 1.0)
    low_transition = (1.0 - gates["attribute_transition"]).clamp(0.0, 1.0)

    luminance = (0.2126 * color[:, 0] + 0.7152 * color[:, 1] + 0.0722 * color[:, 2]).clamp(0.0, 1.0)
    dark_score = torch.pow((1.0 - luminance).clamp(0.0, 1.0), 1.25)
    chroma = torch.linalg.vector_norm(color - color.mean(dim=-1, keepdim=True), dim=-1) / 0.816496580927726
    low_chroma = (1.0 - _normalize_positive(chroma)).clamp(0.0, 1.0)

    common_bias = (
        0.24 * area_score
        + 0.20 * density_score
        + 0.16 * support_score
        + 0.24 * flat_score
        + 0.16 * large_uniform
    ).clamp(0.0, 1.0)

    road_water_shadow_bias = (
        flat_score
        * (0.35 + 0.65 * large_uniform)
        * (
            0.34 * low_transition
            + 0.24 * low_chroma
            + 0.20 * dark_score
            + 0.22 * area_score
        ).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)

    visibility_gate = torch.clamp(density_score / 0.08, 0.0, 1.0) * torch.clamp(opacity / 0.05, 0.0, 1.0)
    mid_density = torch.exp(-torch.square((density_score - 0.38) / 0.34))
    not_overdense = (1.0 - torch.clamp((density_score - 0.78) / 0.22, 0.0, 1.0)).clamp(0.0, 1.0)
    recoverable_gate = (visibility_gate * (0.30 + 0.70 * mid_density) * (0.35 + 0.65 * not_overdense)).clamp(0.0, 1.0)
    non_uniform_gate = (0.20 + 0.80 * gates["attribute_transition"]).clamp(0.0, 1.0)
    inverse_area = torch.pow((1.0 - area_score).clamp(0.0, 1.0), 2.0)

    terms = {
        "gates": gates,
        "proxy": proxy,
        "area_score": area_score,
        "density_score": density_score,
        "support_score": support_score,
        "flat_score": flat_score,
        "large_uniform": large_uniform,
        "low_transition": low_transition,
        "dark_score": dark_score,
        "low_chroma": low_chroma,
        "common_bias": common_bias,
        "road_water_shadow_bias": road_water_shadow_bias,
        "visibility_gate": visibility_gate,
        "mid_density": mid_density,
        "not_overdense": not_overdense,
        "recoverable_gate": recoverable_gate,
        "non_uniform_gate": non_uniform_gate,
        "inverse_area": inverse_area,
    }
    cache["terms"] = terms
    return terms


def _score_projection_inverse_common_bias(ctx):
    terms = _projection_inverse_bias_terms(ctx)
    inverse_bias = torch.pow((1.0 - terms["common_bias"]).clamp(0.0, 1.0), 3.0)
    score = inverse_bias * terms["recoverable_gate"] * terms["non_uniform_gate"]
    return _normalize_positive(score)


def _score_projection_hard_reject_flat_dense(ctx):
    terms = _projection_inverse_bias_terms(ctx)
    flat = terms["flat_score"]
    dense = terms["density_score"]
    area = terms["area_score"]
    large = terms["large_uniform"]
    low_transition = terms["low_transition"]

    flat_thr = torch.quantile(flat, 0.70).clamp_min(0.25)
    dense_thr = torch.quantile(dense, 0.70).clamp_min(0.25)
    area_thr = torch.quantile(area, 0.70).clamp_min(0.20)
    large_thr = torch.quantile(large, 0.70).clamp_min(0.20)
    reject = ((flat > flat_thr) & (dense > dense_thr) & (area > area_thr)) | ((large > large_thr) & (flat > flat_thr) & (low_transition > 0.45))
    keep_gate = torch.where(reject, torch.full_like(flat, 0.02), torch.ones_like(flat))

    base = torch.pow((1.0 - terms["common_bias"]).clamp(0.0, 1.0), 2.0)
    structure_gate = 0.25 + 0.75 * terms["non_uniform_gate"]
    score = base * keep_gate * terms["recoverable_gate"] * structure_gate
    return _normalize_positive(score)


def _score_projection_mid_support_anti_flat(ctx):
    terms = _projection_inverse_bias_terms(ctx)
    anti_flat = torch.pow((1.0 - terms["flat_score"]).clamp(0.0, 1.0), 3.0)
    anti_large = torch.pow((1.0 - terms["large_uniform"]).clamp(0.0, 1.0), 2.0)
    score = terms["mid_density"] * anti_flat * anti_large * terms["visibility_gate"] * terms["non_uniform_gate"]
    return _normalize_positive(score)


def _score_projection_anti_road_water_shadow(ctx):
    terms = _projection_inverse_bias_terms(ctx)
    anti_region = torch.pow((1.0 - terms["road_water_shadow_bias"]).clamp(0.0, 1.0), 4.0)
    inverse_common = torch.pow((1.0 - terms["common_bias"]).clamp(0.0, 1.0), 1.5)
    structure = 0.20 + 0.80 * terms["non_uniform_gate"]
    score = anti_region * inverse_common * terms["recoverable_gate"] * structure
    return _normalize_positive(score)


def _score_projection_inverse_area_with_visibility_gate(ctx):
    terms = _projection_inverse_bias_terms(ctx)
    anti_flat = torch.pow((1.0 - terms["flat_score"]).clamp(0.0, 1.0), 1.5)
    anti_large = torch.pow((1.0 - terms["large_uniform"]).clamp(0.0, 1.0), 1.5)
    score = terms["inverse_area"] * terms["recoverable_gate"] * anti_flat * anti_large * terms["non_uniform_gate"]
    return _normalize_positive(score)


SCORE_FNS = {
    "projected_area": _score_projected_area,
    "tile_complexity": _score_tile_complexity,
    "image_edge": _score_image_edge,
    "harris_corner": _score_harris_corner,
    "laplacian_blob": _score_laplacian_blob,
    "entropy": _score_entropy,
    "depth_boundary": _score_depth_boundary,
    "curvature": _score_curvature,
    "density_penalized_edge": _score_density_penalized_edge,
    "coarse_residual": _score_coarse_residual,
    "hybrid_structure": _score_hybrid_structure,
    "projection_image_edge": _score_projection_image_edge,
    "projection_density_penalized_edge": _score_projection_density_penalized_edge,
    "projection_entropy": _score_projection_entropy,
    "projection_harris_corner": _score_projection_harris_corner,
    "projection_laplacian_blob": _score_projection_laplacian_blob,
    "projection_hybrid_structure": _score_projection_hybrid_structure,
    "projection_knn_laplacian": _score_projection_knn_laplacian,
    "projection_plane_residual": _score_projection_plane_residual,
    "projection_multiscale_plane_residual": _score_projection_multiscale_plane_residual,
    "projection_depth_layer_boundary": _score_projection_depth_layer_boundary,
    "projection_occlusion_support": _score_projection_occlusion_support,
    "projection_support_gated_complexity": _score_projection_support_gated_complexity,
    "projection_density_suppressed_curvature": _score_projection_density_suppressed_curvature,
    "projection_gain_proxy": _score_projection_gain_proxy,
    "projection_structure_hybrid": _score_projection_structure_hybrid,
    "projection_flat_surface_suppressed_boundary": _score_projection_flat_surface_suppressed_boundary,
    "projection_component_penalized_structure": _score_projection_component_penalized_structure,
    "projection_object_boundary_gain": _score_projection_object_boundary_gain,
    "projection_learnable_detail_gain": _score_projection_learnable_detail_gain,
    "projected_area_flat_suppressed": _score_projected_area_flat_suppressed,
    "projection_inverse_common_bias": _score_projection_inverse_common_bias,
    "projection_hard_reject_flat_dense": _score_projection_hard_reject_flat_dense,
    "projection_mid_support_anti_flat": _score_projection_mid_support_anti_flat,
    "projection_anti_road_water_shadow": _score_projection_anti_road_water_shadow,
    "projection_inverse_area_with_visibility_gate": _score_projection_inverse_area_with_visibility_gate,
}

IMAGE_FEATURE_METHODS = {
    "image_edge",
    "harris_corner",
    "laplacian_blob",
    "entropy",
    "density_penalized_edge",
    "hybrid_structure",
}

RESIDUAL_METHODS = {
    "coarse_residual",
    "hybrid_structure",
}


def _allocate_detail_counts(scores, max_detail_slots, detail_budget, eps=1e-6):
    visible_count = int(scores.shape[0])
    max_detail_slots = int(max_detail_slots)
    detail_budget = max(0, min(int(detail_budget), visible_count * max_detail_slots))
    counts = torch.zeros(visible_count, dtype=torch.long, device=scores.device)
    if visible_count == 0 or max_detail_slots <= 0 or detail_budget <= 0:
        return counts

    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if scores.sum().item() <= eps:
        scores = torch.ones_like(scores)
    raw_counts = scores / scores.sum().clamp_min(eps) * float(detail_budget)
    counts = torch.floor(raw_counts).long().clamp(max=max_detail_slots)

    remaining = detail_budget - int(counts.sum().item())
    while remaining > 0:
        candidate_idx = torch.nonzero(counts < max_detail_slots, as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            break
        take_count = min(remaining, int(candidate_idx.numel()))
        priorities = raw_counts[candidate_idx] - counts[candidate_idx].float()
        chosen = candidate_idx[torch.topk(priorities, k=take_count, largest=True).indices]
        counts[chosen] += 1
        remaining -= take_count
    return counts


def _resize_panel_height(panel, height):
    if panel.shape[-2] == height:
        return panel
    width = max(1, int(round(panel.shape[-1] * float(height) / float(panel.shape[-2]))))
    return F.interpolate(panel.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]


def _save_maps(base_dir, idx, image, allocation_map, coverage_map, score_map, max_detail_slots, alpha, cmap):
    detail_dir = os.path.join(base_dir, "detail_map")
    heat_dir = os.path.join(base_dir, "detail_map_heat")
    panel_dir = os.path.join(base_dir, "detail_map_panel")
    score_dir = os.path.join(base_dir, "score_heat")
    makedirs(detail_dir, exist_ok=True)
    makedirs(heat_dir, exist_ok=True)
    makedirs(panel_dir, exist_ok=True)
    makedirs(score_dir, exist_ok=True)

    max_detail = float(max(1, int(max_detail_slots)))
    detail_heat = colorize(
        allocation_map.cpu(),
        cmap_name=cmap,
        range=(0.0, max_detail),
    ).to(device=image.device, dtype=image.dtype).permute(2, 0, 1)
    detail_heat_cbar = colorize(
        allocation_map.cpu(),
        cmap_name=cmap,
        range=(0.0, max_detail),
        append_cbar=True,
        cbar_in_image=False,
    ).to(dtype=image.dtype).permute(2, 0, 1)
    score_heat_cbar = colorize(
        score_map.cpu(),
        cmap_name=cmap,
        range=(0.0, 1.0),
        append_cbar=True,
        cbar_in_image=False,
    ).to(dtype=image.dtype).permute(2, 0, 1)

    density_mask = coverage_map > 1e-4
    strength = torch.clamp(allocation_map.to(device=image.device, dtype=image.dtype) / max_detail, 0.0, 1.0)
    alpha_map = float(max(0.0, min(1.0, alpha))) * density_mask.to(device=image.device, dtype=image.dtype)
    alpha_map = (alpha_map * (0.25 + 0.75 * strength)).unsqueeze(0)
    overlay = torch.clamp(image.detach() * (1.0 - alpha_map) + detail_heat * alpha_map, 0.0, 1.0)

    torchvision.utils.save_image(overlay, os.path.join(detail_dir, f"{idx:05d}.png"))
    torchvision.utils.save_image(detail_heat_cbar, os.path.join(heat_dir, f"{idx:05d}.png"))
    torchvision.utils.save_image(score_heat_cbar, os.path.join(score_dir, f"{idx:05d}.png"))

    panel_height = image.shape[-2]
    detail_panel = _resize_panel_height(detail_heat_cbar.to(device=image.device), panel_height)
    score_panel = _resize_panel_height(score_heat_cbar.to(device=image.device), panel_height)
    separator = torch.ones((3, panel_height, 8), dtype=image.dtype, device=image.device)
    panel = torch.cat([image.detach(), separator, detail_panel, separator, score_panel], dim=2)
    torchvision.utils.save_image(panel, os.path.join(panel_dir, f"{idx:05d}.png"))


def _resolve_detail_budget(args, pipeline, visible_count, max_detail_slots):
    explicit_budget = args.render_gaussian_budget
    config_budget = int(getattr(pipeline, "render_gaussian_budget", 0) or 0)
    render_budget = explicit_budget if explicit_budget is not None else config_budget
    if render_budget is not None and int(render_budget) > 0:
        return max(0, min(int(render_budget) - int(visible_count), int(visible_count) * int(max_detail_slots)))
    fraction = max(0.0, min(1.0, float(args.detail_budget_fraction)))
    return int(round(int(visible_count) * int(max_detail_slots) * fraction))


def _select_cameras(scene, split, max_views, view_stride):
    cameras = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
    stride = max(1, int(view_stride))
    cameras = cameras[::stride]
    if int(max_views) > 0:
        cameras = cameras[:int(max_views)]
    return cameras


def run(args):
    safe_state(args.quiet)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join("output", "coarse_allocation_scores", os.path.splitext(os.path.basename(args.config))[0])
    makedirs(output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        dataset, opt, pipeline = parse_cfg(cfg, args)

    if args.coarse_ply is not None:
        dataset.pretrain_path = args.coarse_ply
    if args.source_path is not None:
        dataset.source_path = args.source_path

    scene_cache_dir = os.path.join(output_dir, "_scene_cache")
    makedirs(scene_cache_dir, exist_ok=True)
    dataset.model_path = scene_cache_dir

    modules = __import__("scene")
    model_config = dataset.model_config
    model_kwargs = dict(model_config["kwargs"])
    detail_count_choices = getattr(dataset, "detail_count_choices", None)
    model_kwargs.setdefault("max_detail_slots", getattr(dataset, "detail_max_slots", 1))
    model_kwargs.setdefault("detail_count_choices", detail_count_choices)
    gaussians = getattr(modules, model_config["name"])(dataset.sh_degree, **model_kwargs)
    scene = LargeScene(dataset, gaussians, load_iteration=None, load_vq=False, shuffle=False)

    methods = DEFAULT_METHODS if args.methods == ["all"] else tuple(args.methods)
    unknown = sorted(set(methods) - set(SCORE_FNS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(SCORE_FNS)}")
    needs_image_features = bool(set(methods) & IMAGE_FEATURE_METHODS)
    needs_residual = bool(set(methods) & RESIDUAL_METHODS)

    max_detail_slots = int(args.max_detail_slots or getattr(dataset, "detail_max_slots", 1))
    tile_size = int(args.tile_size or getattr(pipeline, "projected_complexity_tile_size", 32))
    cameras = _select_cameras(scene, args.split, args.max_views, args.view_stride)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    summary = {
        "config": args.config,
        "coarse_ply": dataset.pretrain_path,
        "source_path": dataset.source_path,
        "split": args.split,
        "view_count": len(cameras),
        "methods": list(methods),
        "max_detail_slots": max_detail_slots,
        "detail_budget_fraction": float(args.detail_budget_fraction),
        "render_gaussian_budget": args.render_gaussian_budget,
        "tile_size": tile_size,
        "views": [],
    }

    for view_idx, cam_info in enumerate(tqdm(cameras, desc=f"Visualizing {args.split} coarse allocation")):
        viewpoint, image = _viewpoint_from_camera(dataset, cam_info, view_idx)
        vis_mask = prefilter_voxel(viewpoint, scene.gaussians, pipeline, background)
        visible_count = int(vis_mask.sum().item())
        if visible_count == 0:
            continue

        cache = _visible_cache_from_mask(scene.gaussians, vis_mask)
        residual_map = _render_coarse_residual_map(viewpoint, scene.gaussians, pipeline, cache, image) if needs_residual else None
        ctx = _build_context(
            viewpoint,
            image,
            cache,
            tile_size,
            residual_map=residual_map,
            build_image_features=needs_image_features,
        )
        detail_budget = _resolve_detail_budget(args, pipeline, visible_count, max_detail_slots)

        view_row = {
            "view_index": view_idx,
            "image_name": str(viewpoint.get("image_name", view_idx)),
            "visible_anchor_count": visible_count,
            "detail_budget": int(detail_budget),
            "method_stats": {},
        }

        for method in methods:
            scores = SCORE_FNS[method](ctx)
            detail_counts = _allocate_detail_counts(scores, max_detail_slots, detail_budget)
            allocation_map, coverage_map = _rasterize_anchor_values(
                viewpoint,
                scene.gaussians,
                pipeline,
                cache,
                detail_counts.float(),
                max_detail_slots,
            )
            score_map, score_coverage = _rasterize_anchor_values(
                viewpoint,
                scene.gaussians,
                pipeline,
                cache,
                _normalize_positive(scores),
                1.0,
            )
            if allocation_map is None or score_map is None:
                continue
            method_dir = os.path.join(output_dir, method)
            _save_maps(
                method_dir,
                view_idx,
                image,
                allocation_map,
                coverage_map,
                score_map,
                max_detail_slots,
                args.overlay_alpha,
                args.cmap,
            )
            score_detached = scores.detach().float()
            view_row["method_stats"][method] = {
                "score_mean": float(score_detached.mean().item()),
                "score_max": float(score_detached.max().item()),
                "score_min": float(score_detached.min().item()),
                "selected_detail_count": int(detail_counts.sum().item()),
            }
            del allocation_map, coverage_map, score_map, score_coverage

        summary["views"].append(view_row)

    with open(os.path.join(output_dir, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Output: {output_dir}")
    print(f"Views: {len(summary['views'])}")
    print(f"Methods: {', '.join(methods)}")


def build_parser():
    parser = ArgumentParser(description="Visualize coarse Gaussian allocation heatmaps for several score definitions.")
    parser.add_argument("--config", type=str, required=True, help="train config file path")
    parser.add_argument("--output_dir", type=str, default=None, help="directory where method folders are saved")
    parser.add_argument("--coarse_ply", type=str, default=None, help="override model_params.pretrain_path")
    parser.add_argument("--source_path", type=str, default=None, help="override model_params.source_path")
    parser.add_argument("--split", type=str, default="test", choices=("test", "train"))
    parser.add_argument("--max_views", type=int, default=-1, help="limit number of views; <=0 renders all")
    parser.add_argument("--view_stride", type=int, default=1, help="sample every Nth view")
    parser.add_argument("--methods", nargs="+", default=["all"], help=f"score methods or all. Available: {', '.join(DEFAULT_METHODS)}")
    parser.add_argument("--detail_budget_fraction", type=float, default=0.35, help="detail slot budget fraction when no render budget is set")
    parser.add_argument("--render_gaussian_budget", type=int, default=None, help="optional total render Gaussian budget; detail budget is budget-visible")
    parser.add_argument("--max_detail_slots", type=int, default=None, help="override model detail_max_slots")
    parser.add_argument("--tile_size", type=int, default=None, help="screen tile size for tile/depth scores")
    parser.add_argument("--overlay_alpha", type=float, default=0.35)
    parser.add_argument("--cmap", type=str, default="turbo")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_NOFILE, [11264, 65535])
    run(build_parser().parse_args(sys.argv[1:]))
