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
from utils.general_utils import safe_state, parse_cfg, resolve_render_gaussian_budget
from tqdm import tqdm
from os import makedirs
from utils.image_utils import psnr
from utils.log_utils import PerformanceMetricLogger, TrainingResourceLogger, tensorboard_log_image, wandb_log_image
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
        if self.base_budget < 0:
            return self.base_budget
        if self.base_budget == 0:
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


class StageBudgetDecaySchedule:
    def __init__(self, pipe, max_detail_slots, joint_start_iter):
        self.enabled = bool(getattr(pipe, "stage_budget_decay_schedule", False))
        if self.enabled and bool(getattr(pipe, "stage_budget_adjustment", False)):
            raise ValueError("stage_budget_decay_schedule cannot be used with stage_budget_adjustment")

        self.max_detail_slots = max(1, int(max_detail_slots or 1))
        self.joint_start_iter = int(joint_start_iter)
        self.decay_iters = max(0, int(getattr(pipe, "stage_budget_decay_iters", 20_000) or 0))
        min_detail_multiplier = getattr(pipe, "stage_budget_decay_min_detail_multiplier", None)
        if min_detail_multiplier is None:
            min_detail_multiplier = float(self.max_detail_slots) / 2.0
        min_detail_multiplier = max(0.0, min(float(min_detail_multiplier), float(self.max_detail_slots)))
        self.detail_multiplier = 1.0 + float(self.max_detail_slots)
        self.joint_multiplier = 1.0 + min_detail_multiplier

    @property
    def effective_joint_start_iter(self):
        if not self.enabled:
            return self.joint_start_iter
        return self.joint_start_iter + self.decay_iters

    def multiplier_for(self, iteration):
        if not self.enabled:
            return None

        iteration = int(iteration)
        if iteration < self.joint_start_iter:
            return self.detail_multiplier
        if self.decay_iters <= 0 or iteration >= self.effective_joint_start_iter:
            return self.joint_multiplier

        progress = float(iteration - self.joint_start_iter) / float(self.decay_iters)
        progress = max(0.0, min(progress, 1.0))
        return self.detail_multiplier + (self.joint_multiplier - self.detail_multiplier) * progress

    def current_scale(self, iteration):
        if not self.enabled:
            return 1.0
        return self.multiplier_for(iteration) / self.detail_multiplier

    def budget_for(self, iteration, visible_anchor_count):
        if not self.enabled:
            return None
        visible_anchor_count = int(visible_anchor_count)
        return int(round(visible_anchor_count * self.multiplier_for(iteration)))

    def stats(self, iteration, visible_anchor_count, effective_budget):
        if not self.enabled:
            return {}
        base_budget = int(round(int(visible_anchor_count) * self.detail_multiplier))
        return {
            "base_render_gaussian_budget": base_budget,
            "effective_render_gaussian_budget": int(effective_budget),
            "stage_budget_scale": float(self.current_scale(iteration)),
        }


def _resolve_iteration_render_budget(pipe, visible_anchor_count, iteration, requested_render_gaussian_budget,
                                     budget_decay_schedule=None):
    if budget_decay_schedule is not None and budget_decay_schedule.enabled:
        return budget_decay_schedule.budget_for(iteration, visible_anchor_count)
    return resolve_render_gaussian_budget(
        requested_render_gaussian_budget,
        visible_anchor_count,
        getattr(pipe, "render_gaussian_budget_multiplier", 0.0),
    )


def _camera_pose(viewpoint):
    transform = viewpoint["world_view_transform"]
    if isinstance(transform, torch.Tensor):
        while transform.dim() > 2:
            transform = transform[0]
    return transform[-1, :-1]


def _camera_center(viewpoint):
    center = viewpoint.get("camera_center")
    if isinstance(center, torch.Tensor):
        while center.dim() > 1:
            center = center[0]
    return center


def _visible_hash_input(gaussians, vis_mask):
    return [
        gaussians.get_xyz[vis_mask].detach(),
        gaussians.get_scaling[vis_mask].detach(),
        gaussians.get_rotation[vis_mask].detach(),
        gaussians.get_offset_slots[vis_mask].detach(),
        torch.nonzero(vis_mask, as_tuple=False).flatten().detach(),
    ]


def _gate_step_kwargs(pipe, iteration=None, training=False):
    return {
        "allocation_mode": getattr(pipe, "allocation_mode", "proposal"),
        "training": training,
        "iteration": iteration,
        "gate_train_mode": getattr(pipe, "gate_train_mode", "soft_all"),
        "gate_eval_mode": getattr(pipe, "gate_eval_mode", "topk"),
        "gate_temperature_init": getattr(pipe, "gate_temperature_init", 1.0),
        "gate_temperature_final": getattr(pipe, "gate_temperature_final", 0.2),
        "gate_temperature_max_steps": getattr(pipe, "gate_temperature_max_steps", 30000),
        "gate_budget_lambda": getattr(pipe, "gate_budget_lambda", 0.01),
        "gate_binary_lambda": getattr(pipe, "gate_binary_lambda", 0.001),
        "gate_feature_mode": getattr(pipe, "gate_feature_mode", "detail_view"),
        "detail_feature_mode": getattr(pipe, "detail_feature_mode", "detail_hash"),
        "gate_opacity_mode": getattr(pipe, "gate_opacity_mode", "st_identity"),
        "gate_all_detail_until": getattr(pipe, "gate_all_detail_until", 0),
        "clone_score_warmup_until": getattr(pipe, "clone_score_warmup_until", 20000),
        "clone_score_ramp_until": getattr(pipe, "clone_score_ramp_until", 40000),
        "clone_score_freeze_after": getattr(pipe, "clone_score_freeze_after", 220000),
        "clone_score_ema_beta": getattr(pipe, "clone_score_ema_beta", 0.95),
        "clone_score_eps": getattr(pipe, "clone_score_eps", 1e-6),
        "clone_score_detail_grad_weight": getattr(pipe, "clone_score_detail_grad_weight", 1.0),
        "clone_score_grad_clip": getattr(pipe, "clone_score_grad_clip", 0.0),
        "projected_area_all_detail_until": getattr(pipe, "projected_area_all_detail_until", 50000),
        "projected_area_detail_stage_full_budget": getattr(pipe, "projected_area_detail_stage_full_budget", True),
        "projected_area_scale_power": getattr(pipe, "projected_area_scale_power", 2.0),
        "projected_area_distance_power": getattr(pipe, "projected_area_distance_power", 2.0),
        "projected_area_eps": getattr(pipe, "projected_area_eps", 1e-6),
    }


def _render_with_allocation_budget(
        viewpoint, gaussians, mixgs, pipe, background, vis_mask, render_gaussian_budget,
        iteration=None, training=False):
    hash_input = _visible_hash_input(gaussians, vis_mask)
    decoded_data = mixgs.step(
        hash_input,
        _camera_pose(viewpoint),
        camera_center=_camera_center(viewpoint),
        render_gaussian_budget=render_gaussian_budget,
        scale_min=getattr(pipe, "scale_min", 0.0),
        **_gate_step_kwargs(pipe, iteration=iteration, training=training),
    )
    allocation_mode = str(getattr(pipe, "allocation_mode", "proposal") or "proposal").lower()
    capture_clone_grad = allocation_mode in ("clone_score", "clone_score_allocation") and bool(training)
    if capture_clone_grad:
        freeze_after = int(getattr(pipe, "clone_score_freeze_after", 220000) or 0)
        capture_clone_grad = iteration is None or freeze_after <= 0 or int(iteration) <= freeze_after
    render_pkg = render_mix(
        viewpoint,
        gaussians,
        pipe,
        background,
        vis_mask,
        decoded_data,
        capture_viewspace_grad=capture_clone_grad,
    )
    return render_pkg, decoded_data


def _render_with_proposal_budget(
        viewpoint, gaussians, mixgs, pipe, background, vis_mask, render_gaussian_budget,
        iteration=None, training=False):
    return _render_with_allocation_budget(
        viewpoint,
        gaussians,
        mixgs,
        pipe,
        background,
        vis_mask,
        render_gaussian_budget,
        iteration=iteration,
        training=training,
    )


def _gate_utility_default_stats(device=None):
    zero = torch.zeros((), device=device or "cuda") if torch.cuda.is_available() else torch.zeros(())
    return zero, {
        "gate_utility_loss": 0.0,
        "gate_utility_samples": 0,
        "gate_utility_mean": 0.0,
        "gate_utility_positive_ratio": 0.0,
        "gate_utility_eval_time_ms": 0.0,
    }


def _decoded_data_with_detail_mask(decoded_data, keep_mask):
    masked = dict(decoded_data)
    for field in ("d_color", "d_rotation", "d_scaling", "d_opacity", "detail_anchor_idx", "detail_slot_idx"):
        masked[field] = decoded_data[field][keep_mask]
    detail_counts = decoded_data.get("detail_counts")
    if detail_counts is not None:
        anchor_count = int(detail_counts.shape[0])
        if masked["detail_anchor_idx"].numel() > 0:
            masked["detail_counts"] = torch.bincount(
                masked["detail_anchor_idx"],
                minlength=anchor_count,
            ).to(dtype=torch.long)
        else:
            masked["detail_counts"] = torch.zeros(anchor_count, device=keep_mask.device, dtype=torch.long)
    return masked


def _mse_metric(image, gt_image):
    return torch.mean((image - gt_image) ** 2)


def _gate_utility_ranking_loss(scores, targets, min_delta=0.0):
    if scores.numel() < 2:
        return scores.new_zeros(())
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    valid = torch.abs(target_diff) > float(min_delta or 0.0)
    if not torch.any(valid):
        return scores.new_zeros(())
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    target_sign = torch.sign(target_diff[valid])
    return torch.nn.functional.softplus(-target_sign * score_diff[valid]).mean()


def _compute_gate_utility_loss(
        viewpoint,
        gaussians,
        pipe,
        background,
        vis_mask,
        decoded_data,
        image,
        gt_image,
        iteration,
):
    selected_logits = decoded_data.get("gate_selected_logits")
    device = image.device
    zero_loss, stats = _gate_utility_default_stats(device)
    if not bool(getattr(pipe, "gate_utility_loss", False)):
        return zero_loss, stats
    if str(getattr(pipe, "allocation_mode", "proposal") or "proposal").lower() not in ("gate", "learned_gate"):
        return zero_loss, stats
    if selected_logits is None or selected_logits.numel() < 2:
        return zero_loss, stats

    interval = max(0, int(getattr(pipe, "gate_utility_interval", 100) or 0))
    if interval <= 0 or int(iteration) % interval != 0:
        return zero_loss, stats

    selected_count = int(selected_logits.numel())
    sample_count = min(max(0, int(getattr(pipe, "gate_utility_sample_count", 64) or 0)), selected_count)
    if sample_count < 2:
        return zero_loss, stats

    sample_pos = torch.randperm(selected_count, device=device)[:sample_count]
    group_size = max(1, int(getattr(pipe, "gate_utility_group_size", 8) or 1))
    eval_time_ms = 0.0
    utility_positions = []
    utility_targets = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    utility_start = time.time()
    with torch.no_grad():
        mse_with = _mse_metric(image.detach(), gt_image.detach())
        for group_pos in torch.split(sample_pos, group_size):
            keep_mask = torch.ones(selected_count, device=device, dtype=torch.bool)
            keep_mask[group_pos] = False
            masked_decoded = _decoded_data_with_detail_mask(decoded_data, keep_mask)
            counterfactual_pkg = render_mix(viewpoint, gaussians, pipe, background, vis_mask, masked_decoded)
            mse_without = _mse_metric(counterfactual_pkg["render"].detach(), gt_image.detach())
            utility_value = (mse_without - mse_with) / max(1, int(group_pos.numel()))
            utility_positions.append(group_pos.detach())
            utility_targets.append(utility_value.to(dtype=selected_logits.dtype).expand(group_pos.numel()).detach())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_time_ms = (time.time() - utility_start) * 1000.0

    if not utility_positions:
        stats["gate_utility_eval_time_ms"] = float(eval_time_ms)
        return zero_loss, stats

    known_pos_tensor = torch.cat(utility_positions, dim=0).to(device=device, dtype=torch.long)
    targets = torch.cat(utility_targets, dim=0).to(device=device, dtype=selected_logits.dtype)
    if known_pos_tensor.numel() < 2:
        stats["gate_utility_eval_time_ms"] = float(eval_time_ms)
        return zero_loss, stats

    scores = selected_logits[known_pos_tensor]
    utility_loss = _gate_utility_ranking_loss(
        scores,
        targets,
        min_delta=float(getattr(pipe, "gate_utility_min_delta", 0.0) or 0.0),
    )
    utility_loss = utility_loss * float(getattr(pipe, "gate_utility_lambda", 0.01) or 0.0)

    with torch.no_grad():
        stats.update({
            "gate_utility_loss": float(utility_loss.detach().item()),
            "gate_utility_samples": int(targets.numel()),
            "gate_utility_mean": float(targets.detach().mean().item()),
            "gate_utility_positive_ratio": float((targets.detach() > 0).to(torch.float32).mean().item()),
            "gate_utility_eval_time_ms": float(eval_time_ms),
        })
    return utility_loss, stats


def training(dataset, opt, pipe, testing_iterations, saving_iterations, refilter_iterations, checkpoint_iterations,
             checkpoint, max_cache_num, debug_from, metric_log_interval):
    first_iter = 0
    training_start_timestamp = time.time()
    training_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_timestamp))
    log_writer, image_logger = prepare_output_and_logger(dataset)
    metric_logger = PerformanceMetricLogger(
        getattr(log_writer, "log_dir", None) or dataset.model_path,
        metric_log_interval,
    )
    detail_count_choices = getattr(dataset, "detail_count_choices", None)
    resource_log_interval = int(getattr(pipe, "resource_log_interval", 0) or 0)
    resource_plot_on_complete = bool(getattr(pipe, "resource_plot_on_complete", False))
    resource_logger = None
    if resource_log_interval > 0 and log_writer:
        resource_logger = TrainingResourceLogger(
            getattr(log_writer, "log_dir", None) or dataset.model_path,
            getattr(dataset, "detail_max_slots", 1),
            detail_count_choices,
        )
    stage_budget_controller = StageBudgetController(pipe, getattr(opt, "joint_start_iter", opt.iterations + 1))

    modules = __import__('scene')
    model_config = dataset.model_config
    model_kwargs = dict(model_config['kwargs'])
    model_kwargs.setdefault("max_detail_slots", getattr(dataset, "detail_max_slots", 1))
    model_kwargs.setdefault("detail_count_choices", detail_count_choices)
    gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_kwargs)

    use_slot_embedding = bool(getattr(pipe, "use_slot_embedding", False))
    mixgs = MixGSModel(
        hash_args=dataset.hash_args,
        net_args=dataset.network_args,
        max_detail_slots=getattr(dataset, "detail_max_slots", 1),
        detail_count_choices=detail_count_choices,
        gate_feature_mode=getattr(pipe, "gate_feature_mode", "detail_view"),
        gate_view_context_dim=getattr(pipe, "gate_view_context_dim", 32),
        use_slot_embedding=use_slot_embedding,
    )
    budget_decay_schedule = StageBudgetDecaySchedule(
        pipe,
        mixgs.max_detail_slots,
        getattr(opt, "joint_start_iter", opt.iterations + 1),
    )
    effective_joint_start_iter = budget_decay_schedule.effective_joint_start_iter
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

            if iteration == effective_joint_start_iter:
                gaussians.gaussian_training()

            requested_render_gaussian_budget = stage_budget_controller.budget_for(iteration)
            stage_budget_scale = stage_budget_controller.current_scale(iteration)
            vis_mask = prefilter_voxel(cam_info, gaussians, pipe, background)
            visible_anchor_count = int(vis_mask.sum().item())
            effective_render_gaussian_budget = _resolve_iteration_render_budget(
                pipe,
                visible_anchor_count,
                iteration,
                requested_render_gaussian_budget,
                budget_decay_schedule,
            )
            if budget_decay_schedule.enabled:
                stage_budget_scale = budget_decay_schedule.current_scale(iteration)
            gt_image = gt_image.cuda()
            render_pkg, decoded_data = _render_with_allocation_budget(
                cam_info,
                gaussians,
                mixgs,
                pipe,
                background,
                vis_mask,
                effective_render_gaussian_budget,
                iteration=iteration,
                training=True,
            )

            image, radii = render_pkg["render"], render_pkg["radii"]
            end = time.time()
            ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render

            # Loss
            start = time.time()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
            gate_losses = decoded_data.get("gate_losses")
            if gate_losses:
                loss = loss + gate_losses.get("loss", image.new_zeros(()))
            gate_utility_loss, gate_utility_stats = _compute_gate_utility_loss(
                cam_info,
                gaussians,
                pipe,
                background,
                vis_mask,
                decoded_data,
                image,
                gt_image,
                iteration,
            )
            loss = loss + gate_utility_loss

            loss.backward()
            clone_score_update_stats = mixgs.update_clone_scores(
                render_pkg,
                decoded_data,
                ema_beta=getattr(pipe, "clone_score_ema_beta", 0.95),
                eps=getattr(pipe, "clone_score_eps", 1e-6),
                detail_grad_weight=getattr(pipe, "clone_score_detail_grad_weight", 1.0),
                grad_clip=getattr(pipe, "clone_score_grad_clip", 0.0),
                frozen=bool(decoded_data.get("clone_score_update_enabled") is False),
            ) if str(getattr(pipe, "allocation_mode", "proposal") or "proposal").lower() in ("clone_score", "clone_score_allocation") else {}
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

                budget_stats = render_pkg.get("budget_stats", {})
                budget_stats.update(gate_utility_stats)
                budget_stats.update(clone_score_update_stats)
                observed_vram_mb = 0.0
                if torch.cuda.is_available():
                    observed_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
                stage_budget_controller.observe(iteration, observed_vram_mb)
                if budget_decay_schedule.enabled:
                    budget_stats.update(
                        budget_decay_schedule.stats(
                            iteration,
                            visible_anchor_count,
                            effective_render_gaussian_budget,
                        )
                    )
                else:
                    budget_stats.update(stage_budget_controller.stats(effective_render_gaussian_budget, stage_budget_scale))
                if should_log_resources:
                    resource_logger.log(iteration, budget_stats, decoded_data.get("detail_counts"))

                ema_time = {
                    "render": ema_time_render,
                    "loss": ema_time_loss,
                    "num_points": radii.shape[0],
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
                    "allocation_mode_id": budget_stats.get("allocation_mode_id", 0.0),
                    "allocation_warmup_active": budget_stats.get("allocation_warmup_active", 0.0),
                    "gate_mass": budget_stats.get("gate_mass", 0.0),
                    "gate_mean": budget_stats.get("gate_mean", 0.0),
                    "gate_budget_loss": budget_stats.get("gate_budget_loss", 0.0),
                    "gate_binary_loss": budget_stats.get("gate_binary_loss", 0.0),
                    "gate_utility_loss": budget_stats.get("gate_utility_loss", 0.0),
                    "gate_utility_samples": budget_stats.get("gate_utility_samples", 0),
                    "gate_utility_mean": budget_stats.get("gate_utility_mean", 0.0),
                    "gate_utility_positive_ratio": budget_stats.get("gate_utility_positive_ratio", 0.0),
                    "gate_utility_eval_time_ms": budget_stats.get("gate_utility_eval_time_ms", 0.0),
                    "clone_score_stage_id": budget_stats.get("clone_score_stage_id", 0.0),
                    "clone_score_mean": budget_stats.get("clone_score_mean", 0.0),
                    "clone_score_max": budget_stats.get("clone_score_max", 0.0),
                    "clone_score_updated_count": budget_stats.get("clone_score_updated_count", 0),
                    "clone_score_signal_mean": budget_stats.get("clone_score_signal_mean", 0.0),
                    "clone_score_signal_max": budget_stats.get("clone_score_signal_max", 0.0),
                    "clone_score_frozen": budget_stats.get("clone_score_frozen", 0),
                    "clone_target_detail_budget": budget_stats.get("clone_target_detail_budget", 0),
                    "clone_effective_detail_budget": budget_stats.get("clone_effective_detail_budget", 0),
                    "clone_score_ramp_progress": budget_stats.get("clone_score_ramp_progress", 0.0),
                }

                lr = {}
                for param_group in gaussians.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                for param_group in mixgs.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                # Log and save
                training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr,
                                iter_start.elapsed_time(iter_end), testing_iterations, scene, mixgs, (pipe, background),
                                metric_logger, training_start_timestamp, training_start_time,
                                effective_joint_start_iter, budget_decay_schedule)

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
    if metric_logger is not None and metric_log_interval and metric_log_interval > 0:
        metric_logger.plot()


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
                    testing_iterations, scene: LargeScene, mixgs, renderArgs, metric_logger,
                    training_start_timestamp, training_start_time, joint_start_iter,
                    budget_decay_schedule=None):
    if log_writer:
        metrics_to_log = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_time/render": ema_time["render"],
            "train_time/loss": ema_time["loss"],
            "train_time/num_points": ema_time["num_points"],
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
        for key in (
            "allocation_mode_id",
            "allocation_warmup_active",
            "gate_mass",
            "gate_mean",
            "gate_budget_loss",
            "gate_binary_loss",
            "gate_utility_loss",
            "gate_utility_samples",
            "gate_utility_mean",
            "gate_utility_positive_ratio",
            "gate_utility_eval_time_ms",
        ):
            if key in ema_time:
                metrics_to_log["train_gate/" + key] = ema_time[key]
        for key in (
            "projected_area_stage_id",
            "projected_area_all_detail_active",
            "projected_area_target_detail_budget",
            "projected_area_effective_detail_budget",
            "projected_area_score_mean",
            "projected_area_score_max",
            "projected_area_score_min",
        ):
            if key in ema_time:
                metrics_to_log["train_projected_area/" + key] = ema_time[key]
        for key in (
            "clone_score_stage_id",
            "clone_score_mean",
            "clone_score_max",
            "clone_score_updated_count",
            "clone_score_signal_mean",
            "clone_score_signal_max",
            "clone_score_frozen",
            "clone_target_detail_budget",
            "clone_effective_detail_budget",
            "clone_score_ramp_progress",
        ):
            if key in ema_time:
                metrics_to_log["train_clone_score/" + key] = ema_time[key]
        for key, value in lr.items():
            metrics_to_log["trainer/" + key] = value
        log_writer.log_metrics(metrics_to_log, iteration)

    # Report test and samples of training set
    should_log_metrics = metric_logger is not None and metric_logger.should_log(iteration)
    if iteration in testing_iterations or should_log_metrics:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                metric_start_timestamp = time.time()
                metric_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric_start_timestamp))
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
                    render_gaussian_budget = _resolve_iteration_render_budget(
                        renderArgs[0],
                        int(vis_mask.sum().item()),
                        iteration,
                        getattr(renderArgs[0], "render_gaussian_budget", 0),
                        budget_decay_schedule,
                    )

                    gt_image = torch.clamp(org_img.to("cuda"), 0.0, 1.0)
                    render_pkg, decoded_data = _render_with_proposal_budget(
                        viewpoint,
                        scene.gaussians,
                        mixgs,
                        renderArgs[0],
                        renderArgs[1],
                        vis_mask,
                        render_gaussian_budget,
                        iteration=iteration,
                    )

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                    if log_writer and (iteration in testing_iterations) and (idx < 5):
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

                metric_end_timestamp = time.time()
                metric_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metric_end_timestamp))
                metric_elapsed_sec = metric_end_timestamp - metric_start_timestamp
                elapsed_since_training_start_sec = metric_end_timestamp - training_start_timestamp
                metrics = {
                    "l1_loss": l1_test,
                    "psnr": psnr_test,
                    "ssim": ssims_test,
                    "lpips": lpips_test,
                }
                metric_timing = {
                    "training_start_time": training_start_time,
                    "metric_start_time": metric_start_time,
                    "metric_end_time": metric_end_time,
                    "metric_elapsed_sec": metric_elapsed_sec,
                    "elapsed_since_training_start_sec": elapsed_since_training_start_sec,
                }
                metrics_to_log = {
                    config['name'] + '/loss_viewpoint/l1_loss': metrics["l1_loss"],
                    config['name'] + '/loss_viewpoint/psnr': metrics["psnr"],
                    config['name'] + '/loss_viewpoint/ssim': metrics["ssim"],
                    config['name'] + '/loss_viewpoint/lpips': metrics["lpips"],
                    config['name'] + '/metric_time/elapsed_sec': metric_elapsed_sec,
                    config['name'] + '/metric_time/elapsed_since_training_start_sec': elapsed_since_training_start_sec,
                }

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} EvalTime {:.2f}s".format(iteration, config['name'], l1_test, psnr_test, ssims_test, lpips_test, metric_elapsed_sec))
                if log_writer:
                    log_writer.log_metrics(metrics_to_log, iteration)
                if should_log_metrics:
                    metric_logger.log(iteration, config['name'], len(config['cameras']), metrics, metric_timing)

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
    parser.add_argument("--metric_log_interval", type=int, default=None)
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
             args.checkpoint_iterations, args.start_checkpoint, args.max_cache_num, args.debug_from,
             pp.metric_log_interval)

    # All done
    print("\nTraining complete.")
