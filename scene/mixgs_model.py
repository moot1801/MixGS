import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.network import GSDecoder, GSEncoder
from scene.gate_allocation import GateAllocator, SlotGateSTAllocator
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, resolve_detail_count_choices


class ViewContextEncoder(nn.Module):
    def __init__(self, spatial_dim, context_dim=32, hidden_dim=64):
        super().__init__()
        context_dim = max(2, int(context_dim or 32))
        pooled_dim = max(1, context_dim // 2)
        self.context_dim = pooled_dim * 2
        self.anchor_mlp = nn.Sequential(
            nn.Linear(spatial_dim + 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pooled_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, anchor_hash, view_dir, distance, scale_mean):
        if anchor_hash.numel() == 0:
            return anchor_hash.new_zeros((self.context_dim,))
        anchor_feature = torch.cat([anchor_hash, view_dir, distance, scale_mean], dim=-1)
        pooled_feature = self.anchor_mlp(anchor_feature)
        return torch.cat(
            [
                pooled_feature.mean(dim=0),
                pooled_feature.max(dim=0).values,
            ],
            dim=-1,
        )


class MixGSModel:
    def __init__(
            self,
            hash_args,
            net_args,
            max_detail_slots=1,
            detail_count_choices=None,
            gate_feature_mode="detail_view",
            gate_view_context_dim=32,
            use_slot_embedding=False,
    ):
        self.encoder = GSEncoder(**hash_args).cuda()
        self.spatial_dim = self.encoder.canonical_level_dim * self.encoder.canonical_num_levels
        self.mlp_dim = 10
        self.max_detail_slots, self.detail_count_choices, _ = resolve_detail_count_choices(
            max_detail_slots, detail_count_choices
        )
        net_args = dict(net_args)
        net_args["max_detail_slots"] = self.max_detail_slots
        net_args["use_slot_embedding"] = bool(use_slot_embedding)
        self.use_slot_embedding = bool(use_slot_embedding)
        self.decoder = GSDecoder(spatial_in_dim=self.spatial_dim, mlp_in_dim=self.mlp_dim, **net_args).cuda()
        self.gate_feature_mode = str(gate_feature_mode or "detail_view").lower()
        self.gate_uses_view_context = self.gate_feature_mode in ("hash_view_context", "view_context")
        self.view_context_encoder = (
            ViewContextEncoder(self.spatial_dim, gate_view_context_dim).cuda()
            if self.gate_uses_view_context else None
        )
        self.gate_extra_dim = self.view_context_encoder.context_dim if self.gate_uses_view_context else 8
        self.gate_feature_dim = self.spatial_dim + self.gate_extra_dim
        self.gate_allocator = GateAllocator(self.gate_feature_dim).cuda()
        self.slot_gate_allocator = SlotGateSTAllocator(self.gate_feature_dim).cuda()
        self._gate_used = False
        self._slot_gate_used = False
        self.clone_score_ema = None
        self.clone_score_seen_ema = None
        self._clone_score_used = False

        self.decoder_lr_scale = 50.0
        self.encoder_lr_scale = 100.0

    def _quantize_detail_counts(self, counts):
        if counts.numel() == 0:
            return counts
        quantized = torch.zeros_like(counts)
        for choice in self.detail_count_choices:
            quantized = torch.where(counts >= choice, counts.new_full((), choice), quantized)
        return quantized

    def _projected_area_uses_contiguous_choices(self):
        return self.detail_count_choices == list(range(1, self.max_detail_slots + 1))

    def _allocate_projected_area_counts(self, scores, render_gaussian_budget):
        if not self._projected_area_uses_contiguous_choices():
            return self._allocate_detail_counts(scores, render_gaussian_budget)

        visible_count = scores.shape[0]
        device = scores.device
        counts = torch.zeros(visible_count, dtype=torch.long, device=device)
        render_gaussian_budget = int(render_gaussian_budget or 0)

        if visible_count == 0:
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": 0,
                "detail_budget": 0,
                "selected_detail_count": 0,
                "budget_overflow": False,
            }

        if render_gaussian_budget <= 0:
            detail_count = self.detail_count_choices[0]
            counts.fill_(detail_count)
            selected_detail_count = visible_count * detail_count
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": selected_detail_count,
                "selected_detail_count": selected_detail_count,
                "budget_overflow": False,
            }

        detail_budget = max(render_gaussian_budget - visible_count, 0)
        detail_budget = min(detail_budget, visible_count * self.max_detail_slots)
        budget_overflow = visible_count > render_gaussian_budget
        if detail_budget == 0:
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": 0,
                "selected_detail_count": 0,
                "budget_overflow": budget_overflow,
            }
        if detail_budget == visible_count * self.max_detail_slots:
            counts.fill_(self.max_detail_slots)
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": detail_budget,
                "selected_detail_count": detail_budget,
                "budget_overflow": budget_overflow,
            }

        scores = torch.clamp_min(scores, 0.0)
        score_sum = torch.clamp_min(scores.sum(), torch.finfo(scores.dtype).eps)
        raw_counts = scores / score_sum * detail_budget
        counts = torch.floor(raw_counts).to(torch.long).clamp(max=self.max_detail_slots)

        remaining = detail_budget - int(counts.sum().item())
        while remaining > 0:
            candidate_idx = torch.nonzero(counts < self.max_detail_slots, as_tuple=False).flatten()
            if candidate_idx.numel() == 0:
                break
            take_count = min(remaining, candidate_idx.numel())
            priorities = raw_counts[candidate_idx] - counts[candidate_idx].to(raw_counts.dtype)
            chosen = candidate_idx[torch.topk(priorities, k=take_count, largest=True).indices]
            counts[chosen] += 1
            remaining -= take_count

        return counts, {
            "render_gaussian_budget": render_gaussian_budget,
            "visible_anchor_count": visible_count,
            "detail_budget": detail_budget,
            "selected_detail_count": detail_budget - max(remaining, 0),
            "budget_overflow": budget_overflow,
        }

    def _allocate_detail_counts(self, scores, render_gaussian_budget):
        visible_count = scores.shape[0]
        device = scores.device
        counts = torch.zeros(visible_count, dtype=torch.long, device=device)
        render_gaussian_budget = int(render_gaussian_budget or 0)

        if visible_count == 0:
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": 0,
                "detail_budget": 0,
                "selected_detail_count": 0,
                "budget_overflow": False,
            }

        if render_gaussian_budget <= 0:
            detail_count = self.detail_count_choices[0]
            counts.fill_(detail_count)
            selected_detail_count = visible_count * detail_count
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": selected_detail_count,
                "selected_detail_count": selected_detail_count,
                "budget_overflow": False,
            }

        detail_budget = max(render_gaussian_budget - visible_count, 0)
        detail_budget = min(detail_budget, visible_count * self.max_detail_slots)
        budget_overflow = visible_count > render_gaussian_budget
        if detail_budget == 0:
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": 0,
                "selected_detail_count": 0,
                "budget_overflow": budget_overflow,
            }
        if detail_budget == visible_count * self.max_detail_slots:
            counts.fill_(self.max_detail_slots)
            return counts, {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": detail_budget,
                "selected_detail_count": detail_budget,
                "budget_overflow": budget_overflow,
            }

        scores = torch.clamp_min(scores, 0.0)
        if scores.sum().item() <= 0.0:
            scores = torch.ones_like(scores)
        raw_counts = scores / scores.sum() * detail_budget
        counts = torch.floor(raw_counts).to(torch.long).clamp(max=self.max_detail_slots)
        counts = self._quantize_detail_counts(counts)

        remaining = detail_budget - int(counts.sum().item())
        for _ in self.detail_count_choices:
            if remaining <= 0:
                break
            next_counts = torch.full_like(counts, self.max_detail_slots + 1)
            for choice in self.detail_count_choices:
                choice_tensor = counts.new_full(counts.shape, choice)
                next_counts = torch.where((counts < choice) & (next_counts > choice), choice_tensor, next_counts)
            increments = next_counts - counts
            candidate_idx = torch.nonzero(
                (next_counts <= self.max_detail_slots) & (increments > 0) & (increments <= remaining),
                as_tuple=False,
            ).flatten()
            if candidate_idx.numel() == 0:
                break
            candidate_increments = increments[candidate_idx]
            priorities = (raw_counts[candidate_idx] - counts[candidate_idx].to(raw_counts.dtype)) / candidate_increments.to(raw_counts.dtype)
            order = torch.argsort(priorities, descending=True)
            ordered_idx = candidate_idx[order]
            ordered_increments = candidate_increments[order]
            cumulative = torch.cumsum(ordered_increments, dim=0)
            take_mask = cumulative <= remaining
            if not torch.any(take_mask):
                break
            chosen = ordered_idx[take_mask]
            counts[chosen] = next_counts[chosen]
            remaining -= int(increments[chosen].sum().item())

        selected_detail_count = int(counts.sum().item())
        return counts, {
            "render_gaussian_budget": render_gaussian_budget,
            "visible_anchor_count": visible_count,
            "detail_budget": detail_budget,
            "selected_detail_count": selected_detail_count,
            "budget_overflow": budget_overflow,
        }

    def _selected_indices_from_counts(self, counts):
        anchor_count = counts.shape[0]
        anchor_idx = torch.repeat_interleave(torch.arange(anchor_count, device=counts.device), counts)
        if anchor_idx.numel() == 0:
            return anchor_idx, torch.empty(0, dtype=torch.long, device=counts.device)
        starts = torch.cumsum(counts, dim=0) - counts
        slot_idx = torch.arange(anchor_idx.shape[0], device=counts.device) - torch.repeat_interleave(starts, counts)
        return anchor_idx, slot_idx.to(torch.long)

    def _ensure_clone_score_buffers(self, anchor_indices):
        if anchor_indices is None or anchor_indices.numel() == 0:
            return
        anchor_indices = anchor_indices.detach().to(dtype=torch.long)
        required_size = int(anchor_indices.max().item()) + 1
        device = anchor_indices.device
        if self.clone_score_ema is None:
            self.clone_score_ema = torch.zeros(required_size, dtype=torch.float32, device=device)
            self.clone_score_seen_ema = torch.zeros(required_size, dtype=torch.float32, device=device)
            return
        if self.clone_score_ema.device != device:
            self.clone_score_ema = self.clone_score_ema.to(device=device)
            self.clone_score_seen_ema = self.clone_score_seen_ema.to(device=device)
        if self.clone_score_ema.shape[0] < required_size:
            pad_size = required_size - self.clone_score_ema.shape[0]
            self.clone_score_ema = torch.cat([
                self.clone_score_ema,
                torch.zeros(pad_size, dtype=self.clone_score_ema.dtype, device=device),
            ], dim=0)
            self.clone_score_seen_ema = torch.cat([
                self.clone_score_seen_ema,
                torch.zeros(pad_size, dtype=self.clone_score_seen_ema.dtype, device=device),
            ], dim=0)

    def _clone_visible_scores(self, anchor_indices, coords, eps=1e-6):
        if anchor_indices is None or anchor_indices.numel() == 0:
            return coords.new_ones((coords.shape[0],))
        self._ensure_clone_score_buffers(anchor_indices)
        if self.clone_score_ema is None:
            return coords.new_ones((coords.shape[0],))
        anchor_indices = anchor_indices.to(device=self.clone_score_ema.device, dtype=torch.long)
        seen = self.clone_score_seen_ema[anchor_indices]
        score = self.clone_score_ema[anchor_indices] / (seen + float(eps))
        score = torch.where(seen > float(eps), score, torch.ones_like(score))
        score = torch.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)
        return torch.clamp_min(score.to(device=coords.device, dtype=coords.dtype), 0.0)

    @staticmethod
    def _clone_score_stage(training, iteration, warmup_until, ramp_until, freeze_after):
        if iteration is None:
            return "fixed", 2.0, False
        iteration = int(iteration)
        warmup_until = int(warmup_until or 0)
        ramp_until = int(ramp_until or 0)
        freeze_after = int(freeze_after or 0)
        if warmup_until > 0 and iteration <= warmup_until:
            return "warmup", 0.0, False
        if ramp_until > warmup_until and iteration <= ramp_until:
            return "ramp", 1.0, False
        if freeze_after > 0 and iteration > freeze_after:
            return "freeze", 3.0, True
        return "fixed", 2.0, False

    @staticmethod
    def _clone_stage_detail_budget(
            visible_count,
            max_detail_slots,
            render_gaussian_budget,
            stage_name,
            iteration,
            warmup_until,
            ramp_until):
        full_detail_budget = int(visible_count) * int(max_detail_slots)
        render_gaussian_budget = int(render_gaussian_budget or 0)
        if render_gaussian_budget <= 0:
            target_detail_budget = full_detail_budget
        else:
            target_detail_budget = max(render_gaussian_budget - int(visible_count), 0)
            target_detail_budget = min(target_detail_budget, full_detail_budget)

        if stage_name == "warmup":
            detail_budget = full_detail_budget
            ramp_progress = 0.0
        elif stage_name == "ramp":
            warmup_until = int(warmup_until or 0)
            ramp_until = int(ramp_until or 0)
            denom = max(1, ramp_until - warmup_until)
            ramp_progress = max(0.0, min(1.0, (int(iteration) - warmup_until) / float(denom)))
            detail_budget = int(round(full_detail_budget + (target_detail_budget - full_detail_budget) * ramp_progress))
        else:
            detail_budget = target_detail_budget
            ramp_progress = 1.0
        detail_budget = max(0, min(int(detail_budget), full_detail_budget))
        return target_detail_budget, detail_budget, int(visible_count) + detail_budget, ramp_progress

    @staticmethod
    def _clone_score_update_default_stats(frozen=False):
        return {
            "clone_score_updated_count": 0,
            "clone_score_signal_mean": 0.0,
            "clone_score_signal_max": 0.0,
            "clone_score_frozen": int(bool(frozen)),
        }

    def update_clone_scores(
            self,
            render_pkg,
            decoded_data,
            ema_beta=0.95,
            eps=1e-6,
            detail_grad_weight=1.0,
            grad_clip=0.0,
            frozen=False):
        if frozen:
            return self._clone_score_update_default_stats(frozen=True)
        if not self._clone_score_used:
            return self._clone_score_update_default_stats(frozen=False)
        viewspace_points = render_pkg.get("viewspace_points") if render_pkg is not None else None
        if viewspace_points is None or viewspace_points.grad is None:
            return self._clone_score_update_default_stats(frozen=False)
        anchor_indices = decoded_data.get("anchor_indices") if decoded_data is not None else None
        if anchor_indices is None or anchor_indices.numel() == 0:
            return self._clone_score_update_default_stats(frozen=False)

        anchor_indices = anchor_indices.detach().to(device=viewspace_points.device, dtype=torch.long)
        visible_count = int(anchor_indices.numel())
        detail_anchor_idx = decoded_data.get("detail_anchor_idx")
        detail_count = int(detail_anchor_idx.numel()) if detail_anchor_idx is not None else 0
        grad_norm = viewspace_points.grad.detach().norm(dim=-1).to(torch.float32)
        if grad_norm.numel() < detail_count + visible_count:
            return self._clone_score_update_default_stats(frozen=False)

        base_signal = grad_norm[detail_count:detail_count + visible_count]
        signal = base_signal.clone()
        if detail_count > 0 and float(detail_grad_weight or 0.0) != 0.0:
            detail_anchor_idx = detail_anchor_idx.detach().to(device=grad_norm.device, dtype=torch.long)
            valid = (detail_anchor_idx >= 0) & (detail_anchor_idx < visible_count)
            if torch.any(valid):
                local_idx = detail_anchor_idx[valid]
                detail_signal = grad_norm[:detail_count][valid] * float(detail_grad_weight)
                accum = torch.zeros(visible_count, dtype=torch.float32, device=grad_norm.device)
                counts = torch.zeros(visible_count, dtype=torch.float32, device=grad_norm.device)
                accum.index_add_(0, local_idx, detail_signal)
                counts.index_add_(0, local_idx, torch.ones_like(detail_signal))
                detail_mean = accum / counts.clamp_min(1.0)
                signal = torch.maximum(signal, detail_mean)

        signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        if float(grad_clip or 0.0) > 0.0:
            signal = signal.clamp_max(float(grad_clip))
        self._ensure_clone_score_buffers(anchor_indices)
        ema_beta = max(0.0, min(0.9999, float(ema_beta)))
        old_score = self.clone_score_ema[anchor_indices]
        old_seen = self.clone_score_seen_ema[anchor_indices]
        self.clone_score_ema[anchor_indices] = old_score * ema_beta + signal.to(old_score.dtype) * (1.0 - ema_beta)
        self.clone_score_seen_ema[anchor_indices] = old_seen * ema_beta + (1.0 - ema_beta)
        return {
            "clone_score_updated_count": int(visible_count),
            "clone_score_signal_mean": float(signal.mean().item()) if signal.numel() > 0 else 0.0,
            "clone_score_signal_max": float(signal.max().item()) if signal.numel() > 0 else 0.0,
            "clone_score_frozen": 0,
        }

    def _empty_decoded_data(self, coords, detail_counts, budget_stats, gate_losses=None):
        decoded = {
            "d_color": coords.new_empty((0, 3)),
            "d_rotation": coords.new_empty((0, 4)),
            "d_scaling": coords.new_empty((0, 3)),
            "d_opacity": coords.new_empty((0, 1)),
            "detail_anchor_idx": torch.empty(0, dtype=torch.long, device=coords.device),
            "detail_slot_idx": torch.empty(0, dtype=torch.long, device=coords.device),
            "detail_counts": detail_counts,
            "budget_stats": budget_stats,
        }
        if gate_losses is not None:
            decoded["gate_losses"] = gate_losses
        return decoded

    def _zero_gate_losses(self, coords):
        zero = coords.new_zeros(())
        return {"loss": zero, "budget": zero, "binary": zero}

    def _match_gate_extra_dim(self, extra):
        if extra.shape[-1] == self.gate_extra_dim:
            return extra
        if extra.shape[-1] > self.gate_extra_dim:
            return extra[:, :self.gate_extra_dim]
        pad = extra.new_zeros((extra.shape[0], self.gate_extra_dim - extra.shape[-1]))
        return torch.cat([extra, pad], dim=-1)

    def _view_context_feature(self, coords, scale_input, camera_center):
        with torch.no_grad():
            anchor_hash = self.encoder.encode_xyz(coords.detach()).detach()
        feature_count = anchor_hash.shape[0]
        if camera_center is not None:
            if not isinstance(camera_center, torch.Tensor):
                camera_center = coords.new_tensor(camera_center)
            camera_center = camera_center.to(device=coords.device, dtype=coords.dtype)
            while camera_center.dim() > 1:
                camera_center = camera_center[0]
            delta = camera_center.unsqueeze(0) - coords.detach()
            view_dir = F.normalize(delta, dim=-1, eps=1e-6).detach()
            distance = torch.log1p(delta.norm(dim=-1, keepdim=True)).detach()
            distance_scale = distance.detach().max().clamp_min(1.0)
            distance = distance / distance_scale
        else:
            view_dir = anchor_hash.new_zeros((feature_count, 3))
            distance = anchor_hash.new_zeros((feature_count, 1))
        scale_mean = scale_input.detach().to(device=coords.device, dtype=coords.dtype).mean(dim=-1, keepdim=True)
        return self.view_context_encoder(anchor_hash, view_dir, distance, scale_mean)

    @staticmethod
    def _gate_all_detail_warmup_active(training, iteration, all_detail_until):
        if not training:
            return False
        all_detail_until = int(all_detail_until or 0)
        if all_detail_until <= 0 or iteration is None:
            return False
        return int(iteration) <= all_detail_until

    def _gate_input_feature(
            self,
            detail_spatial_h,
            candidate_xyz,
            camera_center,
            gate_feature_mode,
            candidate_offset=None,
            candidate_slot_idx=None,
            slot_count=None,
            view_context=None,
    ):
        mode = str(gate_feature_mode or "detail_view").lower()
        spatial_feature = detail_spatial_h.detach()
        feature_count = spatial_feature.shape[0]
        if mode in ("hash_view_context", "view_context"):
            if view_context is None:
                extra_feature = spatial_feature.new_zeros((feature_count, self.gate_extra_dim))
            else:
                view_context = view_context.to(device=spatial_feature.device, dtype=spatial_feature.dtype)
                if view_context.dim() == 1:
                    extra_feature = view_context.unsqueeze(0).expand(feature_count, -1)
                else:
                    extra_feature = view_context.expand(feature_count, -1)
            extra_feature = self._match_gate_extra_dim(extra_feature)
            return torch.cat([spatial_feature, extra_feature], dim=-1)

        use_view = mode in (
            "detail_view",
            "detail+view",
            "utility",
            "utility_detail",
            "detail_view_offset",
        )
        use_offset = mode in (
            "utility",
            "utility_detail",
            "detail_offset",
            "detail_view_offset",
        )
        if mode not in (
                "detail_only",
                "detail",
                "detail_view",
                "detail+view",
                "utility",
                "utility_detail",
                "detail_offset",
                "detail_view_offset",
        ):
            raise ValueError(f"Unsupported gate_feature_mode: {gate_feature_mode}")

        if use_view and camera_center is not None:
            if not isinstance(camera_center, torch.Tensor):
                camera_center = spatial_feature.new_tensor(camera_center)
            camera_center = camera_center.to(device=spatial_feature.device, dtype=spatial_feature.dtype)
            while camera_center.dim() > 1:
                camera_center = camera_center[0]
            view_dir = F.normalize(camera_center.unsqueeze(0) - candidate_xyz.detach(), dim=-1, eps=1e-6).detach()
        else:
            view_dir = spatial_feature.new_zeros((feature_count, 3))

        if use_offset and candidate_offset is not None:
            offset_feature = candidate_offset.to(device=spatial_feature.device, dtype=spatial_feature.dtype).detach()
            offset_norm = offset_feature.norm(dim=-1, keepdim=True)
        else:
            offset_feature = spatial_feature.new_zeros((feature_count, 3))
            offset_norm = spatial_feature.new_zeros((feature_count, 1))

        if use_offset and candidate_slot_idx is not None:
            slot_feature = candidate_slot_idx.to(device=spatial_feature.device, dtype=spatial_feature.dtype).view(-1, 1)
            slot_denominator = max(1, int(slot_count or 1) - 1)
            slot_feature = (slot_feature / float(slot_denominator)).detach()
        else:
            slot_feature = spatial_feature.new_zeros((feature_count, 1))

        extra_feature = torch.cat([view_dir, offset_feature, offset_norm, slot_feature], dim=-1)
        extra_feature = self._match_gate_extra_dim(extra_feature)
        return torch.cat([spatial_feature, extra_feature], dim=-1)

    @staticmethod
    def _detail_feature_mode(detail_feature_mode):
        mode = str(detail_feature_mode or "detail_hash").lower()
        if mode in (
                "detail_hash",
                "hash_detail",
                "hash_detail_xyz",
                "detail_xyz_hash",
                "hash_anchor_xyz_offset",
        ):
            return "detail_hash"
        if mode in (
                "anchor_slot",
                "anchor_h_slot",
                "anchor_hidden_slot",
                "anchor_feature_slot",
                "anchor_slot_embedding",
                "anchor_h+slot_embedding",
        ):
            return "anchor_slot"
        raise ValueError(f"Unsupported detail_feature_mode: {detail_feature_mode}")

    def _decode_detail_hidden(
            self,
            coords,
            temporal_h,
            scale_input,
            rotate_input,
            anchor_idx,
            slot_idx,
            detail_xyz=None,
            detail_spatial_h=None,
            detail_feature_mode="detail_hash",
    ):
        mode = self._detail_feature_mode(detail_feature_mode)
        if mode == "anchor_slot":
            anchor_spatial_h = self.encoder.encode_xyz(coords.detach())
            anchor_h = self.decoder.compute_hidden(
                anchor_spatial_h,
                temporal_h,
                scale_input,
                rotate_input,
            )
            return anchor_h[anchor_idx]

        if detail_spatial_h is None:
            if detail_xyz is None:
                raise ValueError("detail_xyz is required for detail_hash mode.")
            detail_spatial_h = self.encoder.encode_xyz(detail_xyz.detach())
        return self.decoder.compute_hidden(
            detail_spatial_h,
            temporal_h[anchor_idx],
            scale_input[anchor_idx],
            rotate_input[anchor_idx],
        )

    @staticmethod
    def _projected_area_detail_stage_active(iteration, all_detail_until):
        all_detail_until = int(all_detail_until or 0)
        if all_detail_until <= 0 or iteration is None:
            return False
        return int(iteration) < all_detail_until

    @staticmethod
    def _projected_area_score_stats(scores):
        if scores.numel() == 0:
            return {
                "projected_area_score_mean": 0.0,
                "projected_area_score_max": 0.0,
                "projected_area_score_min": 0.0,
            }
        detached = scores.detach()
        return {
            "projected_area_score_mean": float(detached.mean().item()),
            "projected_area_score_max": float(detached.max().item()),
            "projected_area_score_min": float(detached.min().item()),
        }

    @staticmethod
    def _projected_complexity_score_stats(scores):
        if scores.numel() == 0:
            return {
                "projected_complexity_score_mean": 0.0,
                "projected_complexity_score_max": 0.0,
                "projected_complexity_score_min": 0.0,
            }
        detached = scores.detach()
        return {
            "projected_complexity_score_mean": float(detached.mean().item()),
            "projected_complexity_score_max": float(detached.max().item()),
            "projected_complexity_score_min": float(detached.min().item()),
        }

    def _projected_score_stats_for_mode(self, scores, score_mode):
        stats = self._projected_area_score_stats(scores)
        if str(score_mode or "area").lower() in ("complexity", "projected_complexity"):
            stats.update(self._projected_complexity_score_stats(scores))
        return stats

    def _projected_area_scores(
            self,
            coords,
            scale_input,
            camera_center=None,
            scale_power=2.0,
            distance_power=2.0,
            eps=1e-6,
            use_fast_score=True,
    ):
        eps = max(float(eps or 1e-6), 1e-12)
        scale_power = float(scale_power or 0.0)
        distance_power = float(distance_power or 0.0)
        scale = scale_input.detach().to(device=coords.device, dtype=coords.dtype)
        scale = torch.clamp_min(torch.abs(scale[:, :3]).mean(dim=-1), eps)
        if not use_fast_score:
            score = torch.pow(scale, scale_power)
            if camera_center is not None:
                if not isinstance(camera_center, torch.Tensor):
                    camera_center = coords.new_tensor(camera_center)
                camera_center = camera_center.to(device=coords.device, dtype=coords.dtype)
                while camera_center.dim() > 1:
                    camera_center = camera_center[0]
                distance = (coords.detach() - camera_center.unsqueeze(0)).norm(dim=-1).clamp_min(eps)
                score = score / torch.pow(distance, distance_power)
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.clamp_min(score, 0.0)

        if scale_power == 2.0:
            score = scale * scale
        elif scale_power == 1.0:
            score = scale
        elif scale_power == 0.0:
            score = torch.ones_like(scale)
        else:
            score = torch.pow(scale, scale_power)
        if camera_center is not None:
            if not isinstance(camera_center, torch.Tensor):
                camera_center = coords.new_tensor(camera_center)
            camera_center = camera_center.to(device=coords.device, dtype=coords.dtype)
            while camera_center.dim() > 1:
                camera_center = camera_center[0]
            delta = coords.detach() - camera_center.unsqueeze(0)
            if distance_power == 2.0:
                distance_term = torch.clamp_min((delta * delta).sum(dim=-1), eps * eps)
            elif distance_power == 1.0:
                distance_term = delta.norm(dim=-1).clamp_min(eps)
            elif distance_power == 0.0:
                distance_term = None
            else:
                distance = delta.norm(dim=-1).clamp_min(eps)
                distance_term = torch.pow(distance, distance_power)
            if distance_term is not None:
                score = score / distance_term
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp_min(score, 0.0)

    @staticmethod
    def _viewpoint_int(viewpoint_camera, key, default=0):
        if viewpoint_camera is None or key not in viewpoint_camera:
            return int(default)
        value = viewpoint_camera[key]
        if isinstance(value, torch.Tensor):
            return int(value.detach().item())
        return int(value)

    @staticmethod
    def _viewpoint_tensor(viewpoint_camera, key, device, dtype):
        if viewpoint_camera is None or key not in viewpoint_camera:
            return None
        value = viewpoint_camera[key]
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _projected_complexity_scores(
            self,
            coords,
            scale_input,
            color_dc,
            opacity_input,
            viewpoint_camera=None,
            camera_center=None,
            scale_power=2.0,
            distance_power=2.0,
            eps=1e-6,
            tile_size=32,
            color_weight=0.40,
            depth_weight=0.30,
            depth_var_weight=0.20,
            shape_weight=0.10,
            area_tau=0.0,
            large_area_tau=0.0,
            opacity_power=1.0,
            use_fast_score=True,
    ):
        with torch.no_grad():
            fallback_scores = self._projected_area_scores(
                coords,
                scale_input,
                camera_center=camera_center,
                scale_power=scale_power,
                distance_power=distance_power,
                eps=eps,
                use_fast_score=use_fast_score,
            ).detach().to(dtype=torch.float32)
            visible_count = coords.shape[0]
            if visible_count == 0 or color_dc is None or viewpoint_camera is None:
                return fallback_scores.to(dtype=coords.dtype)

            device = coords.device
            score_dtype = torch.float32
            eps = max(float(eps or 1e-6), 1e-12)
            tile_size = max(1, int(tile_size or 32))
            image_height = self._viewpoint_int(viewpoint_camera, "image_height", 0)
            image_width = self._viewpoint_int(viewpoint_camera, "image_width", 0)
            full_proj = self._viewpoint_tensor(viewpoint_camera, "full_proj_transform", device, score_dtype)
            if image_height <= 0 or image_width <= 0 or full_proj is None:
                return fallback_scores.to(dtype=coords.dtype)

            coords_f = coords.detach().to(device=device, dtype=score_dtype)
            ones = torch.ones((visible_count, 1), device=device, dtype=score_dtype)
            hom = torch.cat([coords_f, ones], dim=-1)
            clip = hom @ full_proj
            w = clip[:, 3]
            ndc = clip[:, :3] / w.clamp_min(eps).unsqueeze(-1)
            ndc = torch.nan_to_num(ndc, nan=0.0, posinf=0.0, neginf=0.0)
            valid_idx = torch.arange(visible_count, device=device)

            pixel_x = torch.floor((ndc[:, 0] + 1.0) * 0.5 * float(image_width)).to(dtype=torch.long)
            pixel_y = torch.floor((1.0 - ndc[:, 1]) * 0.5 * float(image_height)).to(dtype=torch.long)
            pixel_x = pixel_x.clamp_(0, image_width - 1)
            pixel_y = pixel_y.clamp_(0, image_height - 1)
            tile_w = max(1, (image_width + tile_size - 1) // tile_size)
            tile_h = max(1, (image_height + tile_size - 1) // tile_size)
            tile_count = tile_w * tile_h
            tile_idx = (pixel_y // tile_size) * tile_w + (pixel_x // tile_size)
            tile_idx_v = tile_idx[valid_idx].clamp_(0, tile_count - 1)

            color = color_dc.detach().to(device=device, dtype=score_dtype)
            if color.dim() == 3:
                color = color[:, 0, :]
            if color.shape[0] != visible_count:
                return fallback_scores.to(dtype=coords.dtype)
            color = torch.clamp(color + 0.5, 0.0, 1.0)
            color_v = color[valid_idx]

            if camera_center is None and viewpoint_camera is not None:
                camera_center = viewpoint_camera.get("camera_center")
            if camera_center is not None:
                if not isinstance(camera_center, torch.Tensor):
                    camera_center = coords_f.new_tensor(camera_center)
                camera_center = camera_center.detach().to(device=device, dtype=score_dtype)
                while camera_center.dim() > 1:
                    camera_center = camera_center[0]
                depth = (coords_f - camera_center.unsqueeze(0)).norm(dim=-1)
            else:
                view = self._viewpoint_tensor(viewpoint_camera, "world_view_transform", device, score_dtype)
                if view is not None:
                    depth = torch.abs((hom @ view)[:, 2])
                else:
                    depth = torch.abs(ndc[:, 2])
            depth_v = depth[valid_idx].clamp_min(eps)

            scale_values = torch.clamp_min(
                torch.abs(scale_input.detach().to(device=device, dtype=score_dtype)[:, :3]).mean(dim=-1),
                eps,
            )
            scale_v = scale_values[valid_idx]

            counts = torch.zeros(tile_count, device=device, dtype=score_dtype)
            ones_v = torch.ones_like(depth_v)
            counts.index_add_(0, tile_idx_v, ones_v)
            safe_counts = counts.clamp_min(1.0)

            sum_color = torch.zeros((tile_count, 3), device=device, dtype=score_dtype)
            sum_color.index_add_(0, tile_idx_v, color_v)
            mean_color = sum_color[tile_idx_v] / safe_counts[tile_idx_v].unsqueeze(-1)
            color_dev = torch.linalg.vector_norm(color_v - mean_color, dim=-1) / 1.7320508075688772

            sum_depth = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_depth2 = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_depth.index_add_(0, tile_idx_v, depth_v)
            sum_depth2.index_add_(0, tile_idx_v, depth_v * depth_v)
            mean_depth_all = sum_depth / safe_counts
            var_depth_all = torch.clamp_min(sum_depth2 / safe_counts - mean_depth_all * mean_depth_all, 0.0)
            depth_scale = depth_v.mean().clamp_min(eps)
            depth_dev = torch.abs(depth_v - mean_depth_all[tile_idx_v]) / depth_scale
            depth_var = torch.sqrt(var_depth_all[tile_idx_v]) / depth_scale

            sum_scale = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_scale2 = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_scale.index_add_(0, tile_idx_v, scale_v)
            sum_scale2.index_add_(0, tile_idx_v, scale_v * scale_v)
            mean_scale_all = sum_scale / safe_counts
            var_scale_all = torch.clamp_min(sum_scale2 / safe_counts - mean_scale_all * mean_scale_all, 0.0)
            scale_norm = scale_v.mean().clamp_min(eps)
            shape_dev = 0.5 * torch.abs(scale_v - mean_scale_all[tile_idx_v]) / scale_norm
            shape_dev = shape_dev + 0.5 * torch.sqrt(var_scale_all[tile_idx_v]) / scale_norm

            complexity = (
                float(color_weight or 0.0) * color_dev
                + float(depth_weight or 0.0) * depth_dev
                + float(depth_var_weight or 0.0) * depth_var
                + float(shape_weight or 0.0) * shape_dev
            )
            complexity = torch.nan_to_num(complexity, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

            area = fallback_scores.clamp_min(0.0)
            area_values = area[valid_idx]
            tau_value = float(area_tau or 0.0)
            if tau_value <= 0.0:
                tau = area_values.mean().clamp_min(eps)
            else:
                tau = area_values.new_tensor(tau_value).clamp_min(eps)
            large_tau_value = float(large_area_tau or 0.0)
            if large_tau_value <= 0.0:
                large_tau = (tau * 4.0).clamp_min(eps)
            else:
                large_tau = area_values.new_tensor(large_tau_value).clamp_min(eps)
            saturated_area = 1.0 - torch.exp(-area_values / tau)
            large_area_penalty = torch.rsqrt(1.0 + area_values / large_tau)

            if opacity_input is not None:
                opacity = opacity_input.detach().to(device=device, dtype=score_dtype).reshape(-1)
                if opacity.shape[0] == visible_count:
                    opacity = opacity[valid_idx].clamp(0.0, 1.0)
                else:
                    opacity = torch.ones_like(area_values)
            else:
                opacity = torch.ones_like(area_values)
            opacity_power = float(opacity_power or 0.0)
            if opacity_power == 0.0:
                opacity_weight = torch.ones_like(opacity)
            elif opacity_power == 1.0:
                opacity_weight = opacity
            else:
                opacity_weight = torch.pow(opacity.clamp_min(eps), opacity_power)

            score_v = opacity_weight * saturated_area * large_area_penalty * complexity
            scores = torch.zeros_like(fallback_scores)
            scores[valid_idx] = torch.nan_to_num(score_v, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            scores = torch.where(scores.sum() > eps, scores, fallback_scores)
            return scores.to(dtype=coords.dtype)


    @staticmethod
    def _normalize_positive_unit(values, eps=1e-6, scale_multiplier=4.0):
        values = torch.nan_to_num(values.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if values.numel() == 0:
            return values
        scale = (values.mean().clamp_min(float(eps)) * float(scale_multiplier)).clamp_min(float(eps))
        return torch.clamp(values / scale, 0.0, 1.0)

    def _projected_inverse_area_visibility_scores(
            self,
            coords,
            scale_input,
            color_dc,
            opacity_input,
            viewpoint_camera=None,
            camera_center=None,
            scale_power=2.0,
            distance_power=2.0,
            eps=1e-6,
            tile_size=4,
            use_fast_score=True,
    ):
        with torch.no_grad():
            fallback_scores = self._projected_area_scores(
                coords,
                scale_input,
                camera_center=camera_center,
                scale_power=scale_power,
                distance_power=distance_power,
                eps=eps,
                use_fast_score=use_fast_score,
            ).detach().to(dtype=torch.float32)
            visible_count = coords.shape[0]
            if visible_count == 0:
                return fallback_scores.to(dtype=coords.dtype)

            device = coords.device
            score_dtype = torch.float32
            eps = max(float(eps or 1e-6), 1e-12)
            tile_size = max(1, int(tile_size or 4))
            image_height = self._viewpoint_int(viewpoint_camera, "image_height", 0)
            image_width = self._viewpoint_int(viewpoint_camera, "image_width", 0)
            full_proj = self._viewpoint_tensor(viewpoint_camera, "full_proj_transform", device, score_dtype)
            if image_height <= 0 or image_width <= 0 or full_proj is None:
                area_score = 1.0 - torch.exp(-fallback_scores / fallback_scores.mean().clamp_min(eps))
                return torch.pow((1.0 - area_score).clamp(0.0, 1.0), 2.0).to(dtype=coords.dtype)

            coords_f = coords.detach().to(device=device, dtype=score_dtype)
            ones = torch.ones((visible_count, 1), device=device, dtype=score_dtype)
            hom = torch.cat([coords_f, ones], dim=-1)
            clip = hom @ full_proj
            w = clip[:, 3]
            ndc = clip[:, :3] / w.clamp_min(eps).unsqueeze(-1)
            ndc = torch.nan_to_num(ndc, nan=0.0, posinf=0.0, neginf=0.0)

            pixel_x = torch.floor((ndc[:, 0] + 1.0) * 0.5 * float(image_width)).to(dtype=torch.long)
            pixel_y = torch.floor((1.0 - ndc[:, 1]) * 0.5 * float(image_height)).to(dtype=torch.long)
            pixel_x = pixel_x.clamp_(0, image_width - 1)
            pixel_y = pixel_y.clamp_(0, image_height - 1)
            tile_w = max(1, (image_width + tile_size - 1) // tile_size)
            tile_h = max(1, (image_height + tile_size - 1) // tile_size)
            tile_count = tile_w * tile_h
            tile_idx = ((pixel_y // tile_size) * tile_w + (pixel_x // tile_size)).clamp_(0, tile_count - 1)

            if color_dc is not None:
                color = color_dc.detach().to(device=device, dtype=score_dtype)
                if color.dim() == 3:
                    color = color[:, 0, :]
                if color.shape[0] == visible_count:
                    color = torch.clamp(color + 0.5, 0.0, 1.0)
                else:
                    color = coords_f.new_zeros((visible_count, 3))
            else:
                color = coords_f.new_zeros((visible_count, 3))

            if opacity_input is not None:
                opacity = opacity_input.detach().to(device=device, dtype=score_dtype).reshape(-1)
                if opacity.shape[0] != visible_count:
                    opacity = torch.ones(visible_count, device=device, dtype=score_dtype)
                else:
                    opacity = opacity.clamp(0.0, 1.0)
            else:
                opacity = torch.ones(visible_count, device=device, dtype=score_dtype)

            if camera_center is None and viewpoint_camera is not None:
                camera_center = viewpoint_camera.get("camera_center")
            if camera_center is not None:
                if not isinstance(camera_center, torch.Tensor):
                    camera_center = coords_f.new_tensor(camera_center)
                camera_center = camera_center.detach().to(device=device, dtype=score_dtype)
                while camera_center.dim() > 1:
                    camera_center = camera_center[0]
                depth = (coords_f - camera_center.unsqueeze(0)).norm(dim=-1).clamp_min(eps)
            else:
                view = self._viewpoint_tensor(viewpoint_camera, "world_view_transform", device, score_dtype)
                if view is not None:
                    depth = torch.abs((hom @ view)[:, 2]).clamp_min(eps)
                else:
                    depth = torch.abs(ndc[:, 2]).clamp_min(eps)

            scale_values = torch.clamp_min(
                torch.abs(scale_input.detach().to(device=device, dtype=score_dtype)[:, :3]).mean(dim=-1),
                eps,
            )
            area_values = fallback_scores.clamp_min(0.0)

            counts = torch.zeros(tile_count, device=device, dtype=score_dtype)
            ones_v = torch.ones(visible_count, device=device, dtype=score_dtype)
            counts.index_add_(0, tile_idx, ones_v)
            safe_counts = counts.clamp_min(1.0)
            occupied = (counts > 0).to(dtype=score_dtype)
            mean_occupied_count = (counts.sum() / occupied.sum().clamp_min(1.0)).clamp_min(1.0)
            tile_population = counts[tile_idx]
            density_score = torch.clamp(tile_population / (mean_occupied_count * 3.0).clamp_min(1.0), 0.0, 1.0)
            high_population = torch.clamp(
                torch.log1p(tile_population) / torch.log1p((mean_occupied_count * 4.0).clamp_min(1.0)),
                0.0,
                1.0,
            )

            sum_depth = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_depth2 = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_depth.index_add_(0, tile_idx, depth)
            sum_depth2.index_add_(0, tile_idx, depth * depth)
            mean_depth = sum_depth / safe_counts
            var_depth = torch.clamp_min(sum_depth2 / safe_counts - mean_depth * mean_depth, 0.0)
            depth_std = torch.sqrt(var_depth[tile_idx]) / depth.mean().clamp_min(eps)

            sum_scale = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_scale2 = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_scale.index_add_(0, tile_idx, scale_values)
            sum_scale2.index_add_(0, tile_idx, scale_values * scale_values)
            mean_scale = sum_scale / safe_counts
            var_scale = torch.clamp_min(sum_scale2 / safe_counts - mean_scale * mean_scale, 0.0)
            scale_std = torch.sqrt(var_scale[tile_idx]) / scale_values.mean().clamp_min(eps)

            sum_area = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_area2 = torch.zeros(tile_count, device=device, dtype=score_dtype)
            sum_area.index_add_(0, tile_idx, area_values)
            sum_area2.index_add_(0, tile_idx, area_values * area_values)
            mean_area = sum_area / safe_counts
            var_area = torch.clamp_min(sum_area2 / safe_counts - mean_area * mean_area, 0.0)
            area_std = torch.sqrt(var_area[tile_idx]) / area_values.mean().clamp_min(eps)

            sum_color = torch.zeros((tile_count, 3), device=device, dtype=score_dtype)
            sum_color2 = torch.zeros((tile_count, 3), device=device, dtype=score_dtype)
            sum_color.index_add_(0, tile_idx, color)
            sum_color2.index_add_(0, tile_idx, color * color)
            mean_color = sum_color / safe_counts.unsqueeze(-1)
            color_var = torch.clamp_min(sum_color2 / safe_counts.unsqueeze(-1) - mean_color * mean_color, 0.0)
            color_std = torch.sqrt(color_var[tile_idx].mean(dim=-1).clamp_min(0.0))

            attribute_transition = self._normalize_positive_unit(
                0.35 * depth_std + 0.20 * scale_std + 0.20 * area_std + 0.25 * color_std,
                eps=eps,
                scale_multiplier=4.0,
            )
            low_transition = (1.0 - attribute_transition).clamp(0.0, 1.0)
            large_uniform = (high_population * low_transition).clamp(0.0, 1.0)
            flat_score = (large_uniform * low_transition * density_score).clamp(0.0, 1.0)

            area_tau = area_values.mean().clamp_min(eps)
            area_score = 1.0 - torch.exp(-area_values / area_tau)
            inverse_area = torch.pow((1.0 - area_score).clamp(0.0, 1.0), 2.0)

            visibility_gate = torch.clamp(tile_population, 0.0, 1.0) * torch.clamp(opacity / 0.05, 0.0, 1.0)
            mid_density = torch.exp(-torch.square((density_score - 0.38) / 0.34))
            not_overdense = (1.0 - torch.clamp((density_score - 0.78) / 0.22, 0.0, 1.0)).clamp(0.0, 1.0)
            recoverable_gate = (visibility_gate * (0.30 + 0.70 * mid_density) * (0.35 + 0.65 * not_overdense)).clamp(0.0, 1.0)
            non_uniform_gate = (0.20 + 0.80 * attribute_transition).clamp(0.0, 1.0)
            anti_flat = torch.pow((1.0 - flat_score).clamp(0.0, 1.0), 1.5)
            anti_large = torch.pow((1.0 - large_uniform).clamp(0.0, 1.0), 1.5)

            scores = inverse_area * recoverable_gate * anti_flat * anti_large * non_uniform_gate
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            fallback_inverse = inverse_area * visibility_gate
            scores = torch.where(scores.sum() > eps, scores, fallback_inverse)
            return scores.to(dtype=coords.dtype)

    def _step_projected_area(
            self,
            coords,
            scale_input,
            rotate_input,
            offset_slots,
            pose,
            render_gaussian_budget=0,
            iteration=None,
            camera_center=None,
            viewpoint_camera=None,
            projected_area_all_detail_until=50000,
            projected_area_detail_stage_full_budget=True,
            projected_area_scale_power=2.0,
            projected_area_distance_power=2.0,
            projected_area_eps=1e-6,
            projected_score_mode="area",
            color_input=None,
            opacity_input=None,
            projected_complexity_tile_size=32,
            projected_complexity_color_weight=0.40,
            projected_complexity_depth_weight=0.30,
            projected_complexity_depth_var_weight=0.20,
            projected_complexity_shape_weight=0.10,
            projected_complexity_area_tau=0.0,
            projected_complexity_large_area_tau=0.0,
            projected_complexity_opacity_power=1.0,
            projected_inverse_area_tile_size=4,
            detail_feature_mode="anchor_slot",
            include_projected_area_stats=True,
            stage_timer=None,
            use_projected_area_optimizations=False,
    ):
        projected_score_mode = str(projected_score_mode or "area").lower()
        complexity_mode = projected_score_mode in ("complexity", "projected_complexity")
        inverse_area_mode = projected_score_mode in (
            "projection_inverse_area_with_visibility_gate",
            "projected_inverse_area",
            "inverse_projected_area",
            "inverse_area_visibility",
        )
        allocation_mode_id = 6.0 if inverse_area_mode else (5.0 if complexity_mode else 4.0)
        visible_count = coords.shape[0]
        render_gaussian_budget = int(render_gaussian_budget or 0)
        slot_count = min(offset_slots.shape[1], self.max_detail_slots)
        if slot_count <= 0 or visible_count == 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": 0,
                "selected_detail_count": 0,
                "budget_overflow": visible_count > render_gaussian_budget > 0,
                "allocation_mode_id": allocation_mode_id,
                "projected_area_stage_id": 1.0,
                "projected_area_all_detail_active": 0.0,
                "projected_area_target_detail_budget": 0,
                "projected_area_effective_detail_budget": 0,
            }
            budget_stats.update(self._projected_score_stats_for_mode(coords.new_empty(0), projected_score_mode))
            return self._empty_decoded_data(coords, detail_counts, budget_stats)

        full_detail_budget = visible_count * slot_count
        target_detail_budget = full_detail_budget if render_gaussian_budget <= 0 else max(render_gaussian_budget - visible_count, 0)
        target_detail_budget = min(target_detail_budget, full_detail_budget)
        detail_stage_active = self._projected_area_detail_stage_active(iteration, projected_area_all_detail_until)
        # Stage controls optimizer scope only; allocation always follows the YAML budget.
        if render_gaussian_budget <= 0:
            effective_render_budget = visible_count + full_detail_budget
        else:
            effective_render_budget = render_gaussian_budget

        if stage_timer is not None:
            stage_timer.start("importance_estimation")
        if inverse_area_mode:
            projected_scores = self._projected_inverse_area_visibility_scores(
                coords,
                scale_input,
                color_input,
                opacity_input,
                viewpoint_camera=viewpoint_camera,
                camera_center=camera_center,
                scale_power=projected_area_scale_power,
                distance_power=projected_area_distance_power,
                eps=projected_area_eps,
                tile_size=projected_inverse_area_tile_size,
                use_fast_score=use_projected_area_optimizations,
            )
        elif complexity_mode:
            projected_scores = self._projected_complexity_scores(
                coords,
                scale_input,
                color_input,
                opacity_input,
                viewpoint_camera=viewpoint_camera,
                camera_center=camera_center,
                scale_power=projected_area_scale_power,
                distance_power=projected_area_distance_power,
                eps=projected_area_eps,
                tile_size=projected_complexity_tile_size,
                color_weight=projected_complexity_color_weight,
                depth_weight=projected_complexity_depth_weight,
                depth_var_weight=projected_complexity_depth_var_weight,
                shape_weight=projected_complexity_shape_weight,
                area_tau=projected_complexity_area_tau,
                large_area_tau=projected_complexity_large_area_tau,
                opacity_power=projected_complexity_opacity_power,
                use_fast_score=use_projected_area_optimizations,
            )
        else:
            projected_scores = self._projected_area_scores(
                coords,
                scale_input,
                camera_center=camera_center,
                scale_power=projected_area_scale_power,
                distance_power=projected_area_distance_power,
                eps=projected_area_eps,
                use_fast_score=use_projected_area_optimizations,
            )
        if stage_timer is not None:
            stage_timer.stop("importance_estimation")

        if stage_timer is not None:
            stage_timer.start("allocation")
        if use_projected_area_optimizations:
            detail_counts, budget_stats = self._allocate_projected_area_counts(projected_scores, effective_render_budget)
        else:
            detail_counts, budget_stats = self._allocate_detail_counts(projected_scores, effective_render_budget)
        budget_stats.update({
            "allocation_mode_id": allocation_mode_id,
            "projected_area_stage_id": 0.0 if detail_stage_active else 1.0,
            "projected_area_all_detail_active": float(bool(detail_stage_active)),
            "projected_area_target_detail_budget": int(target_detail_budget),
            "projected_area_effective_detail_budget": int(budget_stats.get("selected_detail_count", 0)),
        })

        anchor_idx, slot_idx = self._selected_indices_from_counts(detail_counts)
        if stage_timer is not None:
            stage_timer.stop("allocation")
        if anchor_idx.numel() == 0:
            if include_projected_area_stats:
                budget_stats.update(self._projected_score_stats_for_mode(projected_scores, projected_score_mode))
            decoded = self._empty_decoded_data(coords, detail_counts, budget_stats)
            if not include_projected_area_stats:
                decoded["_projected_area_scores"] = projected_scores
            return decoded

        if stage_timer is not None:
            stage_timer.start("decoding")
        detail_xyz = None
        if self._detail_feature_mode(detail_feature_mode) == "detail_hash":
            offset_slots = offset_slots[:, :slot_count]
            detail_xyz = coords[anchor_idx] + offset_slots[anchor_idx, slot_idx]
        if use_projected_area_optimizations:
            temporal_h = pose.unsqueeze(0).expand(coords.size()[0], -1)
        else:
            temporal_h = pose.unsqueeze(0).repeat(coords.size()[0], 1)
        detail_h = self._decode_detail_hidden(
            coords,
            temporal_h,
            scale_input,
            rotate_input,
            anchor_idx,
            slot_idx,
            detail_xyz=detail_xyz,
            detail_feature_mode=detail_feature_mode,
        )
        color, rotation, scaling, opacity = self.decoder.decode_hidden(detail_h, slot_idx)
        if stage_timer is not None:
            stage_timer.stop("decoding")

        if include_projected_area_stats:
            budget_stats.update(self._projected_score_stats_for_mode(projected_scores, projected_score_mode))
        decoded = {
            "d_color": color,
            "d_rotation": rotation,
            "d_scaling": scaling,
            "d_opacity": opacity,
            "detail_anchor_idx": anchor_idx,
            "detail_slot_idx": slot_idx,
            "detail_counts": detail_counts,
            "budget_stats": budget_stats,
        }
        if not include_projected_area_stats:
            decoded["_projected_area_scores"] = projected_scores
        return decoded

    def _step_clone_score(
            self,
            coords,
            scale_input,
            rotate_input,
            offset_slots,
            pose,
            render_gaussian_budget=0,
            training=False,
            iteration=None,
            clone_score_warmup_until=20000,
            clone_score_ramp_until=40000,
            clone_score_freeze_after=220000,
            clone_score_eps=1e-6,
            anchor_indices=None,
            detail_feature_mode="detail_hash"):
        self._clone_score_used = True
        temporal_h = pose.unsqueeze(0).repeat(coords.size()[0], 1)
        visible_count = coords.shape[0]
        render_gaussian_budget = int(render_gaussian_budget or 0)
        if anchor_indices is None:
            anchor_indices = torch.arange(visible_count, dtype=torch.long, device=coords.device)
        else:
            anchor_indices = anchor_indices.to(device=coords.device, dtype=torch.long)
        slot_count = min(offset_slots.shape[1], self.max_detail_slots)
        if slot_count <= 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": visible_count,
                "detail_budget": 0,
                "selected_detail_count": 0,
                "budget_overflow": visible_count > render_gaussian_budget > 0,
                "allocation_mode_id": 3.0,
                "clone_score_stage_id": 2.0,
                "clone_score_stage": "fixed",
                "clone_target_detail_budget": 0,
                "clone_effective_detail_budget": 0,
                "clone_target_render_gaussian_budget": render_gaussian_budget,
                "clone_effective_render_gaussian_budget": visible_count,
                "clone_score_mean": 0.0,
                "clone_score_max": 0.0,
                "clone_score_ramp_progress": 1.0,
                "clone_score_frozen": 0,
            }
            decoded = self._empty_decoded_data(coords, detail_counts, budget_stats)
            decoded["anchor_indices"] = anchor_indices
            decoded["clone_score_update_enabled"] = bool(training)
            return decoded

        stage_name, stage_id, frozen = self._clone_score_stage(
            training,
            iteration,
            clone_score_warmup_until,
            clone_score_ramp_until,
            clone_score_freeze_after,
        )
        target_detail_budget, effective_detail_budget, effective_render_budget, ramp_progress = self._clone_stage_detail_budget(
            visible_count,
            slot_count,
            render_gaussian_budget,
            stage_name,
            iteration if iteration is not None else 0,
            clone_score_warmup_until,
            clone_score_ramp_until,
        )
        visible_scores = self._clone_visible_scores(anchor_indices, coords, eps=clone_score_eps)
        allocation_scores = coords.new_ones((visible_count,)) if stage_name == "warmup" else visible_scores
        detail_counts, budget_stats = self._allocate_detail_counts(allocation_scores, effective_render_budget)
        budget_stats.update({
            "allocation_mode_id": 3.0,
            "clone_score_stage_id": stage_id,
            "clone_score_stage": stage_name,
            "clone_target_detail_budget": int(target_detail_budget),
            "clone_effective_detail_budget": int(effective_detail_budget),
            "clone_target_render_gaussian_budget": int(render_gaussian_budget),
            "clone_effective_render_gaussian_budget": int(effective_render_budget),
            "clone_score_mean": float(visible_scores.mean().item()) if visible_scores.numel() > 0 else 0.0,
            "clone_score_max": float(visible_scores.max().item()) if visible_scores.numel() > 0 else 0.0,
            "clone_score_ramp_progress": float(ramp_progress),
            "clone_score_frozen": int(bool(frozen)),
        })
        anchor_idx, slot_idx = self._selected_indices_from_counts(detail_counts)

        if anchor_idx.numel() > 0:
            offset_slots = offset_slots[:, :slot_count]
            detail_xyz = coords[anchor_idx] + offset_slots[anchor_idx, slot_idx]
            detail_h = self._decode_detail_hidden(
                coords,
                temporal_h,
                scale_input,
                rotate_input,
                anchor_idx,
                slot_idx,
                detail_xyz=detail_xyz,
                detail_feature_mode=detail_feature_mode,
            )
            color, rotation, scaling, opacity = self.decoder.decode_hidden(detail_h, slot_idx)
        else:
            decoded = self._empty_decoded_data(coords, detail_counts, budget_stats)
            decoded["anchor_indices"] = anchor_indices
            decoded["clone_score_update_enabled"] = bool(training and not frozen)
            return decoded

        return {
            "d_color": color,
            "d_rotation": rotation,
            "d_scaling": scaling,
            "d_opacity": opacity,
            "detail_anchor_idx": anchor_idx,
            "detail_slot_idx": slot_idx,
            "detail_counts": detail_counts,
            "anchor_indices": anchor_indices,
            "clone_score_update_enabled": bool(training and not frozen),
            "budget_stats": budget_stats,
        }

    def _step_proposal(
            self,
            coords,
            scale_input,
            rotate_input,
            offset_slots,
            pose,
            render_gaussian_budget=0,
            scale_min=0.0,
            detail_feature_mode="detail_hash",
    ):
        temporal_h = pose.unsqueeze(0).repeat(coords.size()[0], 1)
        render_gaussian_budget = int(render_gaussian_budget or 0)
        full_detail_budget = coords.shape[0] * self.max_detail_slots
        if render_gaussian_budget > 0 and max(render_gaussian_budget - coords.shape[0], 0) >= full_detail_budget:
            scores = coords.new_empty(coords.shape[0])
        elif render_gaussian_budget <= 0:
            scores = coords.new_ones(coords.shape[0])
        else:
            with torch.no_grad():
                spatial_h = self.encoder.encode_xyz(coords)
                h = self.decoder.compute_hidden(spatial_h, temporal_h, scale_input, rotate_input)
                scores = self.decoder.proposal_score(h, scale_min=scale_min)
            scores = scores.detach()
        detail_counts, budget_stats = self._allocate_detail_counts(scores, render_gaussian_budget)
        budget_stats["allocation_mode_id"] = 0.0
        anchor_idx, slot_idx = self._selected_indices_from_counts(detail_counts)

        if anchor_idx.numel() > 0:
            detail_xyz = coords[anchor_idx] + offset_slots[anchor_idx, slot_idx]
            detail_h = self._decode_detail_hidden(
                coords,
                temporal_h,
                scale_input,
                rotate_input,
                anchor_idx,
                slot_idx,
                detail_xyz=detail_xyz,
                detail_feature_mode=detail_feature_mode,
            )
            color, rotation, scaling, opacity = self.decoder.decode_hidden(detail_h, slot_idx)
        else:
            return self._empty_decoded_data(coords, detail_counts, budget_stats)

        return {
            "d_color": color,
            "d_rotation": rotation,
            "d_scaling": scaling,
            "d_opacity": opacity,
            "detail_anchor_idx": anchor_idx,
            "detail_slot_idx": slot_idx,
            "detail_counts": detail_counts,
            "budget_stats": budget_stats,
        }

    def _gate_budget_stats(
            self,
            render_gaussian_budget,
            visible_count,
            detail_budget,
            selected_count,
            budget_overflow,
            gate_stats=None,
            allocation_warmup_active=False,
    ):
        stats = {
            "render_gaussian_budget": int(render_gaussian_budget or 0),
            "visible_anchor_count": int(visible_count),
            "detail_budget": int(detail_budget),
            "selected_detail_count": int(selected_count),
            "budget_overflow": bool(budget_overflow),
            "allocation_mode_id": 1.0,
            "allocation_warmup_active": float(bool(allocation_warmup_active)),
        }
        if gate_stats:
            stats.update(gate_stats)
        return stats

    def _step_gate(
            self,
            coords,
            scale_input,
            rotate_input,
            offset_slots,
            pose,
            render_gaussian_budget=0,
            training=False,
            iteration=None,
            camera_center=None,
            gate_train_mode="soft_all",
            gate_eval_mode="topk",
            gate_temperature_init=1.0,
            gate_temperature_final=0.2,
            gate_temperature_max_steps=30000,
            gate_budget_lambda=0.01,
            gate_binary_lambda=0.001,
            gate_feature_mode="detail_view",
            gate_opacity_mode="st_identity",
            gate_all_detail_until=0,
            anchor_indices=None,
            detail_feature_mode="detail_hash",
    ):
        self._gate_used = True
        visible_count = coords.shape[0]
        slot_count = min(offset_slots.shape[1], self.max_detail_slots)
        render_gaussian_budget = int(render_gaussian_budget or 0)
        full_detail_budget = visible_count * slot_count
        budget_overflow = render_gaussian_budget > 0 and visible_count > render_gaussian_budget
        warmup_all_detail = self._gate_all_detail_warmup_active(
            training,
            iteration,
            gate_all_detail_until,
        )

        if visible_count == 0 or slot_count == 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = self._gate_budget_stats(
                render_gaussian_budget,
                visible_count,
                0,
                0,
                False,
                allocation_warmup_active=warmup_all_detail,
            )
            return self._empty_decoded_data(coords, detail_counts, budget_stats, self._zero_gate_losses(coords))

        if render_gaussian_budget <= 0 or warmup_all_detail:
            detail_budget = full_detail_budget
        else:
            detail_budget = max(render_gaussian_budget - visible_count, 0)
            detail_budget = min(detail_budget, full_detail_budget)

        if detail_budget <= 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = self._gate_budget_stats(
                render_gaussian_budget,
                visible_count,
                0,
                0,
                budget_overflow,
                allocation_warmup_active=warmup_all_detail,
            )
            return self._empty_decoded_data(coords, detail_counts, budget_stats, self._zero_gate_losses(coords))

        offset_slots = offset_slots[:, :slot_count]
        candidate_offset = offset_slots.reshape(-1, 3)
        candidate_slot_idx = torch.arange(slot_count, device=coords.device, dtype=torch.long)
        candidate_slot_idx = candidate_slot_idx.unsqueeze(0).expand(visible_count, -1).reshape(-1)
        candidate_xyz = coords[:, None, :] + offset_slots
        candidate_xyz = candidate_xyz.reshape(-1, 3)
        temporal_h = pose.unsqueeze(0).repeat(visible_count, 1)
        temporal_detail = temporal_h[:, None, :].expand(-1, slot_count, -1).reshape(-1, temporal_h.shape[-1])
        scale_detail = scale_input[:, None, :].expand(-1, slot_count, -1).reshape(-1, scale_input.shape[-1])
        rotate_detail = rotate_input[:, None, :].expand(-1, slot_count, -1).reshape(-1, rotate_input.shape[-1])

        detail_spatial_h = self.encoder.encode_xyz(candidate_xyz.detach())
        mode = str(gate_feature_mode or self.gate_feature_mode or "detail_view").lower()
        view_context = None
        if mode in ("hash_view_context", "view_context"):
            if not self.gate_uses_view_context or self.view_context_encoder is None:
                raise ValueError(
                    "MixGSModel must be constructed with gate_feature_mode='hash_view_context' "
                    "to use view context gate features."
                )
            view_context = self._view_context_feature(coords, scale_input, camera_center)
        gate_feature = self._gate_input_feature(
            detail_spatial_h,
            candidate_xyz,
            camera_center,
            mode,
            candidate_offset=candidate_offset,
            candidate_slot_idx=candidate_slot_idx,
            slot_count=slot_count,
            view_context=view_context,
        )
        if render_gaussian_budget <= 0 or warmup_all_detail:
            selected_idx = torch.arange(full_detail_budget, device=coords.device, dtype=torch.long)
            selected_gate = coords.new_ones(full_detail_budget)
            selected_logits = None
            gate_losses = self._zero_gate_losses(coords)
            gate_stats = {
                "gate_temperature": 0.0,
                "gate_mass": float(full_detail_budget),
                "gate_mean": 1.0,
                "gate_max": 1.0,
                "gate_min": 1.0,
                "gate_budget_loss": 0.0,
                "gate_binary_loss": 0.0,
            }
            if warmup_all_detail and training:
                gate_result = self.gate_allocator.select(
                    gate_feature,
                    full_detail_budget,
                    training=True,
                    train_mode="soft_all",
                    eval_mode=gate_eval_mode,
                    iteration=iteration,
                    temperature_init=gate_temperature_init,
                    temperature_final=gate_temperature_final,
                    temperature_max_steps=gate_temperature_max_steps,
                    budget_lambda=0.0,
                    binary_lambda=0.0,
                )
                selected_logits = gate_result["logits"]
                gate_losses = gate_result["losses"]
                gate_stats = gate_result["stats"]
        else:
            gate_result = self.gate_allocator.select(
                gate_feature,
                detail_budget,
                training=training,
                train_mode=gate_train_mode,
                eval_mode=gate_eval_mode,
                iteration=iteration,
                temperature_init=gate_temperature_init,
                temperature_final=gate_temperature_final,
                temperature_max_steps=gate_temperature_max_steps,
                budget_lambda=gate_budget_lambda,
                binary_lambda=gate_binary_lambda,
            )
            selected_idx = gate_result["selected_idx"]
            selected_gate = gate_result["selected_gate"]
            selected_logits = gate_result["logits"][selected_idx] if selected_idx.numel() > 0 else gate_result["logits"].new_empty(0)
            gate_losses = gate_result["losses"]
            gate_stats = gate_result["stats"]

        if selected_idx.numel() == 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = self._gate_budget_stats(
                render_gaussian_budget,
                visible_count,
                detail_budget,
                0,
                budget_overflow,
                gate_stats,
                allocation_warmup_active=warmup_all_detail,
            )
            return self._empty_decoded_data(coords, detail_counts, budget_stats, gate_losses)

        anchor_idx = torch.div(selected_idx, slot_count, rounding_mode="floor")
        slot_idx = selected_idx - anchor_idx * slot_count
        if anchor_indices is not None:
            selected_anchor_idx = anchor_indices.to(device=coords.device, dtype=torch.long)[anchor_idx]
        else:
            selected_anchor_idx = anchor_idx
        detail_counts = torch.bincount(anchor_idx, minlength=visible_count).to(dtype=torch.long)
        detail_h = self._decode_detail_hidden(
            coords,
            temporal_h,
            scale_input,
            rotate_input,
            anchor_idx,
            slot_idx,
            detail_spatial_h=detail_spatial_h[selected_idx],
            detail_feature_mode=detail_feature_mode,
        )
        color, rotation, scaling, opacity = self.decoder.decode_hidden(detail_h, slot_idx)

        opacity_mode = str(gate_opacity_mode or "st_identity").lower()
        if opacity_mode in ("multiply", "gate_multiply"):
            opacity = opacity * selected_gate.unsqueeze(-1)
        elif opacity_mode in ("st_identity", "straight_through", "identity_st"):
            if training:
                gate_multiplier = 1.0 + selected_gate.unsqueeze(-1) - selected_gate.detach().unsqueeze(-1)
                opacity = opacity * gate_multiplier
        elif opacity_mode in ("none", "identity"):
            pass
        else:
            raise ValueError(f"Unsupported gate_opacity_mode: {gate_opacity_mode}")

        budget_stats = self._gate_budget_stats(
            render_gaussian_budget,
            visible_count,
            detail_budget,
            selected_idx.numel(),
            budget_overflow,
            gate_stats,
            allocation_warmup_active=warmup_all_detail,
        )

        return {
            "d_color": color,
            "d_rotation": rotation,
            "d_scaling": scaling,
            "d_opacity": opacity,
            "detail_anchor_idx": anchor_idx,
            "detail_slot_idx": slot_idx.to(torch.long),
            "detail_counts": detail_counts,
            "budget_stats": budget_stats,
            "gate_losses": gate_losses,
            "gate_selected_idx": selected_idx.to(torch.long),
            "gate_selected_logits": selected_logits if selected_logits is not None else color.new_empty(0),
            "gate_selected_anchor_idx": selected_anchor_idx.to(torch.long),
            "gate_selected_slot_idx": slot_idx.to(torch.long),
            "gate_slot_count": int(slot_count),
        }


    def _step_slot_gate_st(
            self,
            coords,
            scale_input,
            rotate_input,
            offset_slots,
            pose,
            render_gaussian_budget=0,
            training=False,
            iteration=None,
            camera_center=None,
            gate_train_mode="topk_st",
            gate_eval_mode="topk",
            gate_temperature_init=1.0,
            gate_temperature_final=0.2,
            gate_temperature_max_steps=30000,
            gate_budget_lambda=0.0,
            gate_binary_lambda=0.0,
            gate_feature_mode="hash_view_context",
            gate_opacity_mode="st_multiply",
            gate_all_detail_until=0,
            anchor_indices=None,
            detail_feature_mode="detail_hash",
    ):
        self._slot_gate_used = True
        visible_count = coords.shape[0]
        slot_count = min(offset_slots.shape[1], self.max_detail_slots)
        render_gaussian_budget = int(render_gaussian_budget or 0)
        full_detail_budget = visible_count * slot_count
        budget_overflow = render_gaussian_budget > 0 and visible_count > render_gaussian_budget
        warmup_all_detail = self._gate_all_detail_warmup_active(
            training,
            iteration,
            gate_all_detail_until,
        )

        if visible_count == 0 or slot_count == 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = self._gate_budget_stats(
                render_gaussian_budget,
                visible_count,
                0,
                0,
                False,
                allocation_warmup_active=warmup_all_detail,
            )
            budget_stats["allocation_mode_id"] = 2.0
            return self._empty_decoded_data(coords, detail_counts, budget_stats, self._zero_gate_losses(coords))

        if render_gaussian_budget <= 0 or warmup_all_detail:
            detail_budget = full_detail_budget
        else:
            detail_budget = max(render_gaussian_budget - visible_count, 0)
            detail_budget = min(detail_budget, full_detail_budget)

        if detail_budget <= 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = self._gate_budget_stats(
                render_gaussian_budget,
                visible_count,
                0,
                0,
                budget_overflow,
                allocation_warmup_active=warmup_all_detail,
            )
            budget_stats["allocation_mode_id"] = 2.0
            return self._empty_decoded_data(coords, detail_counts, budget_stats, self._zero_gate_losses(coords))

        offset_slots = offset_slots[:, :slot_count]
        candidate_offset = offset_slots.reshape(-1, 3)
        candidate_slot_idx = torch.arange(slot_count, device=coords.device, dtype=torch.long)
        candidate_slot_idx = candidate_slot_idx.unsqueeze(0).expand(visible_count, -1).reshape(-1)
        candidate_xyz = coords[:, None, :] + offset_slots
        candidate_xyz = candidate_xyz.reshape(-1, 3)
        temporal_h = pose.unsqueeze(0).repeat(visible_count, 1)
        temporal_detail = temporal_h[:, None, :].expand(-1, slot_count, -1).reshape(-1, temporal_h.shape[-1])
        scale_detail = scale_input[:, None, :].expand(-1, slot_count, -1).reshape(-1, scale_input.shape[-1])
        rotate_detail = rotate_input[:, None, :].expand(-1, slot_count, -1).reshape(-1, rotate_input.shape[-1])

        detail_spatial_h = self.encoder.encode_xyz(candidate_xyz.detach())
        mode = str(gate_feature_mode or self.gate_feature_mode or "hash_view_context").lower()
        view_context = None
        if mode in ("hash_view_context", "view_context"):
            if not self.gate_uses_view_context or self.view_context_encoder is None:
                raise ValueError(
                    "MixGSModel must be constructed with gate_feature_mode='hash_view_context' "
                    "to use view context gate features."
                )
            view_context = self._view_context_feature(coords, scale_input, camera_center)
        gate_feature = self._gate_input_feature(
            detail_spatial_h,
            candidate_xyz,
            camera_center,
            mode,
            candidate_offset=candidate_offset,
            candidate_slot_idx=candidate_slot_idx,
            slot_count=slot_count,
            view_context=view_context,
        )

        gate_result = self.slot_gate_allocator.select(
            gate_feature,
            detail_budget,
            training=training,
            train_mode=gate_train_mode,
            eval_mode=gate_eval_mode,
            iteration=iteration,
            temperature_init=gate_temperature_init,
            temperature_final=gate_temperature_final,
            temperature_max_steps=gate_temperature_max_steps,
            budget_lambda=gate_budget_lambda,
            binary_lambda=gate_binary_lambda,
        )
        selected_idx = gate_result["selected_idx"]
        selected_gate = gate_result["selected_gate"]
        selected_logits = gate_result["logits"][selected_idx] if selected_idx.numel() > 0 else gate_result["logits"].new_empty(0)
        gate_losses = gate_result["losses"]
        gate_stats = gate_result["stats"]

        if selected_idx.numel() == 0:
            detail_counts = torch.zeros(visible_count, dtype=torch.long, device=coords.device)
            budget_stats = self._gate_budget_stats(
                render_gaussian_budget,
                visible_count,
                detail_budget,
                0,
                budget_overflow,
                gate_stats,
                allocation_warmup_active=warmup_all_detail,
            )
            budget_stats["allocation_mode_id"] = 2.0
            return self._empty_decoded_data(coords, detail_counts, budget_stats, gate_losses)

        anchor_idx = torch.div(selected_idx, slot_count, rounding_mode="floor")
        slot_idx = selected_idx - anchor_idx * slot_count
        if anchor_indices is not None:
            selected_anchor_idx = anchor_indices.to(device=coords.device, dtype=torch.long)[anchor_idx]
        else:
            selected_anchor_idx = anchor_idx
        detail_counts = torch.bincount(anchor_idx, minlength=visible_count).to(dtype=torch.long)
        detail_h = self._decode_detail_hidden(
            coords,
            temporal_h,
            scale_input,
            rotate_input,
            anchor_idx,
            slot_idx,
            detail_spatial_h=detail_spatial_h[selected_idx],
            detail_feature_mode=detail_feature_mode,
        )
        color, rotation, scaling, opacity = self.decoder.decode_hidden(detail_h, slot_idx)

        opacity_mode = str(gate_opacity_mode or "st_multiply").lower()
        if opacity_mode in ("multiply", "gate_multiply", "st_multiply", "straight_through_multiply"):
            opacity = opacity * selected_gate.unsqueeze(-1)
        elif opacity_mode in ("st_identity", "straight_through", "identity_st"):
            if training:
                gate_multiplier = 1.0 + selected_gate.unsqueeze(-1) - selected_gate.detach().unsqueeze(-1)
                opacity = opacity * gate_multiplier
        elif opacity_mode in ("none", "identity"):
            pass
        else:
            raise ValueError(f"Unsupported gate_opacity_mode: {gate_opacity_mode}")

        budget_stats = self._gate_budget_stats(
            render_gaussian_budget,
            visible_count,
            detail_budget,
            selected_idx.numel(),
            budget_overflow,
            gate_stats,
            allocation_warmup_active=warmup_all_detail,
        )
        budget_stats["allocation_mode_id"] = 2.0

        return {
            "d_color": color,
            "d_rotation": rotation,
            "d_scaling": scaling,
            "d_opacity": opacity,
            "detail_anchor_idx": anchor_idx,
            "detail_slot_idx": slot_idx.to(torch.long),
            "detail_counts": detail_counts,
            "budget_stats": budget_stats,
            "gate_losses": gate_losses,
            "gate_selected_idx": selected_idx.to(torch.long),
            "gate_selected_logits": selected_logits,
            "gate_selected_anchor_idx": selected_anchor_idx.to(torch.long),
            "gate_selected_slot_idx": slot_idx.to(torch.long),
            "gate_slot_count": int(slot_count),
        }

    def step(
            self,
            data,
            pose,
            render_gaussian_budget=0,
            scale_min=0.0,
            allocation_mode="proposal",
            training=False,
            iteration=None,
            camera_center=None,
            gate_train_mode="soft_all",
            gate_eval_mode="topk",
            gate_temperature_init=1.0,
            gate_temperature_final=0.2,
            gate_temperature_max_steps=30000,
            gate_budget_lambda=0.01,
            gate_binary_lambda=0.001,
            gate_feature_mode="detail_view",
            gate_opacity_mode="st_identity",
            gate_all_detail_until=0,
            clone_score_warmup_until=20000,
            clone_score_ramp_until=40000,
            clone_score_freeze_after=220000,
            clone_score_ema_beta=0.95,
            clone_score_eps=1e-6,
            clone_score_detail_grad_weight=1.0,
            clone_score_grad_clip=0.0,
            projected_area_all_detail_until=50000,
            projected_area_detail_stage_full_budget=True,
            projected_area_scale_power=2.0,
            projected_area_distance_power=2.0,
            projected_area_eps=1e-6,
            projected_complexity_tile_size=32,
            projected_complexity_color_weight=0.40,
            projected_complexity_depth_weight=0.30,
            projected_complexity_depth_var_weight=0.20,
            projected_complexity_shape_weight=0.10,
            projected_complexity_area_tau=0.0,
            projected_complexity_large_area_tau=0.0,
            projected_complexity_opacity_power=1.0,
            projected_inverse_area_tile_size=4,
            detail_feature_mode="detail_hash",
            include_projected_area_stats=True,
            stage_timer=None,
            use_projected_area_optimizations=False,
            viewpoint_camera=None,
    ):
        coords = data[0]
        scale_input = data[1]
        rotate_input = data[2]
        if len(data) > 3:
            offset_slots = data[3]
        else:
            offset_slots = coords.new_zeros((coords.shape[0], 1, 3))
        anchor_indices = None
        extra_inputs = {}
        if len(data) > 4:
            if isinstance(data[4], dict):
                extra_inputs = data[4]
            else:
                anchor_indices = data[4]
        if len(data) > 5 and isinstance(data[5], dict):
            extra_inputs = data[5]
        offset_slots = offset_slots.to(device=coords.device, dtype=coords.dtype)
        if anchor_indices is not None:
            anchor_indices = anchor_indices.to(device=coords.device, dtype=torch.long)

        mode = str(allocation_mode or "proposal").lower()
        if mode in ("clone_score", "clone_score_allocation"):
            return self._step_clone_score(
                coords,
                scale_input,
                rotate_input,
                offset_slots,
                pose,
                render_gaussian_budget=render_gaussian_budget,
                training=training,
                iteration=iteration,
                clone_score_warmup_until=clone_score_warmup_until,
                clone_score_ramp_until=clone_score_ramp_until,
                clone_score_freeze_after=clone_score_freeze_after,
                clone_score_eps=clone_score_eps,
                anchor_indices=anchor_indices,
                detail_feature_mode=detail_feature_mode,
            )
        if mode in (
                "projected_area",
                "projected_area_score",
                "area_score",
                "projected_complexity",
                "projected_complexity_score",
                "complexity_score",
                "image_complexity",
                "projection_inverse_area_with_visibility_gate",
                "projected_inverse_area",
                "inverse_projected_area",
                "inverse_area_visibility",
        ):
            projected_score_mode = "complexity" if mode in (
                "projected_complexity",
                "projected_complexity_score",
                "complexity_score",
                "image_complexity",
            ) else (
                "inverse_area_visibility" if mode in (
                    "projection_inverse_area_with_visibility_gate",
                    "projected_inverse_area",
                    "inverse_projected_area",
                    "inverse_area_visibility",
                ) else "area"
            )
            return self._step_projected_area(
                coords,
                scale_input,
                rotate_input,
                offset_slots,
                pose,
                render_gaussian_budget=render_gaussian_budget,
                iteration=iteration,
                camera_center=camera_center,
                viewpoint_camera=viewpoint_camera,
                projected_area_all_detail_until=projected_area_all_detail_until,
                projected_area_detail_stage_full_budget=projected_area_detail_stage_full_budget,
                projected_area_scale_power=projected_area_scale_power,
                projected_area_distance_power=projected_area_distance_power,
                projected_area_eps=projected_area_eps,
                projected_score_mode=projected_score_mode,
                color_input=extra_inputs.get("color_dc"),
                opacity_input=extra_inputs.get("opacity"),
                projected_complexity_tile_size=projected_complexity_tile_size,
                projected_complexity_color_weight=projected_complexity_color_weight,
                projected_complexity_depth_weight=projected_complexity_depth_weight,
                projected_complexity_depth_var_weight=projected_complexity_depth_var_weight,
                projected_complexity_shape_weight=projected_complexity_shape_weight,
                projected_complexity_area_tau=projected_complexity_area_tau,
                projected_complexity_large_area_tau=projected_complexity_large_area_tau,
                projected_complexity_opacity_power=projected_complexity_opacity_power,
                projected_inverse_area_tile_size=projected_inverse_area_tile_size,
                detail_feature_mode=detail_feature_mode,
                include_projected_area_stats=include_projected_area_stats,
                stage_timer=stage_timer,
                use_projected_area_optimizations=use_projected_area_optimizations,
            )
        if mode in ("slot_gate_st", "slot_st_gate", "st_slot_gate"):
            return self._step_slot_gate_st(
                coords,
                scale_input,
                rotate_input,
                offset_slots,
                pose,
                render_gaussian_budget=render_gaussian_budget,
                training=training,
                iteration=iteration,
                camera_center=camera_center,
                gate_train_mode=gate_train_mode,
                gate_eval_mode=gate_eval_mode,
                gate_temperature_init=gate_temperature_init,
                gate_temperature_final=gate_temperature_final,
                gate_temperature_max_steps=gate_temperature_max_steps,
                gate_budget_lambda=gate_budget_lambda,
                gate_binary_lambda=gate_binary_lambda,
                gate_feature_mode=gate_feature_mode,
                gate_opacity_mode=gate_opacity_mode,
                gate_all_detail_until=gate_all_detail_until,
                anchor_indices=anchor_indices,
                detail_feature_mode=detail_feature_mode,
            )
        if mode in ("gate", "learned_gate"):
            return self._step_gate(
                coords,
                scale_input,
                rotate_input,
                offset_slots,
                pose,
                render_gaussian_budget=render_gaussian_budget,
                training=training,
                iteration=iteration,
                camera_center=camera_center,
                gate_train_mode=gate_train_mode,
                gate_eval_mode=gate_eval_mode,
                gate_temperature_init=gate_temperature_init,
                gate_temperature_final=gate_temperature_final,
                gate_temperature_max_steps=gate_temperature_max_steps,
                gate_budget_lambda=gate_budget_lambda,
                gate_binary_lambda=gate_binary_lambda,
                gate_feature_mode=gate_feature_mode,
                gate_opacity_mode=gate_opacity_mode,
                gate_all_detail_until=gate_all_detail_until,
                anchor_indices=anchor_indices,
                detail_feature_mode=detail_feature_mode,
            )
        if mode not in ("proposal", "decoder", "decoder_proposal", ""):
            raise ValueError(f"Unsupported allocation_mode: {allocation_mode}")
        return self._step_proposal(
            coords,
            scale_input,
            rotate_input,
            offset_slots,
            pose,
            render_gaussian_budget=render_gaussian_budget,
            scale_min=scale_min,
            detail_feature_mode=detail_feature_mode,
        )

    def train_setting(self, training_args):
        self.decoder_lr_scale = training_args.decoder_lr_scale
        self.encoder_lr_scale = training_args.encoder_lr_scale

        l = [
            {'params': list(self.decoder.parameters()),
             'lr': training_args.position_lr_init * self.decoder_lr_scale,
             "name": "decoder"},
            {'params': list(self.encoder.parameters()),
             'lr': training_args.position_lr_init * self.encoder_lr_scale,
             "name": "encoder"},
            {'params': list(self.gate_allocator.parameters()) + list(self.slot_gate_allocator.parameters()) + (
                list(self.view_context_encoder.parameters())
                if self.gate_uses_view_context and self.view_context_encoder is not None else []
            ),
             'lr': training_args.position_lr_init * self.decoder_lr_scale,
             "name": "gate"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.decoder_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.decoder_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.decoder_lr_max_steps)
        self.encoder_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.encoder_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.encoder_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.decoder_lr_max_steps)

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "decoder/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, "decoder/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
        if self._slot_gate_used or self._clone_score_used:
            weights = {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "gate_allocator": self.gate_allocator.state_dict() if self._gate_used else None,
                "slot_gate_allocator": self.slot_gate_allocator.state_dict() if self._slot_gate_used else None,
                "view_context_encoder": (
                    self.view_context_encoder.state_dict()
                    if self.gate_uses_view_context and self.view_context_encoder is not None else None
                ),
                "clone_score_ema": self.clone_score_ema.detach().cpu() if self.clone_score_ema is not None else None,
                "clone_score_seen_ema": self.clone_score_seen_ema.detach().cpu() if self.clone_score_seen_ema is not None else None,
            }
        elif self._gate_used and self.gate_uses_view_context:
            weights = (
                self.encoder.state_dict(),
                self.decoder.state_dict(),
                self.gate_allocator.state_dict(),
                self.view_context_encoder.state_dict(),
            )
        elif self._gate_used:
            weights = (self.encoder.state_dict(), self.decoder.state_dict(), self.gate_allocator.state_dict())
        else:
            weights = (self.encoder.state_dict(), self.decoder.state_dict())
        torch.save(weights, os.path.join(out_weights_path, 'decoder.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "decoder"))
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(loaded_iter))
        else:
            loaded_iter = iteration
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(loaded_iter))

        print("Load weight:", weights_path)
        weights = torch.load(weights_path, map_location='cuda')
        gate_weight = None
        slot_gate_weight = None
        view_context_weight = None
        clone_score_ema = None
        clone_score_seen_ema = None
        if isinstance(weights, dict):
            grid_weight = weights["encoder"]
            network_weight = weights["decoder"]
            gate_weight = weights.get("gate_allocator")
            slot_gate_weight = weights.get("slot_gate_allocator")
            view_context_weight = weights.get("view_context_encoder")
            clone_score_ema = weights.get("clone_score_ema")
            clone_score_seen_ema = weights.get("clone_score_seen_ema")
        elif isinstance(weights, (tuple, list)) and len(weights) == 4:
            grid_weight, network_weight, gate_weight, view_context_weight = weights
        elif isinstance(weights, (tuple, list)) and len(weights) == 3:
            grid_weight, network_weight, gate_weight = weights
        elif isinstance(weights, (tuple, list)) and len(weights) == 2:
            grid_weight, network_weight = weights
        else:
            raise RuntimeError(f"Unsupported decoder checkpoint format: {weights_path}")
        try:
            self.decoder.load_state_dict(network_weight)
        except RuntimeError as exc:
            print("Decoder state dict loaded with strict=False:", exc)
            self.decoder.load_state_dict(network_weight, strict=False)
        if gate_weight is not None:
            try:
                self.gate_allocator.load_state_dict(gate_weight)
                self._gate_used = True
            except RuntimeError as exc:
                print("Gate allocator state dict skipped due to incompatible shape:", exc)
        if slot_gate_weight is not None:
            try:
                self.slot_gate_allocator.load_state_dict(slot_gate_weight)
                self._slot_gate_used = True
            except RuntimeError as exc:
                print("Slot gate allocator state dict skipped due to incompatible shape:", exc)
        if view_context_weight is not None and self.view_context_encoder is not None:
            try:
                self.view_context_encoder.load_state_dict(view_context_weight)
            except RuntimeError as exc:
                print("View context state dict skipped due to incompatible shape:", exc)
        if clone_score_ema is not None and clone_score_seen_ema is not None:
            self.clone_score_ema = clone_score_ema.detach().to(device="cuda", dtype=torch.float32)
            self.clone_score_seen_ema = clone_score_seen_ema.detach().to(device="cuda", dtype=torch.float32)
            self._clone_score_used = True
        self.encoder.load_state_dict(grid_weight)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "decoder":
                lr = self.decoder_lr_scheduler(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "gate":
                lr = self.decoder_lr_scheduler(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == "encoder":
                lr = self.encoder_lr_scheduler(iteration)
                param_group['lr'] = lr

