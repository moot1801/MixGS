import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.network import GSDecoder, GSEncoder
from scene.gate_allocation import GateAllocator
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
    ):
        self.encoder = GSEncoder(**hash_args).cuda()
        self.spatial_dim = self.encoder.canonical_level_dim * self.encoder.canonical_num_levels
        self.mlp_dim = 10
        self.max_detail_slots, self.detail_count_choices, _ = resolve_detail_count_choices(
            max_detail_slots, detail_count_choices
        )
        net_args = dict(net_args)
        net_args["max_detail_slots"] = self.max_detail_slots
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
        self._gate_used = False

        self.decoder_lr_scale = 50.0
        self.encoder_lr_scale = 100.0

    def _quantize_detail_counts(self, counts):
        if counts.numel() == 0:
            return counts
        quantized = torch.zeros_like(counts)
        for choice in self.detail_count_choices:
            quantized = torch.where(counts >= choice, counts.new_full((), choice), quantized)
        return quantized

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

    def _step_proposal(self, coords, scale_input, rotate_input, offset_slots, pose, render_gaussian_budget=0, scale_min=0.0):
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
            detail_spatial_h = self.encoder.encode_xyz(detail_xyz.detach())
            detail_h = self.decoder.compute_hidden(
                detail_spatial_h,
                temporal_h[anchor_idx],
                scale_input[anchor_idx],
                rotate_input[anchor_idx],
            )
            color, rotation, scaling, opacity = self.decoder.decode_hidden(detail_h)
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
        else:
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
        detail_h = self.decoder.compute_hidden(
            detail_spatial_h[selected_idx],
            temporal_detail[selected_idx],
            scale_detail[selected_idx],
            rotate_detail[selected_idx],
        )
        color, rotation, scaling, opacity = self.decoder.decode_hidden(detail_h)

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
    ):
        coords = data[0]
        scale_input = data[1]
        rotate_input = data[2]
        if len(data) > 3:
            offset_slots = data[3]
        else:
            offset_slots = coords.new_zeros((coords.shape[0], 1, 3))
        anchor_indices = data[4] if len(data) > 4 else None
        offset_slots = offset_slots.to(device=coords.device, dtype=coords.dtype)
        if anchor_indices is not None:
            anchor_indices = anchor_indices.to(device=coords.device, dtype=torch.long)

        mode = str(allocation_mode or "proposal").lower()
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
            {'params': list(self.gate_allocator.parameters()) + (
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
        if self._gate_used and self.gate_uses_view_context:
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
        view_context_weight = None
        if isinstance(weights, (tuple, list)) and len(weights) == 4:
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
        if view_context_weight is not None and self.view_context_encoder is not None:
            try:
                self.view_context_encoder.load_state_dict(view_context_weight)
            except RuntimeError as exc:
                print("View context state dict skipped due to incompatible shape:", exc)
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

