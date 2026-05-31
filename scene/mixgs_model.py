import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.network import GSDecoder, GSEncoder
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, resolve_detail_count_choices


class MixGSModel:
    def __init__(
            self,
            hash_args,
            net_args,
            max_detail_slots=1,
            detail_count_choices=None,
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

    def _all_slot_indices(self, anchor_count, device):
        slot_count = self.max_detail_slots
        anchor_idx = torch.arange(anchor_count, device=device, dtype=torch.long).repeat_interleave(slot_count)
        slot_idx = torch.arange(slot_count, device=device, dtype=torch.long).repeat(anchor_count)
        return anchor_idx, slot_idx

    def _slot_budget_stats(self, render_gaussian_budget, visible_count, detail_budget, selected_detail_count):
        return {
            "render_gaussian_budget": int(render_gaussian_budget or 0),
            "visible_anchor_count": int(visible_count),
            "detail_budget": int(detail_budget),
            "selected_detail_count": int(selected_detail_count),
            "budget_overflow": bool(visible_count > int(render_gaussian_budget or 0)),
        }

    def step(self, data, pose, render_gaussian_budget=0, scale_min=0.0, allocation_scores=None, proposal_scale_power=1.0):
        coords = data[0]
        scale_input = data[1]
        rotate_input = data[2]
        if len(data) > 3:
            offset_slots = data[3]
        else:
            offset_slots = coords.new_zeros((coords.shape[0], 1, 3))
        offset_slots = offset_slots.to(device=coords.device, dtype=coords.dtype)

        visible_count = coords.shape[0]
        render_gaussian_budget = int(render_gaussian_budget or 0)
        has_external_scores = allocation_scores is not None

        if allocation_scores is None:
            anchor_scores = coords.new_ones(visible_count)
        else:
            anchor_scores = allocation_scores.to(device=coords.device, dtype=coords.dtype).flatten()
            if anchor_scores.shape[0] != visible_count:
                raise ValueError("allocation_scores must match visible anchor count")

        temporal_h = pose.unsqueeze(0).repeat(visible_count, 1)
        proposal_scores = None
        slot_scores = coords.new_empty((0,))

        if visible_count == 0:
            detail_counts = torch.zeros(0, dtype=torch.long, device=coords.device)
            anchor_idx = torch.empty(0, dtype=torch.long, device=coords.device)
            slot_idx = torch.empty(0, dtype=torch.long, device=coords.device)
            budget_stats = self._slot_budget_stats(render_gaussian_budget, 0, 0, 0)
        elif render_gaussian_budget <= 0:
            default_count = min(self.detail_count_choices[0], self.max_detail_slots)
            anchor_idx = torch.arange(visible_count, device=coords.device, dtype=torch.long).repeat_interleave(default_count)
            slot_idx = torch.arange(default_count, device=coords.device, dtype=torch.long).repeat(visible_count)
            detail_counts = torch.full((visible_count,), default_count, dtype=torch.long, device=coords.device)
            detail_budget = visible_count * default_count
            budget_stats = {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": int(visible_count),
                "detail_budget": int(detail_budget),
                "selected_detail_count": int(anchor_idx.shape[0]),
                "budget_overflow": False,
            }
        else:
            candidate_count = visible_count * self.max_detail_slots
            detail_budget = max(render_gaussian_budget - visible_count, 0)
            detail_budget = min(detail_budget, candidate_count)
            budget_overflow = visible_count > render_gaussian_budget
            slot_score_chunk_size = 262144
            slot_scores = coords.new_empty((candidate_count,))
            slot_proposal_scores = coords.new_empty((candidate_count,))

            with torch.no_grad():
                for start_idx in range(0, candidate_count, slot_score_chunk_size):
                    end_idx = min(start_idx + slot_score_chunk_size, candidate_count)
                    flat_idx = torch.arange(start_idx, end_idx, device=coords.device, dtype=torch.long)
                    chunk_anchor_idx = torch.div(flat_idx, self.max_detail_slots, rounding_mode="floor")
                    chunk_slot_idx = flat_idx - chunk_anchor_idx * self.max_detail_slots
                    detail_xyz = coords[chunk_anchor_idx] + offset_slots[chunk_anchor_idx, chunk_slot_idx]
                    detail_spatial_h = self.encoder.encode_xyz(detail_xyz.detach())
                    detail_h = self.decoder.compute_hidden(
                        detail_spatial_h,
                        temporal_h[chunk_anchor_idx],
                        scale_input[chunk_anchor_idx],
                        rotate_input[chunk_anchor_idx],
                    )
                    chunk_proposal_scores = self.decoder.proposal_score(
                        detail_h,
                        scale_min=scale_min,
                        scale_power=proposal_scale_power,
                    )
                    slot_proposal_scores[start_idx:end_idx] = chunk_proposal_scores
                    slot_scores[start_idx:end_idx] = chunk_proposal_scores * torch.clamp_min(anchor_scores[chunk_anchor_idx], 0.0)

            proposal_scores = slot_proposal_scores.view(visible_count, self.max_detail_slots).amax(dim=1)
            if detail_budget > 0:
                scores = torch.clamp_min(slot_scores, 0.0)
                if scores.sum().item() <= 0.0:
                    scores = torch.ones_like(scores)
                selected = torch.topk(scores, k=detail_budget, largest=True, sorted=False).indices
                anchor_idx = torch.div(selected, self.max_detail_slots, rounding_mode="floor")
                slot_idx = selected - anchor_idx * self.max_detail_slots
            else:
                anchor_idx = torch.empty(0, dtype=torch.long, device=coords.device)
                slot_idx = torch.empty(0, dtype=torch.long, device=coords.device)

            detail_counts = torch.bincount(anchor_idx, minlength=visible_count).to(torch.long)
            budget_stats = {
                "render_gaussian_budget": render_gaussian_budget,
                "visible_anchor_count": int(visible_count),
                "detail_budget": int(detail_budget),
                "selected_detail_count": int(anchor_idx.shape[0]),
                "budget_overflow": bool(budget_overflow),
            }

        if anchor_idx.numel() > 0:
            detail_xyz = coords[anchor_idx] + offset_slots[anchor_idx, slot_idx]
            detail_spatial_h = self.encoder.encode_xyz(detail_xyz.detach())
            selected_detail_h = self.decoder.compute_hidden(
                detail_spatial_h,
                temporal_h[anchor_idx],
                scale_input[anchor_idx],
                rotate_input[anchor_idx],
            )
            color, rotation, scaling, opacity = self.decoder.decode_hidden(selected_detail_h)
        else:
            color = coords.new_empty((0, 3))
            rotation = coords.new_empty((0, 4))
            scaling = coords.new_empty((0, 3))
            opacity = coords.new_empty((0, 1))

        return {
            "d_color": color,
            "d_rotation": rotation,
            "d_scaling": scaling,
            "d_opacity": opacity,
            "detail_anchor_idx": anchor_idx,
            "detail_slot_idx": slot_idx,
            "detail_counts": detail_counts,
            "budget_stats": budget_stats,
            "proposal_scores": proposal_scores,
            "allocation_scores": anchor_scores if has_external_scores else proposal_scores,
            "slot_allocation_scores": slot_scores,
        }
    
    def train_setting(self, training_args):
        self.decoder_lr_scale = training_args.decoder_lr_scale
        self.encoder_lr_scale = training_args.encoder_lr_scale

        l = [
            {'params': list(self.decoder.parameters()),
             'lr': training_args.position_lr_init * self.decoder_lr_scale,
             "name": "decoder"},
            {'params': list(self.encoder.parameters()),
             'lr': training_args.position_lr_init * self.encoder_lr_scale,
             "name": "encoder"}
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
        torch.save((self.encoder.state_dict(), self.decoder.state_dict()), os.path.join(out_weights_path, 'decoder.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "decoder"))
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(loaded_iter))
        else:
            loaded_iter = iteration
            weights_path = os.path.join(model_path, "decoder/iteration_{}/decoder.pth".format(loaded_iter))

        print("Load weight:", weights_path)
        grid_weight, network_weight = torch.load(weights_path, map_location='cuda')
        try:
            self.decoder.load_state_dict(network_weight)
        except RuntimeError as exc:
            print("Decoder state dict loaded with strict=False:", exc)
            self.decoder.load_state_dict(network_weight, strict=False)
        self.encoder.load_state_dict(grid_weight)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "decoder":
                lr = self.decoder_lr_scheduler(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == "encoder":
                lr = self.encoder_lr_scheduler(iteration)
                param_group['lr'] = lr

