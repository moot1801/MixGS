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

    def step(self, data, pose, render_gaussian_budget=0, scale_min=0.0, allocation_scores=None, proposal_scale_power=1.0):
        coords = data[0]
        scale_input = data[1]
        rotate_input = data[2]
        if len(data) > 3:
            offset_slots = data[3]
        else:
            offset_slots = coords.new_zeros((coords.shape[0], 1, 3))
        offset_slots = offset_slots.to(device=coords.device, dtype=coords.dtype)

        temporal_h = pose.unsqueeze(0).repeat(coords.size()[0], 1)
        render_gaussian_budget = int(render_gaussian_budget or 0)
        if render_gaussian_budget <= 0:
            scores = coords.new_ones(coords.shape[0])
        else:
            if allocation_scores is None:
                with torch.no_grad():
                    spatial_h = self.encoder.encode_xyz(coords)
                    h = self.decoder.compute_hidden(spatial_h, temporal_h, scale_input, rotate_input)
                    scores = self.decoder.proposal_score(
                        h,
                        scale_min=scale_min,
                        scale_power=proposal_scale_power,
                    )
                scores = scores.detach()
            else:
                scores = allocation_scores.detach().to(device=coords.device, dtype=coords.dtype).flatten()
                if scores.shape[0] != coords.shape[0]:
                    raise ValueError("allocation_scores must match visible anchor count")
        detail_counts, budget_stats = self._allocate_detail_counts(scores, render_gaussian_budget)
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

