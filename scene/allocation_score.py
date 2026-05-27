import torch
import torch.nn.functional as F


_MODE_IDS = {
    "decoder": 0.0,
    "uniform": 1.0,
    "residual_patch": 2.0,
    "hybrid": 3.0,
    "residual_correction": 4.0,
}


def _as_config(config):
    if config is None:
        return {"mode": "decoder"}
    if isinstance(config, str):
        stripped = config.strip()
        return {"mode": stripped or "decoder"}
    return dict(config)


def _scalar_int(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return _scalar_int(value[0])
    return int(value)


def _matrix4(value, device):
    if isinstance(value, torch.Tensor):
        matrix = value.to(device=device)
    else:
        matrix = torch.tensor(value, dtype=torch.float32, device=device)
    while matrix.dim() > 2:
        matrix = matrix[0]
    return matrix.to(dtype=torch.float32)


def _residual_hw(rendered_image, gt_image):
    if gt_image.dim() == 4:
        gt_image = gt_image[0]
    if rendered_image.dim() == 4:
        rendered_image = rendered_image[0]
    rendered_image = torch.clamp(rendered_image.detach(), 0.0, 1.0)
    gt_image = torch.clamp(gt_image.detach().to(rendered_image.device), 0.0, 1.0)
    return torch.mean(torch.abs(rendered_image - gt_image), dim=0)


def _project_to_pixel(anchor_xyz, viewpoint):
    device = anchor_xyz.device
    if anchor_xyz.numel() == 0:
        empty = anchor_xyz.new_empty((0,))
        return empty, empty, torch.zeros((0,), dtype=torch.bool, device=device)

    transform = _matrix4(viewpoint["full_proj_transform"], device)
    ones = torch.ones((anchor_xyz.shape[0], 1), dtype=anchor_xyz.dtype, device=device)
    points_h = torch.cat([anchor_xyz, ones], dim=1).to(torch.float32)
    clip = points_h @ transform
    w = clip[:, 3]
    ndc = clip[:, :3] / (w.unsqueeze(-1) + 1e-7)

    width = _scalar_int(viewpoint["image_width"])
    height = _scalar_int(viewpoint["image_height"])
    x = (ndc[:, 0] + 1.0) * 0.5 * max(width - 1, 1)
    y = (1.0 - ndc[:, 1]) * 0.5 * max(height - 1, 1)
    valid = (w.abs() > 1e-7) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    return x, y, valid


class BaseAllocationScorer:
    mode = "decoder"
    requires_residual = False

    def __init__(self, config=None):
        self.config = _as_config(config)
        self.eps = float(self.config.get("eps", 1e-4))

    def initial_scores(self, visible_count, device, dtype, iteration=None, joint_start_iter=None, anchor_indices=None):
        return None

    def final_scores(
            self, rendered_image, gt_image, viewpoint, anchor_xyz, proposal_scores=None,
            iteration=None, joint_start_iter=None, anchor_indices=None, contribution_scores=None):
        return None

    def proposal_scale_power(self):
        return 1.0

    def should_compute_residual(self, iteration=None, joint_start_iter=None):
        return self.requires_residual

    def after_step(self, detail_counts, anchor_indices=None, iteration=None, joint_start_iter=None):
        return

    def score_stats(self, scores):
        stats = {"mode_id": _MODE_IDS.get(self.mode, -1.0)}
        if scores is None or scores.numel() == 0:
            return stats
        values = scores.detach().to(dtype=torch.float32)
        stats.update({
            "mean": values.mean().item(),
            "max": values.max().item(),
            "min": values.min().item(),
        })
        return stats


class DecoderProposalScorer(BaseAllocationScorer):
    mode = "decoder"


class UniformScorer(BaseAllocationScorer):
    mode = "uniform"

    def initial_scores(self, visible_count, device, dtype, iteration=None, joint_start_iter=None, anchor_indices=None):
        return torch.ones((int(visible_count),), dtype=dtype, device=device)


class ResidualPatchScorer(BaseAllocationScorer):
    mode = "residual_patch"
    requires_residual = True

    def __init__(self, config=None):
        super().__init__(config)
        patch_size = max(1, int(self.config.get("residual_patch_size", 5)))
        if patch_size % 2 == 0:
            patch_size += 1
        self.patch_size = patch_size
        self.pooling = str(self.config.get("residual_pooling", "mean")).lower()
        self.normalize = str(self.config.get("residual_normalize", "rank")).lower()
        self.residual_alpha = float(self.config.get("residual_alpha", 1.0))
        self.topk_fraction = float(self.config.get("residual_topk_fraction", 0.25))

    def _patch_scores(self, residual_map, viewpoint, anchor_xyz):
        residual_map = residual_map.to(dtype=torch.float32, device=anchor_xyz.device)
        height, width = residual_map.shape[-2], residual_map.shape[-1]
        x, y, valid = _project_to_pixel(anchor_xyz, viewpoint)
        if x.numel() == 0:
            return x

        radius = self.patch_size // 2
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=anchor_xyz.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        sample_x = x[:, None] + dx.reshape(1, -1)
        sample_y = y[:, None] + dy.reshape(1, -1)
        norm_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
        norm_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            residual_map.view(1, 1, height, width),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(x.shape[0], -1)

        if self.pooling == "max":
            scores = sampled.max(dim=1).values
        elif self.pooling == "topk":
            k = max(1, int(sampled.shape[1] * self.topk_fraction))
            scores = torch.topk(sampled, k=k, dim=1).values.mean(dim=1)
        else:
            scores = sampled.mean(dim=1)
        return torch.where(valid, scores, torch.zeros_like(scores))

    def _normalize_scores(self, scores):
        if scores.numel() == 0:
            return scores
        positive = scores > 0
        if not torch.any(positive):
            return torch.ones_like(scores)

        normalized = torch.zeros_like(scores)
        values = scores[positive]
        if values.numel() > 1 and (values.max() - values.min()).item() <= self.eps:
            normalized[positive] = 1.0
            return normalized
        if self.normalize == "mean":
            normalized[positive] = values / torch.clamp(values.mean(), min=self.eps)
        elif self.normalize == "minmax":
            denom = torch.clamp(values.max() - values.min(), min=self.eps)
            normalized[positive] = (values - values.min()) / denom
        else:
            order = torch.argsort(values)
            ranks = torch.empty_like(values)
            if values.numel() == 1:
                ranks.fill_(1.0)
            else:
                ranks[order] = torch.linspace(0.0, 1.0, values.numel(), dtype=values.dtype, device=values.device)
            normalized[positive] = ranks
        return torch.clamp_min(normalized, 0.0)

    def residual_scores(self, rendered_image, gt_image, viewpoint, anchor_xyz, contribution_scores=None):
        if contribution_scores is not None:
            scores = contribution_scores.detach().to(device=anchor_xyz.device, dtype=torch.float32).flatten()
            if scores.shape[0] != anchor_xyz.shape[0]:
                raise ValueError("contribution_scores must match visible anchor count")
            return self._normalize_scores(scores)

        residual_map = _residual_hw(rendered_image, gt_image)
        return self._normalize_scores(self._patch_scores(residual_map, viewpoint, anchor_xyz))

    def combine_scores(self, residual_scores, proposal_scores=None):
        return torch.pow(torch.clamp_min(residual_scores, 0.0) + self.eps, self.residual_alpha)

    def final_scores(
            self, rendered_image, gt_image, viewpoint, anchor_xyz, proposal_scores=None,
            iteration=None, joint_start_iter=None, anchor_indices=None, contribution_scores=None):
        residual_scores = self.residual_scores(
            rendered_image,
            gt_image,
            viewpoint,
            anchor_xyz,
            contribution_scores=contribution_scores,
        )
        return self.combine_scores(residual_scores, proposal_scores)


class HybridScorer(ResidualPatchScorer):
    mode = "hybrid"

    def __init__(self, config=None):
        super().__init__(config)
        self.decoder_beta = float(self.config.get("decoder_beta", 1.0))

    def combine_scores(self, residual_scores, proposal_scores=None):
        residual_term = torch.pow(torch.clamp_min(residual_scores, 0.0) + self.eps, self.residual_alpha)
        if proposal_scores is None or self.decoder_beta == 0.0:
            return residual_term
        proposal_scores = proposal_scores.detach().to(device=residual_scores.device, dtype=residual_scores.dtype)
        proposal_term = torch.pow(torch.clamp_min(proposal_scores, 0.0) + self.eps, self.decoder_beta)
        return residual_term * proposal_term


class ResidualCorrectionScorer(ResidualPatchScorer):
    mode = "residual_correction"

    def __init__(self, config=None):
        super().__init__(config)
        self.base_scale_power = float(self.config.get("base_scale_power", 0.5))
        self.residual_clip_max = float(self.config.get("residual_clip_max", 3.0))
        self.gamma_start = float(self.config.get("residual_gamma_start", 0.0))
        self.gamma_end = float(self.config.get("residual_gamma_end", 0.5))
        self.gamma_ramp_iters = max(1, int(self.config.get("residual_gamma_ramp_iters", 50_000)))
        self.start_stage = str(self.config.get("residual_start_stage", "joint")).lower()
        self.start_iter = int(self.config.get("residual_start_iter", 1))
        self.selection_penalty_power = float(self.config.get("selection_penalty_power", 0.5))
        self.selection_counts = None
        self._last_stats = {"gamma": 0.0}

    def proposal_scale_power(self):
        return self.base_scale_power

    def _start_iteration(self, joint_start_iter=None):
        if self.start_stage == "joint" and joint_start_iter is not None:
            return int(joint_start_iter)
        return self.start_iter

    def gamma_at(self, iteration=None, joint_start_iter=None):
        if iteration is None:
            return 0.0
        start_iteration = self._start_iteration(joint_start_iter)
        if int(iteration) < start_iteration:
            return 0.0
        progress = min(max((int(iteration) - start_iteration) / float(self.gamma_ramp_iters), 0.0), 1.0)
        return self.gamma_start + (self.gamma_end - self.gamma_start) * progress

    def should_compute_residual(self, iteration=None, joint_start_iter=None):
        return self.gamma_at(iteration, joint_start_iter) > 0.0

    def initial_scores(self, visible_count, device, dtype, iteration=None, joint_start_iter=None, anchor_indices=None):
        self._last_stats = {"gamma": self.gamma_at(iteration, joint_start_iter)}
        return None

    def _ensure_selection_counts(self, anchor_indices):
        if anchor_indices is None or anchor_indices.numel() == 0:
            return
        max_index = int(anchor_indices.max().item())
        if self.selection_counts is None:
            self.selection_counts = torch.zeros(max_index + 1, dtype=torch.float32, device=anchor_indices.device)
        elif self.selection_counts.shape[0] <= max_index:
            expanded = torch.zeros(max_index + 1, dtype=self.selection_counts.dtype, device=anchor_indices.device)
            expanded[:self.selection_counts.shape[0]] = self.selection_counts.to(anchor_indices.device)
            self.selection_counts = expanded
        elif self.selection_counts.device != anchor_indices.device:
            self.selection_counts = self.selection_counts.to(anchor_indices.device)

    def _selection_penalty(self, anchor_indices, device, dtype):
        if self.selection_penalty_power <= 0.0 or anchor_indices is None or anchor_indices.numel() == 0:
            return None
        anchor_indices = anchor_indices.to(device=device, dtype=torch.long)
        self._ensure_selection_counts(anchor_indices)
        counts = self.selection_counts[anchor_indices].to(device=device, dtype=dtype)
        return torch.pow(counts + 1.0, self.selection_penalty_power)

    def final_scores(
            self, rendered_image, gt_image, viewpoint, anchor_xyz, proposal_scores=None,
            iteration=None, joint_start_iter=None, anchor_indices=None, contribution_scores=None):
        gamma = self.gamma_at(iteration, joint_start_iter)
        residual_scores = self.residual_scores(
            rendered_image,
            gt_image,
            viewpoint,
            anchor_xyz,
            contribution_scores=contribution_scores,
        )
        clipped_residual = torch.clamp(residual_scores, 0.0, self.residual_clip_max)
        if proposal_scores is None:
            base_scores = torch.ones_like(clipped_residual)
        else:
            base_scores = proposal_scores.detach().to(device=clipped_residual.device, dtype=clipped_residual.dtype)
        correction = 1.0 + gamma * clipped_residual
        scores = base_scores * correction
        penalty = self._selection_penalty(anchor_indices, scores.device, scores.dtype)
        if penalty is not None:
            scores = scores / penalty
            selection_mean = (torch.pow(penalty, 1.0 / max(self.selection_penalty_power, self.eps)) - 1.0).mean().item()
        else:
            selection_mean = 0.0
        self._last_stats = {
            "gamma": float(gamma),
            "residual_mean": residual_scores.detach().mean().item() if residual_scores.numel() > 0 else 0.0,
            "residual_max": residual_scores.detach().max().item() if residual_scores.numel() > 0 else 0.0,
            "correction_mean": correction.detach().mean().item() if correction.numel() > 0 else 0.0,
            "selection_count_mean": float(selection_mean),
        }
        return scores

    def after_step(self, detail_counts, anchor_indices=None, iteration=None, joint_start_iter=None):
        if detail_counts is None or anchor_indices is None or detail_counts.numel() == 0:
            return
        if self.gamma_at(iteration, joint_start_iter) <= 0.0:
            return
        anchor_indices = anchor_indices.to(device=detail_counts.device, dtype=torch.long)
        self._ensure_selection_counts(anchor_indices)
        self.selection_counts.index_add_(
            0,
            anchor_indices,
            detail_counts.detach().to(device=detail_counts.device, dtype=self.selection_counts.dtype),
        )

    def score_stats(self, scores):
        stats = super().score_stats(scores)
        stats.update(self._last_stats)
        return stats


def build_allocation_scorer(config=None):
    config = _as_config(config)
    mode = str(config.get("mode", "decoder")).lower()
    if mode in ("decoder", "proposal", "decoder_proposal"):
        return DecoderProposalScorer(config)
    if mode == "uniform":
        return UniformScorer(config)
    if mode in ("residual", "residual_patch"):
        return ResidualPatchScorer(config)
    if mode == "hybrid":
        return HybridScorer(config)
    if mode in ("residual_correction", "bounded_residual", "residual_bounded"):
        return ResidualCorrectionScorer(config)
    raise ValueError("Unknown allocation score mode: {}".format(mode))
