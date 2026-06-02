import torch


_MODE_IDS = {
    "decoder": 0.0,
}


def _as_config(config):
    if config is None:
        return {"mode": "decoder"}
    if isinstance(config, str):
        stripped = config.strip()
        return {"mode": stripped or "decoder"}
    return dict(config)


class DecoderProposalScorer:
    mode = "decoder"

    def __init__(self, config=None):
        self.config = _as_config(config)
        mode = str(self.config.get("mode", "decoder")).lower()
        if mode not in ("decoder", "proposal", "decoder_proposal", ""):
            raise ValueError(
                "Only decoder/proposal allocation_score is supported."
            )

    def initial_scores(self, visible_count, device, dtype, iteration=None, joint_start_iter=None, anchor_indices=None):
        return None

    def proposal_scale_power(self):
        return 1.0

    def score_stats(self, scores):
        stats = {"mode_id": _MODE_IDS[self.mode]}
        if scores is None or scores.numel() == 0:
            return stats
        values = scores.detach().to(dtype=torch.float32)
        stats.update({
            "mean": values.mean().item(),
            "max": values.max().item(),
            "min": values.min().item(),
        })
        return stats


def build_allocation_scorer(config=None):
    return DecoderProposalScorer(config)
