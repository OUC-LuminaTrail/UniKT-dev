"""Linear-time, response-conditioned skill cross-effects for ReKTP."""

import math

import torch
from torch import nn


def _compose_affine(
    left_decay: torch.Tensor,
    left_write: torch.Tensor,
    right_decay: torch.Tensor,
    right_write: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``left`` followed by ``right`` for ``h' = decay * h + write``."""
    return (
        right_decay * left_decay,
        right_decay * left_write + right_write,
    )


def diagonal_affine_exclusive_scan(
    decay: torch.Tensor,
    write: torch.Tensor,
) -> torch.Tensor:
    """Return states before each diagonal affine transition.

    The recursive pair scan performs O(L) work with O(log L) parallel depth and
    never materializes a pairwise ``[L, L]`` interaction matrix. The initial
    state is zero.
    """
    if decay.shape != write.shape:
        raise ValueError("decay and write must have the same shape")
    if decay.ndim < 2:
        raise ValueError("decay and write must include batch and sequence axes")
    if decay.size(1) == 0:
        return torch.zeros_like(write)

    def scan(
        current_decay: torch.Tensor,
        current_write: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = current_decay.size(1)
        if length == 1:
            return torch.ones_like(current_decay), torch.zeros_like(current_write)

        paired_length = length // 2
        paired_decay, paired_write = _compose_affine(
            current_decay[:, : 2 * paired_length : 2],
            current_write[:, : 2 * paired_length : 2],
            current_decay[:, 1 : 2 * paired_length : 2],
            current_write[:, 1 : 2 * paired_length : 2],
        )
        pair_prefix_decay, pair_prefix_write = scan(paired_decay, paired_write)

        odd_prefix_decay, odd_prefix_write = _compose_affine(
            pair_prefix_decay,
            pair_prefix_write,
            current_decay[:, : 2 * paired_length : 2],
            current_write[:, : 2 * paired_length : 2],
        )
        prefix_decay = torch.stack(
            (pair_prefix_decay, odd_prefix_decay), dim=2
        ).flatten(1, 2)
        prefix_write = torch.stack(
            (pair_prefix_write, odd_prefix_write), dim=2
        ).flatten(1, 2)

        if length % 2:
            final_decay, final_write = _compose_affine(
                pair_prefix_decay[:, -1:],
                pair_prefix_write[:, -1:],
                paired_decay[:, -1:],
                paired_write[:, -1:],
            )
            prefix_decay = torch.cat((prefix_decay, final_decay), dim=1)
            prefix_write = torch.cat((prefix_write, final_write), dim=1)
        return prefix_decay, prefix_write

    _, prefix_write = scan(decay, write)
    return prefix_write


class SkillCrossEffect(nn.Module):
    """Low-rank multi-timescale Hawkes-style skill interaction state.

    Source embeddings are conditioned on both skill and response. Separate
    target embeddings make the learned effect directional. Fixed exponential
    rates permit an exact linear-time affine scan for the represented kernel.
    """

    def __init__(
        self,
        num_skills: int,
        rank: int,
        num_scales: int,
        max_gap_bins: int,
    ) -> None:
        super().__init__()
        if num_skills < 1:
            raise ValueError("num_skills must be positive")
        if rank < 1:
            raise ValueError("cross-effect rank must be positive")
        if num_scales < 2:
            raise ValueError("cross-effect num_scales must be at least 2")
        if max_gap_bins < 1:
            raise ValueError("max_gap_bins must be positive")

        self.num_skills = num_skills
        self.rank = rank
        self.num_scales = num_scales
        width = rank * num_scales
        self.source_embed = nn.Embedding(2 * num_skills, width)
        self.target_embed = nn.Embedding(num_skills + 1, width, padding_idx=num_skills)
        self.logit_scale = nn.Parameter(torch.zeros(()))

        positive_scales = num_scales - 1
        max_exponent = max(0.0, float(max_gap_bins - 2))
        half_lives = torch.pow(
            2.0,
            torch.linspace(0.0, max_exponent, positive_scales),
        )
        rates = torch.cat((torch.zeros(1), math.log(2.0) / half_lives))
        self.register_buffer("decay_rates", rates, persistent=True)

        nn.init.normal_(self.source_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.target_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.target_embed.weight[num_skills].zero_()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).unsqueeze(-1).to(values.dtype)
        count = weights.sum(dim=-3).clamp_min(1.0)
        return (values * weights).sum(dim=-3) / count

    def forward(
        self,
        skill_ids: torch.Tensor,
        skill_mask: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a pre-response cross-effect score at every sequence position."""
        if skill_ids.shape != skill_mask.shape or skill_ids.shape[:2] != mask.shape:
            raise ValueError("skill tensors must have shape [B, L, K]")
        if responses.shape != mask.shape or times.shape != mask.shape:
            raise ValueError("responses, times, and mask must have shape [B, L]")

        valid_skills = skill_mask.bool() & mask.bool().unsqueeze(-1)
        safe_skill_ids = skill_ids.masked_fill(~valid_skills, 0)
        interaction_ids = safe_skill_ids + responses.unsqueeze(-1) * self.num_skills

        source = self.source_embed(interaction_ids).view(
            *skill_ids.shape, self.num_scales, self.rank
        )
        target = self.target_embed(safe_skill_ids).view(
            *skill_ids.shape, self.num_scales, self.rank
        )
        source = self._masked_mean(source, valid_skills)
        target = self._masked_mean(target, valid_skills)

        gaps = torch.zeros_like(times, dtype=source.dtype)
        gaps[:, 1:] = (times[:, 1:] - times[:, :-1]).to(source.dtype).clamp_min(0.0)
        rates = self.decay_rates.to(dtype=source.dtype).view(1, 1, -1, 1)
        decay = torch.exp(-gaps.unsqueeze(-1).unsqueeze(-1) * rates)
        decay = decay.expand_as(source)
        valid = mask.bool().unsqueeze(-1).unsqueeze(-1)
        decay = torch.where(valid, decay, torch.ones_like(decay))
        source = torch.where(valid, source, torch.zeros_like(source))

        pre_decay_state = diagonal_affine_exclusive_scan(decay, source)
        pre_state = decay * pre_decay_state
        scale_scores = (pre_state * target).sum(dim=-1) / math.sqrt(self.rank)
        scores = scale_scores.sum(dim=-1)
        scores = scores.masked_fill(~mask.bool(), 0.0)
        return torch.tanh(self.logit_scale) * scores


__all__ = ["SkillCrossEffect", "diagonal_affine_exclusive_scan"]
