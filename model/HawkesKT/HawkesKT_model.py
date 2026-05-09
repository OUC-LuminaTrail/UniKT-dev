import math

import torch
from torch import nn


class HawkesKT(nn.Module):
    """HawkesKT model (WSDM 2021).

    Uses Hawkes Process to model temporal cross-effects between skills,
    where practicing skill A excites mastery of skill B with time-decaying effects.

    Args:
        num_c: Number of skills
        num_q: Number of questions
        emb_size: Embedding dimension
        time_log: Log base for time interval transformation
    """

    def __init__(self, num_c: int, num_q: int, emb_size: int, time_log: float):
        super().__init__()
        self.skill_num = num_c
        self.problem_num = num_q
        self.emb_size = emb_size
        self.time_log = time_log

        self.problem_base = nn.Embedding(self.problem_num, 1)
        self.skill_base = nn.Embedding(self.skill_num, 1)

        self.alpha_inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(
        self,
        skill_seq: torch.Tensor,
        problem_seq: torch.Tensor,
        time_seq: torch.Tensor,
        label_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            skill_seq: Skill IDs, shape [B, S]
            problem_seq: Question IDs, shape [B, S]
            time_seq: Timestamps, shape [B, S]
            label_seq: Correctness labels, shape [B, S]

        Returns:
            Predictions, shape [B, S]. prediction[:, t] predicts label[:, t].
        """
        mask_labels = label_seq * (label_seq > -1).long()
        inters = skill_seq + mask_labels * self.skill_num

        alpha_src_emb = self.alpha_inter_embeddings(inters)
        alpha_target_emb = self.alpha_skill_embeddings(skill_seq)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))

        beta_src_emb = self.beta_inter_embeddings(inters)
        beta_target_emb = self.beta_skill_embeddings(skill_seq)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))
        betas = torch.clamp(betas + 1, min=0, max=10)

        delta_t = (time_seq[:, :, None] - time_seq[:, None, :]).abs().double()
        delta_t = torch.log(delta_t + 1e-10) / math.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t)

        seq_len = skill_seq.shape[1]
        lower_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=skill_seq.device
        ).tril(diagonal=0)
        sum_t = cross_effects.masked_fill(lower_mask.unsqueeze(0), 0).sum(-2)

        problem_bias = self.problem_base(problem_seq).squeeze(dim=-1)
        skill_bias = self.skill_base(skill_seq).squeeze(dim=-1)

        return torch.sigmoid((problem_bias + skill_bias + sum_t).float())
