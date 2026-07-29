"""ReKTP: private question/KC scans with a global Mamba history encoder."""

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn

from model.ReKTP.segmented_scan import segmented_block_affine_exclusive_scan


class GlobalMambaBlock(nn.Module):
    """Mamba block with the residual and normalization used by this project."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.mamba(x)))


class ReKTP(nn.Module):
    """Question-level KT with decoupled per-KC storage and global interaction."""

    def __init__(
        self,
        data_metadata: dict,
        question_skill_ids: np.ndarray | torch.Tensor,
        question_skill_mask: np.ndarray | torch.Tensor,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_gap_bins: int = 16,
        residual_scale: float = 0.1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("ReKTP hidden_dim must be divisible by 2")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.num_questions = int(data_metadata["num_questions"])
        self.num_skills = int(data_metadata["num_skills"])
        self.hidden_dim = hidden_dim
        self.max_gap_bins = max_gap_bins
        self.state_block_size = 2
        self.num_state_blocks = hidden_dim // self.state_block_size
        self.residual_scale = residual_scale

        skill_ids = torch.as_tensor(question_skill_ids, dtype=torch.long)
        skill_mask = torch.as_tensor(question_skill_mask, dtype=torch.bool)
        expected_questions = self.num_questions
        if skill_ids.ndim != 2 or skill_ids.size(0) != expected_questions:
            raise ValueError("question_skill_ids must have shape [num_questions, K]")
        if skill_mask.shape != skill_ids.shape:
            raise ValueError("question_skill_mask must match question_skill_ids")
        self.register_buffer("question_skill_ids", skill_ids, persistent=True)
        self.register_buffer("question_skill_mask", skill_mask, persistent=True)

        self.question_embed = nn.Embedding(self.num_questions, hidden_dim)
        self.skill_embed = nn.Embedding(
            self.num_skills + 1, hidden_dim, padding_idx=self.num_skills
        )
        self.answer_embed = nn.Embedding(2, hidden_dim)
        self.gap_embed = nn.Embedding(max_gap_bins, hidden_dim)
        self.question_diff = nn.Embedding(self.num_questions, 1)
        self.skill_change = nn.Embedding(
            self.num_skills + 1, hidden_dim, padding_idx=self.num_skills
        )

        self.local_residual = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.local_write = nn.Linear(hidden_dim, hidden_dim)
        self.local_init = nn.Linear(hidden_dim, hidden_dim)
        self.local_decay = nn.Linear(hidden_dim, hidden_dim)
        self.local_readout = nn.Linear(3 * hidden_dim, 1)
        self.local_global_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.question_residual = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.question_write = nn.Linear(hidden_dim, hidden_dim)
        self.question_init = nn.Linear(hidden_dim, hidden_dim)
        self.question_decay = nn.Linear(hidden_dim, hidden_dim)

        self.global_blocks = nn.ModuleList(
            GlobalMambaBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        )
        self.global_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.question_diff.weight)
        for layer in (
            self.local_residual,
            self.local_write,
            self.local_readout,
            self.local_global_film,
            self.question_residual,
            self.question_write,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.local_decay.weight)
        nn.init.constant_(self.local_decay.bias, -4.0)
        nn.init.zeros_(self.question_decay.weight)
        nn.init.constant_(self.question_decay.bias, -4.0)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        count = weights.sum(dim=-2).clamp_min(1.0)
        return (values * weights).sum(dim=-2) / count

    def _question_conditioned_local_readout(
        self,
        local_state: torch.Tensor,
        skill_ids: torch.Tensor,
        readout_mask: torch.Tensor,
        questions: torch.Tensor,
    ) -> torch.Tensor:
        """Read decoupled KC states with current-question dependent weights."""
        skill_embedding = self.skill_embed(skill_ids)
        question_embedding = self.question_embed(questions).unsqueeze(-2)
        question_embedding = question_embedding.expand_as(local_state)
        score_input = torch.cat(
            (local_state, skill_embedding, question_embedding),
            dim=-1,
        )
        scores = self.local_readout(score_input).squeeze(-1)
        masked_scores = scores.masked_fill(
            ~readout_mask,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(masked_scores, dim=-1)
        weights = torch.where(readout_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (local_state * weights.unsqueeze(-1)).sum(dim=-2)

    def _condition_local_input(
        self,
        local_input: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.local_global_film(global_context).chunk(2, dim=-1)
        return local_input * (1.0 + gamma) + beta

    def _block_affine_transition(
        self,
        event_input: torch.Tensor,
        residual_layer: nn.Linear,
        write_layer: nn.Linear,
        decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = event_input.shape[:-1]
        block_shape = (*leading_shape, self.num_state_blocks)
        raw_residual = residual_layer(event_input).reshape(
            *block_shape, self.state_block_size, self.state_block_size
        )
        residual_norm = (
            raw_residual.square().sum(dim=(-2, -1), keepdim=True).add(1e-12).sqrt()
        )
        residual = self.residual_scale * raw_residual / (1.0 + residual_norm)

        identity = torch.eye(
            self.state_block_size,
            device=event_input.device,
            dtype=event_input.dtype,
        )
        decay_blocks = decay.reshape(*block_shape, self.state_block_size)
        transition = (identity + residual) * decay_blocks.unsqueeze(-2)
        bias = torch.tanh(write_layer(event_input)).reshape(
            *block_shape, self.state_block_size
        )
        return transition, bias

    def _pack_kc_occurrences(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, seq_len = questions.shape
        skill_ids = self.question_skill_ids[questions]
        skill_mask = self.question_skill_mask[questions]
        occurrence_mask = skill_mask & mask.unsqueeze(-1)
        max_skills = skill_ids.size(-1)

        times = torch.arange(seq_len, device=questions.device).view(1, seq_len, 1)
        times = times.expand(batch_size, seq_len, max_skills)
        question_occ = questions.unsqueeze(-1).expand_as(skill_ids)
        response_occ = responses.unsqueeze(-1).expand_as(skill_ids)

        flat_skill = skill_ids.flatten(1)
        flat_time = times.flatten(1)
        flat_question = question_occ.flatten(1)
        flat_response = response_occ.flatten(1)
        flat_valid = occurrence_mask.flatten(1)

        invalid_key = (self.num_skills + 1) * (seq_len + 1)
        sort_key = flat_skill * (seq_len + 1) + flat_time
        sort_key = torch.where(flat_valid, sort_key, invalid_key + flat_time)
        order = torch.argsort(sort_key, dim=1, stable=True)

        def gather(values: torch.Tensor) -> torch.Tensor:
            return torch.gather(values, 1, order)

        return (
            gather(flat_skill),
            gather(flat_time),
            gather(flat_question),
            gather(flat_response),
            gather(flat_valid),
            order,
            occurrence_mask,
        )

    def _local_pre_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = questions.shape
        (
            packed_skill,
            packed_time,
            packed_question,
            packed_response,
            packed_valid,
            order,
            occurrence_mask,
        ) = self._pack_kc_occurrences(questions, responses, mask)

        previous_time = torch.zeros_like(packed_time)
        previous_time[:, 1:] = packed_time[:, :-1]
        same_segment = torch.zeros_like(packed_valid)
        same_segment[:, 1:] = (
            packed_valid[:, 1:]
            & packed_valid[:, :-1]
            & (packed_skill[:, 1:] == packed_skill[:, :-1])
        )
        gap = torch.where(same_segment, packed_time - previous_time, packed_time + 1)
        gap_bucket = torch.floor(torch.log2(gap.clamp_min(1).float())).long()
        gap_bucket = gap_bucket.clamp_max(self.max_gap_bins - 1)

        skill_embedding = self.skill_embed(packed_skill)
        local_input = (
            self.question_embed(packed_question)
            + skill_embedding
            + self.question_diff(packed_question) * self.skill_change(packed_skill)
            + self.answer_embed(packed_response)
        )
        if global_context is None:
            packed_global = torch.zeros_like(local_input)
        else:
            expected_context_shape = (*questions.shape, self.hidden_dim)
            if global_context.shape != expected_context_shape:
                raise ValueError(
                    "global_context must have shape [batch_size, seq_len, hidden_dim]"
                )
            max_skills = occurrence_mask.size(-1)
            flat_global = (
                global_context.unsqueeze(-2)
                .expand(batch_size, seq_len, max_skills, self.hidden_dim)
                .flatten(1, 2)
            )
            packed_global = torch.gather(
                flat_global,
                1,
                order.unsqueeze(-1).expand_as(local_input),
            )
        local_input = self._condition_local_input(local_input, packed_global)
        decay = torch.exp(
            -torch.nn.functional.softplus(self.local_decay(self.gap_embed(gap_bucket)))
        )
        valid_3d = packed_valid.unsqueeze(-1)
        decay = torch.where(valid_3d, decay, torch.ones_like(decay))
        initial_state = torch.tanh(self.local_init(skill_embedding))
        transition, bias = self._block_affine_transition(
            local_input,
            self.local_residual,
            self.local_write,
            decay,
        )
        initial_blocks = initial_state.reshape(
            *initial_state.shape[:-1],
            self.num_state_blocks,
            self.state_block_size,
        )

        packed_pre_decay_blocks = segmented_block_affine_exclusive_scan(
            transition,
            bias,
            packed_skill,
            packed_valid,
            initial_blocks,
        )
        packed_pre_decay = packed_pre_decay_blocks.flatten(-2)
        packed_state = decay * packed_pre_decay
        unpacked_state = torch.zeros_like(packed_state)
        scatter_index = order.unsqueeze(-1).expand_as(packed_state)
        unpacked_state.scatter_(1, scatter_index, packed_state)

        max_skills = occurrence_mask.size(-1)
        local_state = unpacked_state.view(
            batch_size, seq_len, max_skills, self.hidden_dim
        )
        skill_ids = self.question_skill_ids[questions]
        pooled_state = self._question_conditioned_local_readout(
            local_state,
            skill_ids,
            occurrence_mask,
            questions,
        )
        return pooled_state, local_state

    def _question_pre_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
        event_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exclusive state of each question at every raw position."""
        _, seq_len = questions.shape
        times = torch.arange(seq_len, device=questions.device).unsqueeze(0)
        times = times.expand_as(questions)

        invalid_key = (self.num_questions + 1) * (seq_len + 1)
        sort_key = questions * (seq_len + 1) + times
        sort_key = torch.where(mask, sort_key, invalid_key + times)
        order = torch.argsort(sort_key, dim=1, stable=True)

        packed_question = torch.gather(questions, 1, order)
        packed_time = torch.gather(times, 1, order)
        packed_response = torch.gather(responses, 1, order)
        packed_valid = torch.gather(mask, 1, order)
        gather_index = order.unsqueeze(-1).expand_as(event_embeddings)
        packed_event = torch.gather(event_embeddings, 1, gather_index)

        previous_time = torch.zeros_like(packed_time)
        previous_time[:, 1:] = packed_time[:, :-1]
        same_segment = torch.zeros_like(packed_valid)
        same_segment[:, 1:] = (
            packed_valid[:, 1:]
            & packed_valid[:, :-1]
            & (packed_question[:, 1:] == packed_question[:, :-1])
        )
        gap = torch.where(same_segment, packed_time - previous_time, packed_time + 1)
        gap_bucket = torch.floor(torch.log2(gap.clamp_min(1).float())).long()
        gap_bucket = gap_bucket.clamp_max(self.max_gap_bins - 1)

        question_input = packed_event + self.answer_embed(packed_response)
        decay = torch.exp(
            -torch.nn.functional.softplus(
                self.question_decay(self.gap_embed(gap_bucket))
            )
        )
        valid_3d = packed_valid.unsqueeze(-1)
        decay = torch.where(valid_3d, decay, torch.ones_like(decay))
        initial_state = torch.tanh(
            self.question_init(self.question_embed(packed_question))
        )
        transition, bias = self._block_affine_transition(
            question_input,
            self.question_residual,
            self.question_write,
            decay,
        )
        initial_blocks = initial_state.reshape(
            *initial_state.shape[:-1],
            self.num_state_blocks,
            self.state_block_size,
        )

        packed_pre_decay_blocks = segmented_block_affine_exclusive_scan(
            transition,
            bias,
            packed_question,
            packed_valid,
            initial_blocks,
        )
        packed_pre_decay = packed_pre_decay_blocks.flatten(-2)
        packed_state = decay * packed_pre_decay
        question_state = torch.zeros_like(packed_state)
        question_state.scatter_(1, gather_index, packed_state)
        return question_state

    def _event_embeddings(
        self, questions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skill_ids = self.question_skill_ids[questions]
        skill_mask = self.question_skill_mask[questions]
        pooled_skill = self._masked_mean(self.skill_embed(skill_ids), skill_mask)
        pooled_change = self._masked_mean(self.skill_change(skill_ids), skill_mask)
        event = (
            self.question_embed(questions)
            + pooled_skill
            + self.question_diff(questions) * pooled_change
        )
        return event, skill_mask

    def _global_history_states(
        self,
        event_embedding: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        global_state = event_embedding + self.answer_embed(responses)
        global_state = global_state.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.global_blocks:
            global_state = block(global_state)
        return self.global_norm(global_state + self.global_ffn(global_state))

    def forward(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return next-item logits where output[t] predicts response[t+1]."""
        mask = mask.bool()
        event_embedding, _ = self._event_embeddings(questions)
        global_state = self._global_history_states(event_embedding, responses, mask)
        local_pre_state, _ = self._local_pre_states(
            questions,
            responses,
            mask,
            global_state,
        )
        question_pre_state = self._question_pre_states(
            questions, responses, mask, event_embedding
        )

        features = torch.cat(
            [
                global_state[:, :-1],
                question_pre_state[:, 1:],
                local_pre_state[:, 1:],
                event_embedding[:, 1:],
            ],
            dim=-1,
        )
        logits = self.out(features).squeeze(-1)
        logits = logits.masked_fill(~mask[:, 1:], 0.0)
        return torch.cat([logits, logits.new_zeros(logits.size(0), 1)], dim=1)


__all__ = ["ReKTP"]
