"""ReKTP: private KC scans with a swappable global history encoder.

The global encoder is selected by ``encoder_type`` (``mamba`` / ``lstm`` /
``transformer`` / ``qrnn``), which keeps the rest of the model fixed across
those choices.
"""

from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model.ReKTP.segmented_scan import segmented_block_affine_exclusive_scan

# Global sequence-encoder types selectable via ``encoder_type``.
GLOBAL_ENCODER_TYPES = ("mamba", "lstm", "transformer", "qrnn", "conv")

# Question-derived tensors computed once per forward and shared across the
# sub-methods that would otherwise re-gather and re-embed them.
_QuestionFeatures = namedtuple(
    "_QuestionFeatures",
    [
        "skill_ids",
        "skill_mask",
        "question_vector",
        "skill_embedding",
        "skill_change",
    ],
)


class GlobalMambaBlock(nn.Module):
    """Mamba-2 block with the residual and normalization used by this project."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        # Imported lazily so the LSTM/Transformer variants run without mamba_ssm.
        from mamba_ssm import Mamba2

        # Mamba-2's conv1d channels are d_ssm + 2 * d_state and must be a
        # multiple of 8; with d_ssm == d_inner that constrains d_state.
        d_inner = expand * d_model
        if (d_inner + 2 * d_state) % 8 != 0:
            raise ValueError(
                "Mamba-2 requires expand * d_model + 2 * d_state to be a "
                f"multiple of 8, got d_model={d_model}, expand={expand}, "
                f"d_state={d_state}"
            )
        # d_ssm must be divisible by headdim; pick the largest power of two
        # no larger than 64 so small hidden sizes still work.
        headdim = min(64, d_inner)
        while d_inner % headdim != 0:
            headdim //= 2
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.norm(x + self.dropout(self.mamba(x)))


class GlobalLSTMBlock(nn.Module):
    """LSTM block with the same residual shell as the Mamba block."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Input padding is pre-zeroed; a unidirectional pass keeps trailing
        # padding from reaching earlier valid positions.
        out, _ = self.lstm(x)
        return self.norm(x + self.dropout(out))


class GlobalQRNNBlock(nn.Module):
    """QRNN-fo block (Bradbury et al., 2017) with the same residual shell.

    A causal windowed linear layer produces the candidate ``z``, forget ``f``,
    and output ``o`` gates; the recurrent pooling
    ``c_t = f_t * z_t + (1 - f_t) * c_{t-1}`` runs in parallel (Triton on CUDA,
    log-depth scan on CPU), and the block emits ``h_t = o_t * c_t``. Matches
    the reference implementation (salesforce/pytorch-qrnn): window 1 or 2 with
    zero-padded history at the sequence start, ``tanh`` on ``z`` and
    ``sigmoid`` on ``f``/``o``. ``zoneout`` is not ported; the residual shell
    already applies ``dropout``.
    """

    def __init__(self, d_model: int, dropout: float, window: int = 2):
        super().__init__()
        if window not in (1, 2):
            raise ValueError(
                "QRNN window must be 1 or 2, matching the reference implementation"
            )
        self.window = window
        # One large matmul over the concatenated window beats several small ones.
        self.linear = nn.Linear(window * d_model, 3 * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        from model.ReKTP.qrnn_scan import qrnn_pool

        # Input padding is pre-zeroed; the pooling is causal, so trailing
        # padding cannot reach earlier valid positions (same as the LSTM).
        if self.window == 2:
            # Pack [x_t, x_{t-1}] into one buffer: current frame in the head
            # half, causally-shifted frame in the tail half.
            b_sz, seq_len, hidden = x.shape
            source = x.new_empty(b_sz, seq_len, 2 * hidden)
            source[:, :, :hidden] = x
            source[:, 1:, hidden:] = x[:, :-1]
            source[:, 0, hidden:] = 0
        else:
            source = x
        y = self.linear(source)
        z, f, o = y.chunk(3, dim=-1)
        # qrnn_pool applies tanh/sigmoid internally (fused into the scan kernel),
        # so the raw gated-linear outputs feed it directly.
        cell = qrnn_pool(z, f)
        hidden = torch.sigmoid(o) * cell
        return self.norm(x + self.dropout(hidden))


class GlobalTransformerBlock(nn.Module):
    """Causal self-attention block with the same residual shell.

    Requires trailing (right-side) padding, which the sequence builder
    guarantees. ``mask`` is accepted for interface parity but unused: under
    right padding the causal mask alone already hides every padding key from
    every valid query.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Projections come from ``self.attn`` so parameter names and their
        # initialization are unchanged; only the attention call is fused.
        # No key-padding mask: padding is trailing, so every padding key sits at
        # an index above any valid query and the causal mask already hides it.
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.n_heads
        projected = torch.nn.functional.linear(
            x, self.attn.in_proj_weight, self.attn.in_proj_bias
        )
        query, key, value = projected.chunk(3, dim=-1)
        head_shape = (batch_size, seq_len, self.n_heads, head_dim)
        query, key, value = (
            tensor.view(head_shape).transpose(1, 2) for tensor in (query, key, value)
        )
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        out = self.attn.out_proj(attended)
        return self.norm(x + self.dropout(out))


class GlobalConvBlock(nn.Module):
    """Causal depthwise-separable conv block with the same residual shell.

    A left-padded depthwise conv mixes the last ``kernel_size`` positions per
    channel in parallel, a pointwise conv mixes channels, and the residual
    shell matches the other global blocks. Fully parallel across time (no
    sequential scan) and O(T) memory; stacking blocks with growing dilation
    widens the receptive field.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("conv kernel_size must be >= 1")
        # Causal: left-pad only, so each position looks back
        # (kernel_size - 1) * dilation steps and never at the future.
        self.pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, dilation=dilation, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)  # [B, d, N]
        y = torch.nn.functional.pad(y, (self.pad, 0))
        y = self.depthwise(y)
        y = self.pointwise(y)
        y = torch.nn.functional.gelu(y).transpose(1, 2)  # [B, N, d]
        return self.norm(residual + self.dropout(y))


def build_global_block(
    encoder_type: str,
    d_model: int,
    dropout: float,
    *,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    n_heads: int = 8,
    window: int = 2,
    kernel_size: int = 3,
    dilation: int = 1,
) -> nn.Module:
    """Construct one global sequence-mixing block for the requested encoder type."""
    if encoder_type == "mamba":
        return GlobalMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
    if encoder_type == "lstm":
        return GlobalLSTMBlock(d_model=d_model, dropout=dropout)
    if encoder_type == "qrnn":
        return GlobalQRNNBlock(d_model=d_model, dropout=dropout, window=window)
    if encoder_type == "conv":
        return GlobalConvBlock(
            d_model=d_model,
            dropout=dropout,
            kernel_size=kernel_size,
            dilation=dilation,
        )
    if encoder_type == "transformer":
        return GlobalTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
    raise ValueError(
        f"encoder_type must be one of {GLOBAL_ENCODER_TYPES}, got {encoder_type!r}"
    )


class ReKTP(nn.Module):
    """Question-level KT with decoupled per-KC storage and global interaction.

    Readout concatenates ``[global_state, question_static_state,
    local_pre_state, event_embedding]``.
    """

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
        max_gap_bins: int = 8,
        residual_scale: float = 0.1,
        dropout: float = 0.2,
        encoder_type: str = "mamba",
        n_heads: int = 8,
        window: int = 2,
        conv_kernel_size: int = 3,
        conv_dilation_base: int = 2,
        question_embed_dim: int | None = None,
    ):
        super().__init__()
        if encoder_type not in GLOBAL_ENCODER_TYPES:
            raise ValueError(
                f"encoder_type must be one of {GLOBAL_ENCODER_TYPES}, "
                f"got {encoder_type!r}"
            )
        if hidden_dim % 2 != 0:
            raise ValueError("ReKTP hidden_dim must be divisible by 2")
        if encoder_type == "transformer" and hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads}) "
                "for the transformer encoder"
            )
        if max_gap_bins < 1:
            raise ValueError("max_gap_bins must be at least 1")
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

        # Intrinsic width of the per-question embedding. ``0`` drops the
        # pathway, leaving question identity to ``question_diff`` and the KC side.
        if question_embed_dim is None:
            question_embed_dim = hidden_dim
        if question_embed_dim < 0:
            raise ValueError("question_embed_dim must be non-negative")
        if question_embed_dim == 0:
            self.question_embed = None
            self.question_embed_proj = None
        else:
            self.question_embed = nn.Embedding(self.num_questions, question_embed_dim)
            # Below full width a shared projection lifts the rows back to
            # ``hidden_dim``; at full width the projection is skipped.
            self.question_embed_proj = (
                None
                if question_embed_dim == hidden_dim
                else nn.Linear(question_embed_dim, hidden_dim, bias=False)
            )
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
        # Static question-feature pathway, modulated by sequence position.
        self.question_init = nn.Linear(hidden_dim, hidden_dim)
        self.question_decay = nn.Linear(hidden_dim, hidden_dim)
        self.global_blocks = nn.ModuleList(
            build_global_block(
                encoder_type,
                d_model=hidden_dim,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                n_heads=n_heads,
                window=window,
                kernel_size=conv_kernel_size,
                dilation=conv_dilation_base**i,
            )
            for i in range(n_blocks)
        )
        self.global_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.global_norm = nn.LayerNorm(hidden_dim)
        # IRT prediction head: logit = a·(θ−β), where θ is read from the
        # 4-way readout features by ``ability_head`` and β is the shared
        # ``question_diff`` of the predicted (next) question. Zero-initialized
        # so the model starts at chance (logit 0); the head is a pure
        # 2PL-style IRT decomposition, interpretable as ability minus
        # difficulty scaled by the learned discrimination ``a``.
        self.ability_head = nn.Linear(4 * hidden_dim, 1)
        nn.init.zeros_(self.ability_head.weight)
        nn.init.zeros_(self.ability_head.bias)
        self.irt_disc = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.question_diff.weight)
        for layer in (
            self.local_residual,
            self.local_write,
            self.local_readout,
            self.local_global_film,
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

    def _question_vector(self, questions: torch.Tensor) -> torch.Tensor:
        """Return the per-question vector at ``hidden_dim`` width.

        ``question_embed_dim`` sets the intrinsic width: 0 removes the pathway
        (zeros), a value below ``hidden_dim`` is lifted by a shared projection,
        and ``hidden_dim`` uses the embedding directly.
        """
        if self.question_embed is None:
            return torch.zeros(
                *questions.shape,
                self.hidden_dim,
                device=questions.device,
                dtype=self.skill_embed.weight.dtype,
            )
        vector = self.question_embed(questions)
        if self.question_embed_proj is not None:
            vector = self.question_embed_proj(vector)
        return vector

    def _resolve_question_features(self, questions: torch.Tensor) -> _QuestionFeatures:
        """Gather the question-derived tensors once for reuse across forward.

        ``question_skill_ids[questions]``, the ``skill_embed`` / ``skill_change``
        lookups, and the ``_question_vector`` projection are each needed in
        several sub-methods. Computing them here and threading the result
        avoids repeating the gathers and the large ``[B, N, K, hidden]``
        embedding tensors.
        """
        skill_ids = self.question_skill_ids[questions]
        return _QuestionFeatures(
            skill_ids=skill_ids,
            skill_mask=self.question_skill_mask[questions],
            question_vector=self._question_vector(questions),
            skill_embedding=self.skill_embed(skill_ids),
            skill_change=self.skill_change(skill_ids),
        )

    def _question_conditioned_local_readout(
        self,
        local_state: torch.Tensor,
        skill_ids: torch.Tensor,
        readout_mask: torch.Tensor,
        questions: torch.Tensor,
        skill_embedding: torch.Tensor | None = None,
        question_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read decoupled KC states with current-question dependent weights."""
        if skill_embedding is None:
            skill_embedding = self.skill_embed(skill_ids)
        if question_vector is None:
            question_vector = self._question_vector(questions)
        question_embedding = question_vector.unsqueeze(-2)
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
        times: torch.Tensor,
        mask: torch.Tensor,
        skill_ids: torch.Tensor | None = None,
        skill_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, seq_len = questions.shape
        if skill_ids is None:
            skill_ids = self.question_skill_ids[questions]
        if skill_mask is None:
            skill_mask = self.question_skill_mask[questions]
        occurrence_mask = skill_mask & mask.unsqueeze(-1)
        max_skills = skill_ids.size(-1)

        # Real elapsed seconds (or position indices when unavailable); only
        # differences within a sequence matter.
        times = times.reshape(batch_size, seq_len, 1).expand(
            batch_size, seq_len, max_skills
        )
        question_occ = questions.unsqueeze(-1).expand_as(skill_ids)
        response_occ = responses.unsqueeze(-1).expand_as(skill_ids)

        flat_skill = skill_ids.flatten(1)
        flat_time = times.flatten(1)
        flat_question = question_occ.flatten(1)
        flat_response = response_occ.flatten(1)
        flat_valid = occurrence_mask.flatten(1)

        # Sort by (skill, position): the source data is already chronological, so
        # position order equals time order inside a skill segment. Real seconds
        # cannot seed the sort key (they exceed the skill stride); they are used
        # only for the gap computation below.
        positions = (
            torch.arange(seq_len, device=questions.device)
            .view(1, seq_len, 1)
            .expand(batch_size, seq_len, max_skills)
        )
        flat_pos = positions.flatten(1)
        invalid_key = (self.num_skills + 1) * (seq_len + 1)
        sort_key = flat_skill * (seq_len + 1) + flat_pos
        sort_key = torch.where(flat_valid, sort_key, invalid_key + flat_pos)
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
        times: torch.Tensor,
        mask: torch.Tensor,
        global_context: torch.Tensor,
        q_features: _QuestionFeatures | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = questions.shape
        skill_ids = (
            q_features.skill_ids
            if q_features is not None
            else self.question_skill_ids[questions]
        )
        skill_mask = (
            q_features.skill_mask
            if q_features is not None
            else self.question_skill_mask[questions]
        )
        (
            packed_skill,
            packed_time,
            packed_question,
            packed_response,
            packed_valid,
            order,
            occurrence_mask,
        ) = self._pack_kc_occurrences(
            questions, responses, times, mask, skill_ids, skill_mask
        )

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
            self._question_vector(packed_question)
            + skill_embedding
            + self.question_diff(packed_question) * self.skill_change(packed_skill)
            + self.answer_embed(packed_response)
        )
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
        if q_features is not None:
            readout_skill_embedding = q_features.skill_embedding
            readout_question_vector = q_features.question_vector
        else:
            readout_skill_embedding = self.skill_embed(skill_ids)
            readout_question_vector = self._question_vector(questions)
        return self._question_conditioned_local_readout(
            local_state,
            skill_ids,
            occurrence_mask,
            questions,
            skill_embedding=readout_skill_embedding,
            question_vector=readout_question_vector,
        )

    def _question_static_states(
        self,
        questions: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        question_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the elapsed-time-modulated static question feature.

        ``s_t = exp(-softplus(W_d . e_gap[b_t])) * tanh(W_init . E_q)`` with
        ``b_t = floor(log2(t + 1))``, where ``t`` is the elapsed time in
        seconds since the sequence start. Padding positions are zeroed.
        """
        # Elapsed seconds since the sequence start, floored to avoid a zero gap.
        gap_bucket = torch.floor(torch.log2((times + 1).float())).long()
        gap_bucket = gap_bucket.clamp_max(self.max_gap_bins - 1).expand_as(questions)

        decay = torch.exp(
            -torch.nn.functional.softplus(
                self.question_decay(self.gap_embed(gap_bucket))
            )
        )
        if question_vector is None:
            question_vector = self._question_vector(questions)
        static_state = torch.tanh(self.question_init(question_vector))
        state = decay * static_state
        return state.masked_fill(~mask.unsqueeze(-1), 0.0)

    def _event_embeddings(
        self,
        questions: torch.Tensor,
        q_features: _QuestionFeatures | None = None,
    ) -> torch.Tensor:
        if q_features is None:
            q_features = self._resolve_question_features(questions)
        pooled_skill = self._masked_mean(
            q_features.skill_embedding, q_features.skill_mask
        )
        pooled_change = self._masked_mean(
            q_features.skill_change, q_features.skill_mask
        )
        return (
            q_features.question_vector
            + pooled_skill
            + self.question_diff(questions) * pooled_change
        )

    def _global_history_states(
        self,
        event_embedding: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        global_state = event_embedding + self.answer_embed(responses)
        global_state = global_state.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.global_blocks:
            global_state = block(global_state, mask)
        return self.global_norm(global_state + self.global_ffn(global_state))

    def _irt_term(
        self, features: torch.Tensor, next_questions: torch.Tensor
    ) -> torch.Tensor:
        """IRT logit term ``a·(θ−β)`` — the sole prediction head.

        ``θ`` comes from ``ability_head`` over the readout features and ``β``
        from the shared ``question_diff`` embedding of the predicted question.
        """
        theta = self.ability_head(features).squeeze(-1)
        beta = self.question_diff(next_questions).squeeze(-1)
        return self.irt_disc * (theta - beta)

    def forward(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return next-item logits where output[t] predicts response[t+1].

        ``times`` holds per-position interaction times in seconds; only
        within-sequence differences are used, so any consistent offset works.
        """
        mask = mask.bool()
        if times.shape != questions.shape:
            raise ValueError(
                "times must have shape [batch_size, seq_len] matching questions"
            )
        # Interactions are measured relative to the first valid interaction;
        # padding is pinned to zero so log-bucketing never sees negatives.
        first_time = times.masked_fill(~mask, torch.inf).min(dim=1, keepdim=True).values
        times = times - first_time
        times = times.masked_fill(~mask, 0.0)
        q_features = self._resolve_question_features(questions)
        event_embedding = self._event_embeddings(questions, q_features)
        global_state = self._global_history_states(event_embedding, responses, mask)
        local_pre_state = self._local_pre_states(
            questions,
            responses,
            times,
            mask,
            global_state,
            q_features,
        )
        question_static_state = self._question_static_states(
            questions, times, mask, q_features.question_vector
        )

        features = torch.cat(
            [
                global_state[:, :-1],
                question_static_state[:, 1:],
                local_pre_state[:, 1:],
                event_embedding[:, 1:],
            ],
            dim=-1,
        )
        next_questions = questions[:, 1:]
        logits = self._irt_term(features, next_questions)
        logits = logits.masked_fill(~mask[:, 1:], 0.0)
        return torch.cat([logits, logits.new_zeros(logits.size(0), 1)], dim=1)


__all__ = ["ReKTP"]
