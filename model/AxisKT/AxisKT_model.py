"""AxisKT: private KC scans with a stacked causal-conv global encoder.

The global encoder is a stack of ``n_blocks`` causal depthwise-separable conv
blocks whose dilation grows as ``conv_dilation_base**i``; the rest of the
model is the same question-level KT pipeline.

Ablation: ``use_global`` toggles the global causal dilated-conv branch
(:meth:`AxisKT._global_history_states`), ``use_local`` toggles the local per-KC
affine recursion branch (:meth:`AxisKT._local_pre_states`), and
``use_forgetting`` toggles only the local time-decay transition. An ablated
branch is skipped entirely in the forward pass and feeds all-zero features to
the readout, so its parameters never receive a gradient and stay inert; the
architecture, parameter count, and shared pathway (event embeddings, IRT head)
are unchanged, isolating each branch's contribution.
"""

from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model.AxisKT.triton_scan import segmented_scalar_affine_exclusive_scan

# Question-derived tensors computed once per forward and shared across the
# sub-methods that would otherwise re-gather them.
_QuestionFeatures = namedtuple(
    "_QuestionFeatures",
    [
        "skill_ids",
        "skill_mask",
        "question_vector",
    ],
)


class GlobalConvBlock(nn.Module):
    """Causal depthwise-separable conv block with a residual shell.

    A left-padded depthwise conv mixes the last ``kernel_size`` positions per
    channel in parallel, a pointwise conv mixes channels, and the residual
    shell applies normalization and dropout. Fully parallel across time (no
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)  # [B, d, N]
        y = torch.nn.functional.pad(y, (self.pad, 0))
        y = self.depthwise(y)
        y = self.pointwise(y)
        y = torch.nn.functional.gelu(y).transpose(1, 2)  # [B, N, d]
        return self.norm(residual + self.dropout(y))


class AxisKT(nn.Module):
    """Question-level KT with decoupled per-KC storage and global interaction.

    Readout concatenates ``[global_state, local_pre_state, event_embedding]``.
    ``use_global`` / ``use_local`` ablate the global dilated-conv branch and
    the local affine-recursion branch respectively. ``use_forgetting=False``
    keeps the local per-KC writes and scan but replaces the learned temporal
    decay with an identity transition. An ablated branch emits all-zero
    features of the same shape, keeping the rest of the model intact.
    """

    def __init__(
        self,
        data_metadata: dict,
        question_skill_ids: np.ndarray | torch.Tensor,
        question_skill_mask: np.ndarray | torch.Tensor,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        max_gap_bins: int = 8,
        dropout: float = 0.2,
        conv_kernel_size: int = 3,
        conv_dilation_base: int = 2,
        question_embed_dim: int | None = None,
        use_global: bool = True,
        use_local: bool = True,
        use_forgetting: bool = True,
    ):
        super().__init__()
        if max_gap_bins < 1:
            raise ValueError("max_gap_bins must be at least 1")
        self.num_questions = int(data_metadata["num_questions"])
        self.num_skills = int(data_metadata["num_skills"])
        self.hidden_dim = hidden_dim
        self.max_gap_bins = max_gap_bins
        # Ablation switches. ``use_global=False`` removes the stacked causal
        # dilated-conv branch, ``use_local=False`` removes the per-KC affine
        # recursion branch, and ``use_forgetting=False`` keeps the local branch
        # but makes its temporal transition an identity. The first two together
        # leave only the shared event embedding + IRT readout.
        self.use_global = bool(use_global)
        self.use_local = bool(use_local)
        self.use_forgetting = bool(use_forgetting)

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

        # Scalar per-dimension transition: the gap-modulated decay forgets and
        # the event-conditioned write updates; the segmented scan keeps one
        # private state per KC.
        self.local_write = nn.Linear(hidden_dim, hidden_dim)
        self.local_init = nn.Linear(hidden_dim, hidden_dim)
        self.local_decay = nn.Linear(hidden_dim, hidden_dim)
        self.local_readout = nn.Linear(3 * hidden_dim, 1)
        self.global_blocks = nn.ModuleList(
            GlobalConvBlock(
                d_model=hidden_dim,
                dropout=dropout,
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
        # 3-way readout features by ``ability_head`` and β is the shared
        # ``question_diff`` of the predicted (next) question. Zero-initialized
        # so the model starts at chance (logit 0); the head is a pure
        # 2PL-style IRT decomposition, interpretable as ability minus
        # difficulty scaled by the learned discrimination ``a``.
        self.ability_head = nn.Linear(3 * hidden_dim, 1)
        nn.init.zeros_(self.ability_head.weight)
        nn.init.zeros_(self.ability_head.bias)
        self.irt_disc = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.question_diff.weight)
        zeroed_layers = [self.local_write, self.local_readout]
        for layer in zeroed_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.local_decay.weight)
        nn.init.constant_(self.local_decay.bias, -4.0)

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

        ``question_skill_ids[questions]`` and the ``_question_vector``
        projection are each needed in several sub-methods; computing them here
        and threading the result avoids repeating the gathers. The per-KC
        skill embedding is not materialized: ``_event_embeddings`` pools it
        over the packed occurrences and the readout consumes the packed
        per-occurrence embedding already gathered in the local branch.
        """
        skill_ids = self.question_skill_ids[questions]
        return _QuestionFeatures(
            skill_ids=skill_ids,
            skill_mask=self.question_skill_mask[questions],
            question_vector=self._question_vector(questions),
        )

    def _packed_question_conditioned_readout(
        self,
        packed_state: torch.Tensor,
        packed_skill_embedding: torch.Tensor,
        packed_pos: torch.Tensor,
        packed_valid: torch.Tensor,
        question_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Question-conditioned readout directly in packed space.

        Each occurrence carries the private state of one KC of one position;
        the score combines that state, the KC embedding, and the containing
        question's vector, and a softmax over the occurrences of the same
        position weights the states before summing. The (b, s) groups are
        reduced with ``scatter_reduce`` over the packed position ids, so the
        unpacked [B, S, K, H] layout (and its scatter) is never materialized.
        """
        h = self.hidden_dim
        weight = self.local_readout.weight
        w_local = weight[:, :h]
        w_skill = weight[:, h : 2 * h]
        w_question = weight[:, 2 * h :]
        # Scalar per-feature projections; gathering the projected score avoids
        # a [B, P, H] gather of ``question_vector``.
        question_proj = torch.nn.functional.linear(question_vector, w_question).squeeze(
            -1
        )
        scores = (
            torch.nn.functional.linear(packed_state, w_local).squeeze(-1)
            + torch.nn.functional.linear(packed_skill_embedding, w_skill).squeeze(-1)
            + question_proj.gather(1, packed_pos)
            + self.local_readout.bias
        )
        # Padded occurrences of a row are invalid; exclude them so they cannot
        # join the group of the position their padding slot maps to.
        scores = scores.masked_fill(~packed_valid, torch.finfo(scores.dtype).min)
        seg_max = torch.scatter_reduce(
            torch.zeros_like(scores),
            1,
            packed_pos,
            scores,
            reduce="amax",
            include_self=False,
        )
        weights = torch.exp(scores - seg_max.gather(1, packed_pos))
        seg_sum = torch.scatter_reduce(
            torch.zeros_like(weights),
            1,
            packed_pos,
            weights,
            reduce="sum",
        )
        weights = weights / seg_sum.gather(1, packed_pos)
        weights = torch.where(packed_valid, weights, torch.zeros_like(weights))
        pooled = packed_state * weights.unsqueeze(-1)
        return torch.scatter_reduce(
            pooled.new_zeros(question_vector.size(0), question_vector.size(1), h),
            1,
            packed_pos.unsqueeze(-1).expand(-1, -1, h),
            pooled,
            reduce="sum",
        )

    def _pack_kc_positions(
        self,
        questions: torch.Tensor,
        mask: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_mask: torch.Tensor,
        kc_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Pack valid KC occurrences of each row in skill-major order.

        Returns the packed skill ids, validity mask, and original position of
        every occurrence, plus the permutation that maps the flattened
        [S, K] grid into the packed layout. Shared by the local scan and the
        event-embedding pooling, so the [B, S, K] grid is gathered once per
        use instead of materialized as [B, S, K, hidden].
        """
        batch_size, seq_len = questions.shape
        occurrence_mask = skill_mask & mask.unsqueeze(-1)
        max_skills = skill_ids.size(-1)
        flat_skill = skill_ids.flatten(1)
        flat_valid = occurrence_mask.flatten(1)
        positions = (
            torch.arange(seq_len, device=questions.device)
            .view(1, seq_len, 1)
            .expand(batch_size, seq_len, max_skills)
        )
        flat_pos = positions.flatten(1)

        if kc_order is None:
            # Direct model calls retain the original dynamic-sort fallback.
            invalid_key = (self.num_skills + 1) * (seq_len + 1)
            sort_key = flat_skill * (seq_len + 1) + flat_pos
            sort_key = torch.where(flat_valid, sort_key, invalid_key + flat_pos)
            order = torch.argsort(sort_key, dim=1, stable=True)
            packed_length = int(flat_valid.sum(dim=1).max().item())
            order = order[:, :packed_length]
        else:
            if kc_order.ndim != 2 or kc_order.size(0) != batch_size:
                raise ValueError("kc_order must have shape [batch_size, packed_length]")
            if kc_order.size(1) > flat_skill.size(1):
                raise ValueError("kc_order width exceeds the flattened KC width")
            if kc_order.device != questions.device:
                raise ValueError("kc_order must be on the same device as questions")
            order = kc_order.long()

        def gather(values: torch.Tensor) -> torch.Tensor:
            return torch.gather(values, 1, order)

        return gather(flat_skill), gather(flat_valid), gather(flat_pos), order

    def _pack_kc_occurrences(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        skill_ids: torch.Tensor | None = None,
        skill_mask: torch.Tensor | None = None,
        kc_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, seq_len = questions.shape
        if skill_ids is None:
            skill_ids = self.question_skill_ids[questions]
        if skill_mask is None:
            skill_mask = self.question_skill_mask[questions]
        max_skills = skill_ids.size(-1)
        (
            packed_skill,
            packed_valid,
            packed_pos,
            order,
        ) = self._pack_kc_positions(questions, mask, skill_ids, skill_mask, kc_order)

        # Real elapsed seconds (or position indices when unavailable); only
        # differences within a sequence matter.
        times = times.reshape(batch_size, seq_len, 1).expand(
            batch_size, seq_len, max_skills
        )
        question_occ = questions.unsqueeze(-1).expand_as(skill_ids)
        response_occ = responses.unsqueeze(-1).expand_as(skill_ids)

        def gather(values: torch.Tensor) -> torch.Tensor:
            return torch.gather(values, 1, order)

        return (
            packed_skill,
            gather(times.flatten(1)),
            gather(question_occ.flatten(1)),
            gather(response_occ.flatten(1)),
            packed_valid,
            order,
            packed_pos,
        )

    def _scan_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        q_features: _QuestionFeatures | None = None,
        kc_order: torch.Tensor | None = None,
        packed: tuple[torch.Tensor, ...] | None = None,
        skill_embedding: torch.Tensor | None = None,
        skill_change_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the gap/decay/write pipeline and the segmented scan.

        Returns each occurrence's exclusive scan state — the KC memory
        accumulated through that KC's previous occurrence, decayed between
        past practices only — together with the shared packed tensors, so the
        readout and the event pooling can reuse one packing pass and one
        skill-embedding gather. The occurrence's own transition decay is not
        applied to its own readout (it composes only into later reads of the
        same KC), so the timestamp of the interaction being predicted never
        enters its prediction.
        """
        if packed is None:
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
            packed = self._pack_kc_occurrences(
                questions,
                responses,
                times,
                mask,
                skill_ids,
                skill_mask,
                kc_order=kc_order,
            )
        (
            packed_skill,
            packed_time,
            packed_question,
            packed_response,
            packed_valid,
            order,
            packed_pos,
        ) = packed

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

        if skill_embedding is None:
            skill_embedding = self.skill_embed(packed_skill)
        if skill_change_embedding is None:
            skill_change_embedding = self.skill_change(packed_skill)
        local_input = (
            self._question_vector(packed_question)
            + skill_embedding
            + self.question_diff(packed_question) * skill_change_embedding
            + self.answer_embed(packed_response)
        )
        if self.use_forgetting:
            # The decay only takes one of max_gap_bins rows, so it enters the
            # scan as a codebook plus per-position codes instead of a full
            # [B, N, H] matrix; the kernel neutralizes invalid positions.
            decay_table = torch.exp(
                -torch.nn.functional.softplus(self.local_decay(self.gap_embed.weight))
            )
            decay_codes = gap_bucket
        else:
            # Preserve the local writes and per-KC segmentation while removing
            # only the temporal forgetting transition for a clean ablation.
            decay_table = torch.ones(
                1,
                self.hidden_dim,
                device=packed_skill.device,
                dtype=skill_embedding.dtype,
            )
            decay_codes = torch.zeros_like(gap_bucket)
        # The initial state only depends on the skill id, and the packed
        # segment ids are exactly those skill ids, so it enters the scan as a
        # per-skill codebook addressed by the segment ids.
        init_table = torch.tanh(self.local_init(self.skill_embed.weight))
        bias = torch.tanh(self.local_write(local_input)).unsqueeze(-1)

        packed_state_blocks = segmented_scalar_affine_exclusive_scan(
            None,
            bias,
            packed_skill,
            packed_valid,
            None,
            matrix_table=decay_table,
            matrix_codes=decay_codes,
            init_table=init_table,
            init_codes=packed_skill,
            post_multiply=False,
        )
        packed_state = packed_state_blocks.squeeze(-1)
        return (
            packed_state,
            skill_embedding,
            packed_skill,
            packed_valid,
            packed_pos,
            order,
        )

    def _local_pre_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        q_features: _QuestionFeatures | None = None,
        kc_order: torch.Tensor | None = None,
        packed: tuple[torch.Tensor, ...] | None = None,
        skill_embedding: torch.Tensor | None = None,
        skill_change_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = questions.shape
        if not self.use_local:
            return torch.zeros(
                batch_size,
                seq_len,
                self.hidden_dim,
                device=questions.device,
                dtype=self.skill_embed.weight.dtype,
            )
        (
            packed_state,
            skill_embedding,
            _packed_skill,
            packed_valid,
            packed_pos,
            _order,
        ) = self._scan_states(
            questions,
            responses,
            times,
            mask,
            q_features,
            kc_order,
            packed=packed,
            skill_embedding=skill_embedding,
            skill_change_embedding=skill_change_embedding,
        )
        question_vector = (
            q_features.question_vector
            if q_features is not None
            else self._question_vector(questions)
        )
        return self._packed_question_conditioned_readout(
            packed_state,
            skill_embedding,
            packed_pos,
            packed_valid,
            question_vector,
        )

    def _event_embeddings(
        self,
        questions: torch.Tensor,
        q_features: _QuestionFeatures | None = None,
        mask: torch.Tensor | None = None,
        kc_order: torch.Tensor | None = None,
        packed: tuple[torch.Tensor, ...] | None = None,
        skill_embedding: torch.Tensor | None = None,
        skill_change_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-position event embedding with packed-occurrence KC pooling.

        The per-KC skill and change embeddings are gathered only for the
        packed occurrences and reduced over each position's group with
        ``scatter_reduce``, so no [B, S, K, hidden] tensor is materialized.
        Without ``mask`` every position is treated as valid (all-of-batch
        pooling, used by direct calls); with it, masked positions pool to
        zero, which the global branch zeroes out anyway.
        """
        if q_features is None:
            q_features = self._resolve_question_features(questions)
        if mask is None:
            mask = torch.ones(
                questions.shape, dtype=torch.bool, device=questions.device
            )
        if packed is not None:
            packed_skill, packed_valid, packed_pos = packed[0], packed[4], packed[6]
        else:
            (packed_skill, packed_valid, packed_pos) = self._pack_kc_positions(
                questions,
                mask.bool(),
                q_features.skill_ids,
                q_features.skill_mask,
                kc_order,
            )[:3]
        if skill_embedding is None:
            skill_embedding = self.skill_embed(packed_skill)
        if skill_change_embedding is None:
            skill_change_embedding = self.skill_change(packed_skill)
        counts = torch.scatter_reduce(
            torch.zeros(
                questions.shape[0], questions.shape[1], device=questions.device
            ),
            1,
            packed_pos,
            packed_valid.float(),
            reduce="sum",
        )
        denom = counts.clamp_min(1.0).unsqueeze(-1)
        pos_index = packed_pos.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        pooled_skill = (
            torch.scatter_reduce(
                packed_valid.new_zeros(
                    questions.shape[0], questions.shape[1], self.hidden_dim
                ).float(),
                1,
                pos_index,
                skill_embedding,
                reduce="sum",
            )
            / denom
        )
        pooled_change = (
            torch.scatter_reduce(
                packed_valid.new_zeros(
                    questions.shape[0], questions.shape[1], self.hidden_dim
                ).float(),
                1,
                pos_index,
                skill_change_embedding,
                reduce="sum",
            )
            / denom
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
        if not self.use_global:
            return torch.zeros_like(event_embedding)
        global_state = event_embedding + self.answer_embed(responses)
        global_state = global_state.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.global_blocks:
            global_state = block(global_state)
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
        kc_order: torch.Tensor | None = None,
        kc_inverse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-item logits where output[t] predicts response[t+1].

        ``times`` holds per-position interaction times in seconds; only
        within-sequence differences are used, so any consistent offset works.
        The timestamp of a position never enters the prediction of that
        position's response: gap decays compose only into later reads of the
        same KC. ``kc_inverse`` is the dataset-precomputed inverse of
        ``kc_order`` over the full flat slot domain; when omitted the fused
        inference path rebuilds it on the fly.
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
        if not torch.is_grad_enabled() and self.use_local:
            # Fused inference path: one Triton kernel produces both the local
            # readout and the event-embedding pooling; training keeps the aten
            # scatter chain so gradient semantics stay untouched.
            from model.AxisKT.packed_readout import fused_readout_event

            (
                packed_state,
                skill_embedding,
                packed_skill,
                packed_valid,
                _packed_pos,
                order,
            ) = self._scan_states(
                questions, responses, times, mask, q_features, kc_order
            )
            skill_change_embedding = self.skill_change(packed_skill)
            question_vector = q_features.question_vector
            if kc_inverse is None:
                kc_inverse = torch.zeros(
                    questions.shape[0],
                    questions.shape[1] * self.question_skill_ids.size(1),
                    dtype=torch.long,
                    device=questions.device,
                )
                kc_inverse.scatter_(
                    1,
                    order,
                    torch.arange(order.size(1), device=order.device).expand_as(order),
                )
            slot_valid = (q_features.skill_mask & mask.unsqueeze(-1)).flatten(1)
            local_pre_state, event_embedding = fused_readout_event(
                packed_state,
                skill_embedding,
                skill_change_embedding,
                question_vector,
                questions,
                self.question_diff.weight,
                slot_valid,
                kc_inverse,
                self.question_skill_ids.size(1),
                self.local_readout.weight,
                self.local_readout.bias,
            )
        else:
            # One packing pass and one embedding gather shared by both branches.
            packed = self._pack_kc_occurrences(
                questions,
                responses,
                times,
                mask,
                q_features.skill_ids,
                q_features.skill_mask,
                kc_order=kc_order,
            )
            skill_embedding = self.skill_embed(packed[0])
            skill_change_embedding = self.skill_change(packed[0])
            event_embedding = self._event_embeddings(
                questions,
                q_features,
                mask=mask,
                kc_order=kc_order,
                packed=packed,
                skill_embedding=skill_embedding,
                skill_change_embedding=skill_change_embedding,
            )
            local_pre_state = self._local_pre_states(
                questions,
                responses,
                times,
                mask,
                q_features,
                kc_order=kc_order,
                packed=packed,
                skill_embedding=skill_embedding,
                skill_change_embedding=skill_change_embedding,
            )
        global_state = self._global_history_states(event_embedding, responses, mask)

        features = torch.cat(
            [
                global_state[:, :-1],
                local_pre_state[:, 1:],
                event_embedding[:, 1:],
            ],
            dim=-1,
        )
        next_questions = questions[:, 1:]
        logits = self._irt_term(features, next_questions)
        logits = logits.masked_fill(~mask[:, 1:], 0.0)
        return torch.cat([logits, logits.new_zeros(logits.size(0), 1)], dim=1)


__all__ = ["AxisKT"]
