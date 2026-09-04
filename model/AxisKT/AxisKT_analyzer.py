"""AxisKT Case Analyzer.

Provides inference-only capabilities for AxisKT case analysis. The knowledge
state of AxisKT is the per-KC private state of the local affine-recursion
branch: each skill owns an exclusive-scan state that only its own events
update. The analyzer captures those states during the forward pass (by
temporarily wrapping :meth:`AxisKT._scan_states` and hooking the
``local_write`` module — the shared pipeline both execution paths use,
mirroring the memory-probe adapters).

Per-skill mastery is not the readout's attention projection (that moves
*against* mastery: the model attends more to recently-missed KCs); it is the
mean over every question covering the skill of the model's IRT head applied
to ``[per-KC standing state, question event embedding]`` — without the
global conv state, which changes at every column and would make a frozen
skill's curve wiggle (sawtooth). Averaging over all associated questions
(instead of a single probe question) cancels per-question noise. A wrong
answer then lowers mastery and a correct answer raises it, matching the
model's behavior.

Alignment convention: row ``r`` of the collected DataFrame predicts
``response[r + 1]``; the exported knowledge state is the state standing
*after* event ``r + 1`` (i.e. after events ``<= r + 1``) — the post-event
per-skill mastery the heatmap displays (the state that produced the
prediction is the previous column).
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.case_analysis.base_analyzer import BaseCaseAnalyzer
from utils.core import get_logger, register_analyzer
from utils.data_process import DataSource
from utils.training.runtime_components import RuntimeComponents

from .AxisKT_data import AxisKTDataset, AxisKTModelData, axiskt_packed_collate_fn
from .AxisKT_model import AxisKT

logger = get_logger(__name__)


class _UserAwareAxisKTDataset(Dataset):
    """AxisKTDataset wrapper that annotates each row with its user id.

    Split rows are subsequences (``sequence_id``), not users, and the split
    compresses each fold's rows — so the dataset index is not a user id.
    ``user_ids`` carries the real student id per row (from the parquet's
    ``user`` column, split-aligned in :meth:`AxisKTModelData.prepare_data`).
    The extra trailing element is transparent to
    :func:`axiskt_packed_collate_fn`.
    """

    def __init__(self, dataset: AxisKTDataset, user_ids: np.ndarray):
        self.dataset = dataset
        self.user_ids = np.asarray(user_ids)

    def __getitem__(self, index: int):
        return (*self.dataset[index], int(self.user_ids[index]))

    def __len__(self) -> int:
        return len(self.dataset)


def _analyzer_collate_fn(batch):
    """Collate a batch, prepending the stacked user ids."""
    users = torch.tensor([row[-1] for row in batch], dtype=torch.long)
    return (users, *axiskt_packed_collate_fn(batch))


@register_analyzer("AxisKT")
class AxisKTAnalyzer(BaseCaseAnalyzer):
    """AxisKT-specific case analyzer for inference and visualization."""

    def __init__(self, rc, data_src: DataSource, checkpoint_path: str, **kwargs):
        """Initialize AxisKT analyzer.

        Args:
            rc: RunConfig (OmegaConf DictConfig)
            data_src: Data source instance
            checkpoint_path: Path to model checkpoint
            **kwargs: Forwarded to ``BaseCaseAnalyzer`` (sink, device,
                batch_size).
        """
        super().__init__(rc, data_src, checkpoint_path, **kwargs)

    def build_components(self, rc, data_src: DataSource) -> RuntimeComponents:
        """Assemble the AxisKT model and user-annotated val dataset."""
        model_data = AxisKTModelData(data_src)
        _, val_data, _, extra = model_data.prepare_data(rc)

        self.num_questions = data_src.get_metadata("num_questions")
        self.num_skills = data_src.get_metadata("num_skills")
        # CPU copies of the question-KC table for skill extraction.
        self.question_skill_ids_np = np.asarray(extra["question_skill_ids"])
        self.question_skill_mask_np = np.asarray(extra["question_skill_mask"])
        # Every question covering each skill, used for the mean mastery
        # readout over all associated questions.
        self._skill_questions = self._build_skill_questions()

        m = rc.model
        model = AxisKT(
            data_metadata=data_src.get_metadata(),
            question_skill_ids=extra["question_skill_ids"],
            question_skill_mask=extra["question_skill_mask"],
            hidden_dim=m.hidden_dim,
            n_blocks=m.n_blocks,
            max_gap_bins=int(extra["max_gap_bins"]),
            dropout=m.dropout,
            conv_kernel_size=m.conv_kernel_size,
            conv_dilation_base=m.conv_dilation_base,
            question_embed_dim=(
                None if m.question_embed_dim < 0 else m.question_embed_dim
            ),
            use_global=m.use_global,
            use_local=m.use_local,
            # Older archived run configs predate this ablation flag.
            use_forgetting=getattr(m, "use_forgetting", True),
        )

        val_dataset = _UserAwareAxisKTDataset(val_data, extra["user_ids"]["val"])
        return RuntimeComponents(
            model=model, val_data=val_dataset, collate_fn=_analyzer_collate_fn
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """AxisKT forward pass for inference.

        Args:
            batch_data: Tuple of (users, questions, responses, times, mask,
                kc_order, valid_idx)

        Returns:
            Dictionary with y_hat (flattened logits), y_label (flattened),
            y_predict, and knowledge_states [B, S, num_skills] per-skill
            mastery pseudo-probabilities.
        """
        users, questions, responses, times, mask, kc_order, _, valid_idx = batch_data
        questions = self._move_tensor_to_device(questions)
        responses = self._move_tensor_to_device(responses)
        times = self._move_tensor_to_device(times)
        mask = self._move_tensor_to_device(mask)
        kc_order = self._move_tensor_to_device(kc_order)
        valid_idx = self._move_tensor_to_device(valid_idx)

        logits_full, captured = self._forward_with_captured_states(
            questions, responses, times, mask, kc_order
        )

        logits = logits_full[:, :-1].flatten()[valid_idx]
        labels = responses.float()[:, 1:].flatten()[valid_idx]
        logits, labels = self._handle_empty_batch(logits, labels)

        knowledge_states = self._per_skill_mastery(
            captured, questions.size(0), questions.size(1)
        )

        return {
            "y_hat": logits,
            "y_label": labels,
            "y_predict": self._generate_binary_predictions(logits, threshold=0.0),
            "knowledge_states": knowledge_states,
        }

    def _forward_with_captured_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        kc_order: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the model forward, capturing per-KC states and features.

        Captures the packed per-KC scan outputs (local pre-states and the
        occurrence layout) by wrapping :meth:`AxisKT._scan_states` — the
        shared pipeline both the training scatter chain and the fused
        no-grad inference path go through — plus the local write ``bias``
        via a forward hook on ``local_write``. The patches are installed
        only around this call and removed in a ``finally`` block, so the
        analyzer never mutates the model's methods outside the forward
        pass.
        """
        captured: dict[str, torch.Tensor] = {}
        model = self.model
        orig_scan = model._scan_states
        write_hook = model.local_write.register_forward_hook(
            lambda mod, inp, out: captured.__setitem__(
                "local_write_out", out.detach().float()
            )
        )

        def hooked_scan(q, r, t, m, q_features=None, kc_order=None, **kwargs):
            out = orig_scan(q, r, t, m, q_features, kc_order, **kwargs)
            captured["packed_state"] = out[0].detach().float()
            captured["packed_skill"] = out[2].detach()
            captured["packed_valid"] = out[3].detach()
            captured["packed_pos"] = out[4].detach()
            return out

        model._scan_states = hooked_scan
        try:
            use_amp = bool(self.run_config.model.amp)
            with torch.autocast(
                device_type=self.device_.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits_full = model(
                    questions, responses, times, mask, kc_order=kc_order
                )
            if use_amp:
                logits_full = logits_full.float()
        finally:
            model._scan_states = orig_scan
            write_hook.remove()
        return logits_full, captured

    def _build_skill_questions(self) -> list[np.ndarray]:
        """Return one list of question ids per skill: every question covering it.

        A question covers a skill when the skill is listed among the
        question's non-masked KC slots.
        """
        ids = self.question_skill_ids_np
        mask = self.question_skill_mask_np
        return [
            np.asarray(
                [q for q in range(len(ids)) if k in ids[q][mask[q]].tolist()],
                dtype=np.int64,
            )
            for k in range(self.num_skills)
        ]

    def _per_skill_mastery(
        self,
        captured: dict[str, torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Model-predicted per-skill mastery at every position.

        Returns:
            Tensor [B, S, num_skills] in [0, 1]: ``mastery[b, t, k]`` is the
            mean over every question covering skill ``k`` of the model's IRT
            head applied to ``[standing state of k after events <= t,
            question event embedding]``. The global conv state is excluded:
            it changes at every column, so including it would make a frozen
            skill's curve wiggle (sawtooth) between its own events. Columns
            before a skill's first occurrence use the skill's learned initial
            state, so every cell is a model prediction (NaN only when the
            local branch is ablated).
        """
        model = self.model
        device = self.device_
        if not model.use_local or "packed_state" not in captured:
            logger.warning(
                "Local branch ablated or not executed; returning NaN knowledge states"
            )
            return torch.full(
                (batch_size, seq_len, self.num_skills),
                float("nan"),
                device=device,
            )

        h = model.hidden_dim
        ability = model.ability_head
        w = ability.weight[0]  # [3h]
        bias = ability.bias  # scalar; included to match the model's own head
        packed_skill = captured["packed_skill"]
        packed_pos = captured["packed_pos"]
        packed_valid = captured["packed_valid"]
        # Post-write standing state: the scan state after the event, before
        # the gap decay of the next occurrence. The packed state captured
        # from the readout is the decayed pre-state; adding the write bias
        # (tanh of the captured linear output) yields the standing state.
        standing = captured["packed_state"] + torch.tanh(captured["local_write_out"])
        # Per-skill initial state (the exclusive scan's segment start) for
        # columns before the skill's first occurrence.
        skill_range = torch.arange(self.num_skills, device=device)
        init_states = torch.tanh(model.local_init(model.skill_embed(skill_range)))

        mastery = torch.full(
            (batch_size, seq_len, self.num_skills), float("nan"), device=device
        )
        rows = torch.arange(batch_size, device=device)
        for k in range(self.num_skills):
            q_ids = torch.as_tensor(self._skill_questions[k], device=device)
            if q_ids.numel() == 0:
                continue
            occ = packed_valid & (packed_skill == k)
            b_idx, p_idx = occ.nonzero(as_tuple=True)
            positions = packed_pos[b_idx, p_idx]

            # Per-column standing state of skill k: scatter the post-write
            # states at their occurrence positions, then forward-fill along
            # time (cummax of the last occurrence position per row). Columns
            # before the first occurrence carry the skill's learned initial
            # state (the exclusive scan's segment start), so every cell is a
            # model prediction — never NaN.
            state = torch.full((batch_size, seq_len, h), float("nan"), device=device)
            state[b_idx, positions] = standing[b_idx, p_idx]
            marker = torch.where(
                torch.isfinite(state[:, :, 0]),
                torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1),
                torch.full((batch_size, seq_len), -1, device=device),
            )
            last = torch.cummax(marker, dim=1).values
            init_k = init_states[k].view(1, 1, h).expand(batch_size, seq_len, h)
            state = torch.where(
                (last >= 0).unsqueeze(-1),
                state[
                    rows.unsqueeze(1).expand(batch_size, seq_len),
                    last.clamp_min(0),
                ],
                init_k,
            )

            # The head is linear, so theta splits as W_s . s + W_e . e + b:
            # the per-column part comes only from the skill's standing state,
            # and the per-question embedding logit differs per question.
            # (The W_g . g global term is deliberately dropped: see the
            # docstring.)
            base = torch.nn.functional.linear(state, w[h : 2 * h].view(1, h)).squeeze(
                -1
            )

            # Question event embeddings and difficulties for skill k; the
            # skill embedding is k's own, as in the model's event encoder.
            skill_k = torch.tensor(k, device=device)
            emb_k = (
                model._question_vector(q_ids)
                + model.skill_embed(skill_k).unsqueeze(0)
                + model.question_diff(q_ids) * model.skill_change(skill_k).unsqueeze(0)
            )  # [Q_k, H]
            beta_k = model.question_diff(q_ids).squeeze(-1)  # [Q_k]
            emb_logit = (
                torch.nn.functional.linear(emb_k, w[2 * h :].view(1, h)).squeeze(-1)
                + bias
            )  # [Q_k]

            # Mean P(correct) over the skill's questions, chunked to bound
            # the [B, S, Q_k] working memory.
            chunk = 256
            mean_parts = []
            for start in range(0, q_ids.numel(), chunk):
                el = emb_logit[start : start + chunk]
                bt = beta_k[start : start + chunk]
                logit = model.irt_disc * (
                    base.unsqueeze(-1) + el.view(1, 1, -1) - bt.view(1, 1, -1)
                )
                mean_parts.append(torch.sigmoid(logit).mean(dim=-1))
            mastery[:, :, k] = torch.stack(mean_parts).mean(dim=0)
        return mastery

    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract case data from batch outputs.

        Args:
            batch_data: Raw batch tuple (users, questions, responses, times,
                mask, kc_order, valid_idx)
            outputs: Output dict from forward_pass

        Returns:
            Dictionary with extracted data (all flattened to valid positions)
        """
        users, questions, responses, times, mask, kc_order, _, _ = batch_data

        batch_size, seq_len = questions.shape
        # Valid positions are the S-grid counterparts of the adjacent-pair
        # mask the collate fn used: pair (t, t+1) -> position t+1, which is
        # where ``y_hat``/``y_label`` rows live. The per-column mastery is
        # indexed by "after events <= t", so the knowledge state for the row
        # predicting response[t+1] is mastery at position t+1 — the state
        # standing after the event (post-event mastery, as displayed by the
        # heatmap).
        masks_bool = mask.bool()
        adjacent = masks_bool[:, :-1] & masks_bool[:, 1:]
        valid_pair = adjacent.view(-1).nonzero(as_tuple=True)[0]
        row_idx = valid_pair // (seq_len - 1)
        t = valid_pair % (seq_len - 1)
        flat_s = row_idx * seq_len + t + 1  # the predicted event
        # Post-event knowledge state: mastery at position t+1 (the state
        # after events <= t+1, i.e. after the row's own event). This is the
        # state the heatmap displays per skill occurrence; the column after
        # the last valid pair is in range because mastery spans 0..S-1.
        flat_t = row_idx * seq_len + t + 1

        question_ids_flat = questions.view(-1)[flat_s].cpu().numpy()
        user_ids_flat = users[row_idx].cpu().numpy()

        knowledge_states = outputs["knowledge_states"]
        num_skills = knowledge_states.shape[-1]
        knowledge_states_flat = (
            knowledge_states.view(-1, num_skills)[flat_t].cpu().numpy()
        )

        return {
            "user_ids": user_ids_flat,
            "question_ids": question_ids_flat,
            "skills": self._get_all_skills_for_questions(question_ids_flat),
            "labels": outputs["y_label"].cpu().numpy(),
            "predictions": outputs["y_predict"].cpu().numpy(),
            "logits": outputs["y_hat"].cpu().numpy(),
            "mask": masks_bool.view(-1)[flat_s].cpu().numpy(),
            "knowledge_states": knowledge_states_flat,
        }

    def _get_all_skills_for_questions(
        self, question_ids: np.ndarray
    ) -> list[list[int]]:
        """Get all skills for each question, returning as list of lists.

        Args:
            question_ids: Array of question IDs

        Returns:
            List of lists, where each inner list contains all skill IDs for
            that question. Returns [0] if no skills found.
        """
        skills_list = []
        for q_id in question_ids:
            slots = np.where(self.question_skill_mask_np[q_id])[0]
            question_skills = self.question_skill_ids_np[q_id][slots].tolist()
            skills_list.append(question_skills if question_skills else [0])
        return skills_list


__all__ = ["AxisKTAnalyzer"]
