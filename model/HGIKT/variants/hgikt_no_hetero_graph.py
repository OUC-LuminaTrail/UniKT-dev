"""HGIKT variant without heterogeneous graph branch."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.nn import HGNNConv
from torch_geometric.nn import Linear

from model.layers import GeneralInteraction, HistoryRecap
from utils.core import register_model


class HyperGNN(nn.Module):
    """Based on HGIKT HyperGNN module."""

    def __init__(
        self,
        in_ch: int,
        n_hid: int,
        n_class: int,
        dropout: float = 0.0,
        use_edge_weights: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge_weights = use_edge_weights

        self.hgc1 = HGNNConv(
            in_ch, n_hid, bias=True, use_bn=False, drop_rate=dropout, is_last=False
        )
        self.hgc2 = HGNNConv(
            n_hid, n_class, bias=True, use_bn=False, drop_rate=dropout, is_last=True
        )

    def forward(self, x: torch.Tensor, hg: Any) -> torch.Tensor:
        """Forward pass."""
        x1 = self.hgc1(x, hg)
        x2 = F.relu(self.hgc2(x1, hg))
        return x2


@register_model("HGIKT_NoHeteroGraph")
class HGIKT_NoHeteroGraph(nn.Module):
    """HGIKT variant with heterogeneous graph branch disabled.

    This variant uses only the hypergraph for knowledge representation.
    The heterogeneous graph convolution is replaced with zeros.
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.data_metadata = data_metadata

        self.hidden_dim = args.hidden_dim
        self.lstm_layers = args.lstm_layers
        self.dropout = args.dropout

        self.question_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.question_embedding_hyper = nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.skill_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_skills"],
            embedding_dim=self.hidden_dim,
        )
        self.assignment_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_assignments"],
            embedding_dim=self.hidden_dim,
        )
        self.template_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_templates"],
            embedding_dim=self.hidden_dim,
        )
        self.answer_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.hidden_dim,
        )
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        # Hetero graph module disabled (no hetero_conv)

        # Hypergraph module (unchanged)
        self.hgnn_conv = HyperGNN(
            in_ch=self.hidden_dim,
            n_hid=self.hidden_dim,
            n_class=self.hidden_dim,
            dropout=self.dropout,
        )

        # Fusion module disabled

        self.fc_exercise = Linear(
            self.hidden_dim * 2, self.hidden_dim, weight_initializer="uniform"
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.history_review = HistoryRecap(
            hist_neighbor_num=args.history_neighbour,
            att_bound=args.att_bound,
        )

        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

    def forward(
        self,
        user_sequence: torch.Tensor,
        user_response: torch.Tensor,
        user_mask: torch.Tensor,
        hetero_graph: Any,
        hypergraph: Any,
        question_skill_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass without heterogeneous graph computation.

        Key differences from parent HGIKT:
        1. Skip hetero_conv computation
        2. Skip fusion (use only hypergraph features)
        """
        B, _ = user_sequence.size()

        answers_embedding = self.answer_embedding(user_response)

        # Hypergraph convolution
        question_hyper_conv: torch.Tensor = self.hgnn_conv(
            self.question_embedding_hyper.weight, hypergraph
        )

        # === MODIFIED: Skip hetero graph convolution, use hypergraph only ===
        question_conv_fused = question_hyper_conv

        # Get skill embeddings for knowledge_status (need hetero graph features)
        # Since we skip hetero_conv, use skill_embedding directly
        skill_hetero_conv = self.skill_embedding.weight

        question_embedding_sequence = question_conv_fused[user_sequence]
        exercise_emb = torch.cat(
            [question_embedding_sequence, answers_embedding], dim=-1
        )
        exercise_emb = F.relu(self.fc_exercise(exercise_emb))
        exercise_emb = self.embedding_dropout(exercise_emb)

        lstm_output, _ = self.lstm(exercise_emb)

        next_user_sequence = torch.zeros_like(user_sequence)
        if user_sequence.size(1) > 1:
            next_user_sequence[:, :-1] = user_sequence[:, 1:]
            next_user_sequence[:, -1] = 0

        next_question_embedding = question_conv_fused[next_user_sequence]

        history_question_neighbors = self.history_review(
            question_embedding_sequence,
            next_question_embedding,
            exercise_emb,
            user_mask,
        )

        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )

        q_skill_vectors = question_skill_matrix[next_user_sequence]
        sorted_skill_indices = torch.argsort(q_skill_vectors, dim=-1, descending=True)
        max_skills_per_question = int(q_skill_vectors.sum(dim=-1).max().item())
        skill_counts = q_skill_vectors.sum(dim=-1).long()

        related_skill_ids = sorted_skill_indices[..., :max_skills_per_question].clone()

        device = next_user_sequence.device
        pos = torch.arange(max_skills_per_question, device=device).view(1, 1, -1)
        valid_pos_mask = pos < skill_counts.unsqueeze(-1)

        padding_index = skill_hetero_conv.size(0)
        padding_ids = torch.full_like(related_skill_ids, padding_index)
        related_skill_ids = torch.where(valid_pos_mask, related_skill_ids, padding_ids)

        skill_conv_padded = torch.cat(
            [
                skill_hetero_conv,
                torch.zeros(
                    1, self.hidden_dim, device=device, dtype=skill_hetero_conv.dtype
                ),
            ],
            dim=0,
        )

        related_skill_embs = skill_conv_padded[related_skill_ids]

        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs], dim=2
        )

        logits = self.general_interaction(student_status, knowledge_status, user_mask)

        return logits
