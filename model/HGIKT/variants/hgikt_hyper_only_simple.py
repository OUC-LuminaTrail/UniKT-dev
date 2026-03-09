"""HGIKT variant with hypergraph branch only (simple, no difficulty weighting)."""

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


@register_model("HGIKT_HyperOnlySimple")
class HGIKT_HyperOnlySimple(nn.Module):
    """HGIKT variant with only simple hypergraph for knowledge representation.

    This variant uses only the simple hypergraph (no difficulty weighting)
    for knowledge representation. The heterogeneous graph branch is removed,
    and MoE fusion is bypassed.
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

        # Only keep question_embedding_hyper for hypergraph
        self.question_embedding_hyper = nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.answer_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.hidden_dim,
        )
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        # Hetero graph module disabled (no hetero_conv, no skill/assignment/template embeddings)

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
        hetero_graph: Any,  # Unused
        hypergraph: Any,
        question_skill_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with only simple hypergraph.

        Key differences from parent HGIKT:
        1. Skip hetero_conv computation
        2. Skip fusion (use only hypergraph features)
        3. Use skill_embedding directly for knowledge_status
        """
        B, _ = user_sequence.size()

        answers_embedding = self.answer_embedding(user_response)

        # Hypergraph convolution
        question_hyper_conv: torch.Tensor = self.hgnn_conv(
            self.question_embedding_hyper.weight, hypergraph
        )

        # Use only hypergraph features
        question_conv_fused = question_hyper_conv

        # Since we skip hetero_conv, we need skill embeddings for knowledge_status
        # Use a dummy skill embedding (zeros) since this is hypergraph-only
        skill_hetero_conv = torch.zeros(
            self.data_metadata["num_skills"], self.hidden_dim, device=user_sequence.device
        )

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

        # For skill-related knowledge status, use zero padding since hetero graph is disabled
        q_skill_vectors = question_skill_matrix[next_user_sequence]
        max_skills_per_question = int(q_skill_vectors.sum(dim=-1).max().item())
        skill_counts = q_skill_vectors.sum(dim=-1).long()

        related_skill_ids = torch.arange(max_skills_per_question, device=user_sequence.device)
        related_skill_ids = related_skill_ids.view(1, 1, -1).expand(B, user_sequence.size(1), -1).clone()

        device = user_sequence.device
        skill_conv_padded = torch.cat(
            [
                skill_hetero_conv,
                torch.zeros(1, self.hidden_dim, device=device),
            ],
            dim=0,
        )

        related_skill_embs = skill_conv_padded[related_skill_ids]
        related_skill_embs = torch.zeros(
            B, user_sequence.size(1), max_skills_per_question, self.hidden_dim, device=device
        )

        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs], dim=2
        )

        logits = self.general_interaction(student_status, knowledge_status, user_mask)

        return logits
