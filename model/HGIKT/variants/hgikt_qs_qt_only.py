"""HGIKT variant with QS+QT heterogeneous graph branch only."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

from model.layers import GeneralInteraction, HistoryRecap
from utils.core import register_model


class HeteroGNN(nn.Module):
    """Based on HGIKT HeteroGNN module."""

    def __init__(
        self,
        embedding_dim: int,
        n_hop: int,
        heads: int,
        dropout: float,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
    ) -> None:
        super().__init__()
        self.n_hop = n_hop
        self.heads = heads
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for _ in range(n_hop):
            conv = HGTConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                metadata=metadata,
                heads=heads,
            )
            self.convs.append(conv)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass."""
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            new_x_dict = {}
            for node_type, x in x_dict.items():
                if x is not None:
                    x = F.gelu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[node_type] = x
            x_dict = new_x_dict
        return x_dict


@register_model("HGIKT_QS_QT_Only")
class HGIKT_QS_QT_Only(nn.Module):
    """HGIKT variant with only Question-Skill (QS) and Question-Template (QT) edges.

    This variant uses a reduced heterogeneous graph.
    The hypergraph branch is removed.
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

        # Model parameters
        self.hidden_dim = args.hidden_dim
        self.lstm_layers = args.lstm_layers
        self.dropout = args.dropout

        # Embedding layers
        self.question_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.skill_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_skills"],
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

        # Hetero graph module
        self.hetero_conv = HeteroGNN(
            embedding_dim=self.hidden_dim,
            n_hop=args.n_hop,
            heads=args.heads,
            dropout=self.dropout,
            metadata=hetero_metadata,
        )

        # Full connected layer
        self.fc_exercise = Linear(
            self.hidden_dim * 2, self.hidden_dim, weight_initializer="uniform"
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # History recap module
        self.history_review = HistoryRecap(
            hist_neighbor_num=args.history_neighbour,
            att_bound=args.att_bound,
        )

        # General interaction module
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
        B, _ = user_sequence.size()
        answers_embedding = self.answer_embedding(user_response)

        # Hetero graph convolution
        x_dict = {
            "question": self.question_embedding.weight,
            "skill": self.skill_embedding.weight,
            "template": self.template_embedding.weight,
        }
        conv = self.hetero_conv(x_dict, hetero_graph.edge_index_dict)
        question_hetero_conv = conv["question"]
        skill_hetero_conv = conv["skill"]

        question_embedding_sequence = question_hetero_conv[user_sequence]
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

        next_question_embedding = question_hetero_conv[next_user_sequence]

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

        return self.general_interaction(student_status, knowledge_status, user_mask)
